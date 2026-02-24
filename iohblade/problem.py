import inspect
import json
import logging
import multiprocessing
import os
import select
import shutil
import struct
import subprocess
import tempfile
import traceback
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import cloudpickle
import numpy as np

logger = logging.getLogger(__name__)

# Standard packages installed in every evaluation environment
BASE_DEPENDENCIES = [
    "numpy>=2",
    "cloudpickle>=3.1.0,<4",
    "joblib>=1.4.2,<2",
]

import copy
import re

from .solution import Solution
from .utils import TimeoutException


def simplify_subprocess_error(stderr: str, solution=None):
    """
    Parse a Python traceback string and produce a concise error summary.
    Optionally include the offending line of code from `solution.code`.
    """
    if not stderr:
        return "Unknown error."

    # Extract the last "File ..." block and the final exception line
    # This regex catches the last occurrence of: File "...", line X, in Y
    file_line_match = list(re.finditer(r'File ".*?", line (\d+), in (.+)', stderr))
    exc_match = re.search(r"([A-Za-z_]+Error): (.*)", stderr.splitlines()[-1])

    if not file_line_match or not exc_match:
        # fallback: just return the final line
        return stderr.strip()

    last = file_line_match[-1]
    line_no = int(last.group(1))
    func = last.group(2).strip()
    exc_type, exc_msg = exc_match.groups()

    # Optional: fetch offending code line if available
    code_line = ""
    if solution and hasattr(solution, "code"):
        code_lines = solution.code.splitlines()
        if 1 <= line_no <= len(code_lines):
            code_line = code_lines[line_no - 1].strip()

    msg = f"In the code, line {line_no}, in {func}, the following error occurred:\n{exc_type}: {exc_msg}"
    if code_line:
        msg += f"\nOn line: {code_line}"
    return msg


_WORKER_LOOP_SCRIPT = r'''
import sys, os, struct, io, types

# Keep real stdout/stderr for the binary protocol
_real_stdout = sys.stdout.buffer
_real_stderr = sys.stderr

# Redirect stdout/stderr so print() in eval'd code can't corrupt our protocol
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

# Register llamea.solution as alias for iohblade.solution so cloudpickle
# can deserialise llamea.Solution without pulling in lizard/networkx/etc.
import iohblade.solution as _blade_sol
if 'llamea' not in sys.modules:
    _pkg = types.ModuleType('llamea')
    _pkg.__path__ = []
    sys.modules['llamea'] = _pkg
if 'llamea.solution' not in sys.modules:
    sys.modules['llamea.solution'] = _blade_sol

import cloudpickle

problem_path = sys.argv[1]
with open(problem_path, 'rb') as f:
    problem = cloudpickle.load(f)

def recv_msg(stream):
    header = stream.read(4)
    if not header or len(header) < 4:
        return None
    length = struct.unpack('>I', header)[0]
    if length == 0:
        return None
    return stream.read(length)

def send_msg(stream, data):
    stream.write(struct.pack('>I', len(data)))
    stream.write(data)
    stream.flush()

stdin = sys.stdin.buffer
while True:
    raw = recv_msg(stdin)
    if raw is None:
        break
    try:
        solution = cloudpickle.loads(raw)
        result = problem.evaluate(solution)
        captured_stdout = sys.stdout.getvalue()
        captured_stderr = sys.stderr.getvalue()
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        response = cloudpickle.dumps({
            'result': result,
            'stdout': captured_stdout,
            'stderr': captured_stderr,
        })
    except Exception as e:
        captured_stdout = sys.stdout.getvalue()
        captured_stderr = sys.stderr.getvalue()
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        response = cloudpickle.dumps({
            'error': str(e),
            'stdout': captured_stdout,
            'stderr': captured_stderr,
        })
    send_msg(_real_stdout, response)
'''


def _is_solution(data):
    """Check if data is a Solution (supports both iohblade and llamea Solution)."""
    return isinstance(data, Solution) or hasattr(data, 'set_scores')


def evaluate_in_subprocess(problem, conn, solution):
    """Evaluate a solution in a dedicated virtual environment."""
    proc = None
    try:
        env_path = problem._env_path
        python_bin = problem._python_bin

        problem_pickle = env_path / "problem.pkl"
        solution_pickle = env_path / f"solution_{uuid.uuid4().hex}.pkl"
        result_pickle = (
            Path(tempfile.gettempdir()) / f"blade_result_{uuid.uuid4().hex}.pkl"
        )
        problem_copy = copy.deepcopy(problem)
        problem_copy.logger = None
        if not os.path.exists(problem_pickle):
            with open(problem_pickle, "wb") as f:
                cloudpickle.dump(problem_copy, f)
        with open(solution_pickle, "wb") as f:
            cloudpickle.dump(solution, f)

        script_path = env_path / "run_eval.py"
        imports_block = getattr(problem, "imports", "")
        script_path.write_text(
            (f"{imports_block}\n" if imports_block else "")
            + "import cloudpickle as cp\n"
            + "import os, json\n"
            + f"problem_path = {json.dumps(str(problem_pickle))}\n"
            + f"solution_path = {json.dumps(str(solution_pickle))}\n"
            + f"result_path  = {json.dumps(str(result_pickle))}\n"
            + "problem=cp.load(open(problem_path,'rb'))\n"
            + "solution=cp.load(open(solution_path,'rb'))\n"
            + "result=problem.evaluate(solution)\n"
            + "with open(result_path,'wb') as f:\n"
            + "    cp.dump(result, f)\n"
        )

        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[1]
        path_parts = [str(repo_root)]
        for p in getattr(problem, 'extra_pythonpath', None) or []:
            if p not in path_parts:
                path_parts.insert(0, str(p))
        env["PYTHONPATH"] = os.pathsep.join(path_parts) + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            [str(python_bin), str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=problem.eval_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            conn.send(
                {
                    "error": f"Evaluation timed out after {problem.eval_timeout} seconds.",
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            return

        if proc.returncode != 0:
            error_msg = simplify_subprocess_error(stderr, solution)
            conn.send({"error": error_msg, "stdout": stdout, "stderr": stderr})
            return

        with open(result_pickle, "rb") as f:
            result = cloudpickle.load(f)
        conn.send({"result": result, "stdout": stdout, "stderr": stderr})

    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        line_no = tb.lineno
        code_line = ""

        code_lines = solution.code.split("\n")
        if line_no and len(code_lines) >= line_no:
            code_line = code_lines[line_no - 1]
        error_type = type(e).__name__
        error_msg = str(e)
        error = f"{error_type}: {error_msg}.\n"
        if code_lines:
            error += f"On line {line_no}: {code_line}.\n"
        conn.send(
            {
                "error": error,
                "stdout": "",
                "stderr": "",
            }
        )
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.communicate()
        conn.close()


class Problem(ABC):
    """
    Abstract problem class.
    """

    def __init__(
        self,
        logger=None,
        training_instances=None,
        test_instances=None,
        name="Problem",
        eval_timeout=6000,
        dependencies=None,
        imports=None,
        use_worker_pool=False,
        worker_recycle_interval=50,
        extra_pythonpath=None,
    ):
        """
        Initializes a problem instance with logging and dataset references.

        Args:
            logger (Logger, optional): Logger object for tracking solutions.
            training_instances (list, optional): List of training problem instances.
            test_instances (list, optional): List of test problem instances.
            name (str, optional): Name of the problem.
            eval_timeout (int, optional): Number of seconds before a timeout error is raised.
            budget (int): number of algorithms are allowed to be generated per run.
            dependencies (list, optional): a list of pypi packages to install before evaluation.
            imports (string, optional): the python string to manage imports in the evaluation file.
            use_worker_pool (bool): If True, use a persistent worker process instead of spawning a new subprocess per evaluation.
            worker_recycle_interval (int): Recycle the worker process every N evaluations.
            extra_pythonpath (list, optional): Additional paths to prepend to PYTHONPATH in subprocesses.
        """
        self.logger = logger
        self.logger_dir = ""
        self.training_instances = training_instances if training_instances else []
        self.test_instances = test_instances if test_instances else []
        self.task_prompt = "Write the problem description part here."
        self.example_prompt = "Write an example code here."
        self.format_prompt = "Write the format description part here."
        self.name = name
        self.eval_timeout = eval_timeout
        # Combine the base dependencies with any problem specific ones
        self.dependencies = BASE_DEPENDENCIES.copy()
        if dependencies:
            self.dependencies.extend(dependencies)
        if imports is None:
            self.imports = "import numpy as np\n"
        else:
            self.imports = imports

        # Path to the virtual environment used for evaluations
        self._env_path: Path | None = None
        self._python_bin: Path | None = None

        # Worker pool settings
        self.use_worker_pool = use_worker_pool
        self.worker_recycle_interval = worker_recycle_interval
        self.extra_pythonpath = extra_pythonpath or []
        self._worker_process = None
        self._eval_count = 0

        # These settings are required for EoH, adapt them based on your problem.
        # The function name, inputs, and outputs should match the expected format.
        # For example, if your problem requires a function that takes a function, budget, and dimension,
        # and returns the optimal fitness and solution, set them accordingly.
        self.func_name = "__call__"
        self.init_inputs = ["budget", "dim"]
        self.func_inputs = ["func"]
        self.func_outputs = ["f_opt", "x_opt"]

    def __call__(self, solution: Solution, logger=None):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Routes to worker pool or subprocess-per-eval based on self.use_worker_pool.
        """
        if logger is not None:
            self.logger = logger

        if self.logger is not None:
            if self.logger.budget_exhausted():
                solution.set_scores(
                    -np.inf,
                    feedback="Budget is exhausted.",
                )
                return solution

        if self.use_worker_pool:
            solution = self._call_worker_pool(solution)
        else:
            solution = self._call_subprocess(solution)

        if self.logger is not None:
            self.logger.log_individual(solution)
        return solution

    def _handle_result(self, result, solution):
        """Process a result dict/object from either subprocess or worker pool."""
        stdout = ""
        stderr = ""
        if isinstance(result, dict):
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            if "error" in result:
                err = result["error"]
                solution.set_scores(-np.inf, feedback=err)
            else:
                data = result.get("result")
                if _is_solution(data):
                    solution = data
                elif isinstance(data, str):
                    solution.set_scores(-np.inf, feedback=data)
                else:
                    raise Exception("No Solution object or string returned.")
        elif isinstance(result, Exception):
            raise result
        elif _is_solution(result):
            solution = result
        elif isinstance(result, str):
            solution.set_scores(-np.inf, feedback=result)
        else:
            raise Exception("No Solution object or string returned.")
        return solution, stdout, stderr

    def _call_subprocess(self, solution):
        """Original subprocess-per-eval implementation."""
        stdout = ""
        stderr = ""
        self._last_stdout = ""
        self._last_stderr = ""
        process: multiprocessing.Process | None = None
        parent_conn = None
        child_conn = None
        try:
            self._ensure_env()
            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(
                target=evaluate_in_subprocess, args=(self, child_conn, solution)
            )
            process.start()
            process.join(timeout=self.eval_timeout + 60)

            if process.is_alive():
                raise TimeoutException(
                    f"Evaluation timed out after {self.eval_timeout} seconds."
                )
            if parent_conn.poll():
                result = parent_conn.recv()
                solution, stdout, stderr = self._handle_result(result, solution)
            else:
                raise Exception("Evaluation failed without an exception.")
        except Exception as e:
            solution.set_scores(-np.inf, feedback=f"{e}")
        finally:
            if process is not None:
                if process.is_alive():
                    process.kill()
                process.join()
            if parent_conn is not None:
                parent_conn.close()
            if child_conn is not None:
                child_conn.close()

        self._last_stdout = stdout
        self._last_stderr = stderr
        return solution

    def _call_worker_pool(self, solution):
        """Persistent worker process implementation."""
        self._last_stdout = ""
        self._last_stderr = ""
        try:
            self._ensure_env()
            # Start or recycle worker
            if self._worker_process is None or self._worker_process.poll() is not None:
                self._start_worker()
            elif (self.worker_recycle_interval > 0
                  and self._eval_count % self.worker_recycle_interval == 0
                  and self._eval_count > 0):
                logger.debug("Recycling worker after %d evaluations", self._eval_count)
                self._restart_worker()

            result = self._send_recv_worker(solution)
            self._eval_count += 1
            solution, stdout, stderr = self._handle_result(result, solution)
            self._last_stdout = stdout
            self._last_stderr = stderr
        except Exception as e:
            logger.warning("Worker pool error: %s — restarting worker", e)
            self._stop_worker()
            solution.set_scores(-np.inf, feedback=f"{e}")
        return solution

    # --- Worker lifecycle ---

    def _start_worker(self):
        """Spawn a persistent worker subprocess."""
        env_path = self._env_path
        python_bin = self._python_bin

        # Write worker script
        script_path = env_path / "worker_loop.py"
        script_path.write_text(_WORKER_LOOP_SCRIPT)

        # Pickle problem (without logger/worker refs)
        problem_pickle = env_path / "problem.pkl"
        problem_copy = copy.deepcopy(self)
        problem_copy.logger = None
        problem_copy._worker_process = None
        with open(problem_pickle, "wb") as f:
            cloudpickle.dump(problem_copy, f)

        # Build PYTHONPATH
        repo_root = Path(__file__).resolve().parents[1]
        path_parts = [str(repo_root)]
        for p in self.extra_pythonpath:
            if str(p) not in path_parts:
                path_parts.insert(0, str(p))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(path_parts) + os.pathsep + env.get("PYTHONPATH", "")

        self._worker_process = subprocess.Popen(
            [str(python_bin), str(script_path), str(problem_pickle)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        logger.debug("Started worker process PID %d", self._worker_process.pid)

    def _stop_worker(self):
        """Gracefully shut down the worker process."""
        proc = self._worker_process
        if proc is None:
            return
        try:
            if proc.poll() is None:
                # Send zero-length shutdown message
                proc.stdin.write(struct.pack('>I', 0))
                proc.stdin.flush()
                proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            self._worker_process = None

    def _restart_worker(self):
        """Stop and restart the worker process."""
        self._stop_worker()
        # Delete stale problem pickle so it gets re-created
        problem_pickle = self._env_path / "problem.pkl"
        if problem_pickle.exists():
            problem_pickle.unlink()
        self._start_worker()

    def _send_recv_worker(self, solution):
        """Send a solution to the worker and receive the result."""
        proc = self._worker_process
        data = cloudpickle.dumps(solution)
        # Send length-prefixed message
        proc.stdin.write(struct.pack('>I', len(data)))
        proc.stdin.write(data)
        proc.stdin.flush()

        # Wait for response with timeout
        if hasattr(select, 'select'):
            ready, _, _ = select.select([proc.stdout], [], [], self.eval_timeout + 60)
            if not ready:
                self._stop_worker()
                raise TimeoutException(
                    f"Worker evaluation timed out after {self.eval_timeout} seconds."
                )

        # Read length-prefixed response
        header = proc.stdout.read(4)
        if not header or len(header) < 4:
            raise Exception("Worker process closed unexpectedly.")
        resp_len = struct.unpack('>I', header)[0]
        resp_data = proc.stdout.read(resp_len)
        return cloudpickle.loads(resp_data)

    def _ensure_env(self):
        """Create the virtual environment for evaluations if it does not exist."""
        if self._env_path is not None:
            return
        import virtualenv

        env_dir = tempfile.mkdtemp(prefix="blade_env_")
        self._env_path = Path(env_dir)
        virtualenv.cli_run([env_dir])
        self._python_bin = (
            self._env_path / ("Scripts" if os.name == "nt" else "bin") / "python"
        )

        deps = getattr(self, "dependencies", [])
        if deps:
            subprocess.run(
                [str(self._python_bin), "-m", "pip", "install", *deps],
                check=True,
                capture_output=True,
                text=True,
            )

    def cleanup(self):
        self._stop_worker()
        try:
            if self._env_path and self._env_path.exists():
                shutil.rmtree(self._env_path)
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_worker_process'] = None
        state['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._worker_process = None

    def __del__(self):
        try:
            self._stop_worker()
        except Exception:
            pass

    def set_logger(self, logger):
        """
        Sets the logger for this problem.
        """
        self.logger = logger
        if logger != None:
            self.logger_dir = logger.get_log_dir()

    def get_prompt(self):
        """
        Get the full prompt describing the problem and how to format the answer.
        """
        return self.task_prompt + self.example_prompt + self.format_prompt

    @abstractmethod
    def evaluate(self, solution: Solution):
        """
        Evaluates a solution on training instances and updates its fitness and feedback.

        Args:
            solution (Solution): Solution object to be evaluated.
        """
        pass

    @abstractmethod
    def test(self, solution: Solution):
        """
        Performs a complete evaluation on test instances and returns the fitness score.

        Args:
            solution (Solution): Solution object to be tested.
        """
        pass

    @abstractmethod
    def to_dict(self):
        """
        Returns a dictionary representation of the problem including all parameters.

        Returns:
            dict: Dictionary representation of the problem.
        """
        pass


class WrappedProblem(Problem):
    def __init__(
        self,
        evaluate_fn,
        *,
        name="Problem",
        eval_timeout=600,
        training_instances=None,
        test_instances=None,
        dependencies=None,
        imports=None,
        task_prompt="",
        example_prompt="",
        logger=None,
    ):
        super().__init__(
            logger=logger,
            training_instances=training_instances,
            test_instances=test_instances,
            name=name,
            eval_timeout=eval_timeout,
            dependencies=dependencies,
            imports=imports,
        )
        if task_prompt:
            self.task_prompt = task_prompt
        if example_prompt:
            self.example_prompt = example_prompt

        self._evaluate_fn = evaluate_fn
        # support both signatures: (solution) and (self, solution)
        self._takes_self = len(inspect.signature(evaluate_fn).parameters) > 1
        # store by value
        self._evaluate_fn_bytes = cloudpickle.dumps(evaluate_fn)
        self._evaluate_fn = None  # reconstructed lazily

    def _get_evaluate_fn(self):
        if self._evaluate_fn is None:
            self._evaluate_fn = cloudpickle.loads(self._evaluate_fn_bytes)
        return self._evaluate_fn

    def evaluate(self, solution: Solution):
        fn = self._get_evaluate_fn()
        if self._takes_self:
            return fn(self, solution)
        return fn(solution)

    def test(self, solution: Solution):
        return self.evaluate(solution)

    def to_dict(self):
        return {
            "name": self.name,
            "eval_timeout": self.eval_timeout,
            "training_instances": self.training_instances,
            "test_instances": self.test_instances,
            "dependencies": self.dependencies,
            "imports": self.imports,
        }


def wrap_problem(
    evaluate_fn,
    *,
    name="Problem",
    eval_timeout=6000,
    training_instances=None,
    test_instances=None,
    dependencies=None,
    imports=None,
    task_prompt="",
    example_prompt="",
    logger=None,
):
    return WrappedProblem(
        evaluate_fn,
        name=name,
        eval_timeout=eval_timeout,
        training_instances=training_instances,
        test_instances=test_instances,
        dependencies=dependencies,
        imports=imports,
        task_prompt=task_prompt,
        example_prompt=example_prompt,
        logger=logger,
    )
