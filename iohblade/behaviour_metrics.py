"""
behavior_metrics.py

Utility functions for analyzing single-run black box optimization traces.

Each function expects a pandas DataFrame that contains at least:
    * a column "evaluations"   integer evaluation index (starting at 1 or 0)
    * a column "raw_y"         objective function value (the lower the better)
    * coordinate columns       named "x0", "x1", … "x{d-1}"

The helper `get_coordinates(df)` extracts the X matrix.

Many metrics also accept optional keyword arguments such as `bounds`
(the search space lower/upper bounds) or `radius`.

All functions return a scalar float unless stated otherwise.

Author: Niki van Stein (May 2025)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def get_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Return (N, d) array with decision variables extracted from x-columns."""
    x_cols = [c for c in df.columns if c.startswith("x")]
    return df[x_cols].to_numpy()


def get_objective(df: pd.DataFrame) -> np.ndarray:
    """Return 1‑D array with objective values (raw_y)."""
    return df["raw_y"].to_numpy()


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Return condensed distance vector (upper triangle, no diag)."""
    return distance_matrix(X, X)[np.triu_indices(X.shape[0], k=1)]


# ---------------------------------------------------------------------
# Exploration metrics
# ---------------------------------------------------------------------


def average_nearest_neighbor_distance(df: pd.DataFrame, step=10, history=1000) -> float:
    """
    Average Euclidean distance from each point (except the first) to its
    nearest previous point -- proxy for sequential novelty / exploration.
    """
    X = get_coordinates(df)
    if len(X) < 2:
        return 0.0
    dists = []
    for k in range(1, len(X), step):
        prev = X[max(0, k - history) : k]
        dmin = np.min(np.linalg.norm(X[k] - prev, axis=1))
        dists.append(dmin)
    return float(np.mean(dists))


# 1 4608790451.0    5e+09     58.5          "dispersion": coverage_dispersion(df, bounds, disp_samples),
#   285         1 3133208393.0    3e+09     39.8          "spatial_entropy": spatial_entropy(df),


def coverage_dispersion(
    df: pd.DataFrame,
    bounds: Sequence[Tuple[float, float]] = None,
    n_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Approximate dispersion: max distance from a random point in the domain
    to its nearest evaluated sample. Lower is better (more coverage).
    """
    if rng is None:
        rng = np.random.default_rng()
    X = get_coordinates(df)
    d = X.shape[1]
    if bounds is None:
        bounds = [(-5.0, 5.0)] * d
    bounds = np.asarray(bounds, dtype=float)
    assert bounds.shape == (d, 2), "bounds should be shape (d, 2)"

    rand_points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, d))

    # Use k-d tree for fast nearest neighbor search
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(X)
    distances, _ = nn.kneighbors(rand_points)

    return float(distances.max())



# ---------------------------------------------------------------------
# Exploitation metrics
# ---------------------------------------------------------------------


def average_distance_to_best_so_far(df: pd.DataFrame) -> float:
    """
    For each evaluation k>0, compute distance to best point found up to k-1.
    Average these distances   lower = more exploitative.
    """
    X = get_coordinates(df)
    y = get_objective(df)
    best_idx = 0

    dists = []
    for k in range(1, len(X)):
        if y[k] < y[best_idx]:
            best_idx = k

        dists.append(np.linalg.norm(X[k] - X[best_idx]))
    return float(np.mean(dists))


def closed_form_random_search_diversity(bounds: Sequence[Tuple[float, float]]) -> float:
    bounds = np.asarray(bounds, dtype=float)
    widths = bounds[:, 1] - bounds[:, 0]
    d = len(bounds)
    w = np.mean(widths)  # assume roughly cubic
    return w * np.sqrt(d / 6)


def estimate_random_search_diversity(
    bounds: Sequence[Tuple[float, float]],
    n: int = 500,
    samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng()
    d = len(bounds)
    bounds = np.asarray(bounds)
    rand_points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(samples, d))
    return pdist(rand_points[:n]).mean()


def avg_exploration_exploitation_chunked(
    df: pd.DataFrame,
    chunk_size: int = 500,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    X = get_coordinates(df)
    n, d = X.shape

    if bounds is None:
        bounds = [(-5.0, 5.0)] * d

    if rng is None:
        rng = np.random.default_rng()

    # Estimate diversity baseline from random sampling
    max_div = closed_form_random_search_diversity(bounds=bounds)

    chunk_diversities = []
    for i in range(0, n, chunk_size):
        chunk = X[i : i + chunk_size]
        if len(chunk) < 2:
            continue
        chunk_div = pdist(chunk).mean()
        chunk_diversities.append(chunk_div)

    if not chunk_diversities:
        return 0.0, 100.0  # no meaningful data

    normalized_explorations = 100 * np.array(chunk_diversities) / max_div
    normalized_explorations = np.clip(normalized_explorations, 0, 100)
    exploration = normalized_explorations.mean()
    exploitation = 100 - exploration

    return float(exploration), float(exploitation)


def intensification_ratio(df: pd.DataFrame, radius: float) -> float:
    """
    Fraction of evaluations lying within a given radius of the final best point.
    Higher -> stronger intensification / exploitation near the best.
    """
    X = get_coordinates(df)
    y = get_objective(df)
    best_idx = y.argmin()
    dists = np.linalg.norm(X - X[best_idx], axis=1)
    return float(np.mean(dists < radius))


# ---------------------------------------------------------------------
# Convergence metrics
# ---------------------------------------------------------------------


def average_convergence_rate(df: pd.DataFrame, optimum: float | None = None) -> float:
    """
    Geometric mean of successive error ratios (Chen & He, 2020).
    ACR < 1 implies convergence; smaller = faster.
    """
    y = get_objective(df)
    best_so_far = np.minimum.accumulate(y)
    if optimum is None:
        optimum = best_so_far[-1]
    errors = best_so_far - optimum
    # avoid zeros by adding tiny eps
    eps = np.finfo(float).eps
    ratios = (errors[1:] + eps) / (errors[:-1] + eps)
    return float(np.exp(np.log(ratios).mean()))


def improvement_statistics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns (avg_improvement, success_rate) where success_rate is the fraction
    of iterations that improved the best, and avg_improvement is the mean
    improvement magnitude on improving iterations.
    """
    y = get_objective(df)
    best = y[0]
    improvements = []
    successes = 0
    for val in y[1:]:
        if val < best:
            improvements.append(best - val)
            successes += 1
            best = val
    avg_imp = float(np.mean(improvements)) if improvements else 0.0
    success_rate = successes / (len(y) - 1) if len(y) > 1 else 0.0
    return avg_imp, success_rate


# ---------------------------------------------------------------------
# Stagnation metrics
# ---------------------------------------------------------------------


def longest_no_improvement_streak(df: pd.DataFrame) -> int:
    """Return the length of the longest consecutive streak with no improvement."""
    y = get_objective(df)
    best = y[0]
    curr = longest = 0
    for val in y[1:]:
        if val < best:
            best = val
            curr = 0
        else:
            curr += 1
            longest = max(longest, curr)
    return int(longest)


def last_improvement_fraction(df: pd.DataFrame) -> float:
    """Fraction of evaluations since the last improvement (0-1)."""
    best_so_far = np.minimum.accumulate(get_objective(df))
    last_imp_idx = np.where(np.diff(best_so_far) < 0)[0]
    last_imp_idx = last_imp_idx.max() + 1 if last_imp_idx.size else 0
    return (len(best_so_far) - 1 - last_imp_idx) / (len(best_so_far) - 1)


# ---------------------------------------------------------------------
# Category 1: Step-Size & Movement Dynamics
# ---------------------------------------------------------------------


def _consecutive_step_sizes(df: pd.DataFrame) -> np.ndarray:
    """Euclidean distances between consecutive evaluations."""
    X = get_coordinates(df)
    if len(X) < 2:
        return np.array([0.0])
    return np.linalg.norm(np.diff(X, axis=0), axis=1)


def step_size_mean(df: pd.DataFrame) -> float:
    """Mean Euclidean distance between consecutive evaluations."""
    return float(np.mean(_consecutive_step_sizes(df)))


def step_size_std(df: pd.DataFrame) -> float:
    """Standard deviation of consecutive step sizes."""
    return float(np.std(_consecutive_step_sizes(df)))


def step_size_trend(df: pd.DataFrame) -> float:
    """Slope of linear regression of step sizes over evaluation index.
    Negative = contracting (convergence), positive = expanding."""
    steps = _consecutive_step_sizes(df)
    if len(steps) < 2:
        return 0.0
    t = np.arange(len(steps)).reshape(-1, 1)
    reg = LinearRegression().fit(t, steps)
    return float(reg.coef_[0])


def directional_persistence(df: pd.DataFrame) -> float:
    """Mean cosine similarity between consecutive displacement vectors.
    High (~1) = persistent direction, low (~0) = random/zigzag."""
    X = get_coordinates(df)
    if len(X) < 3:
        return 0.0
    displacements = np.diff(X, axis=0)
    norms = np.linalg.norm(displacements, axis=1, keepdims=True)
    norms = np.maximum(norms, np.finfo(float).eps)
    unit_d = displacements / norms
    cos_sim = np.sum(unit_d[:-1] * unit_d[1:], axis=1)
    return float(np.mean(cos_sim))


# ---------------------------------------------------------------------
# Category 2: Information-Theoretic Features
# ---------------------------------------------------------------------


def fitness_sample_entropy(df: pd.DataFrame, m: int = 2,
                           subsample_step: int = 10) -> float:
    """Sample entropy of raw_y (Richman & Moorman 2000).
    Subsampled for performance (O(n^2) on full series)."""
    try:
        import antropy
    except ImportError:
        return float("nan")
    y = get_objective(df)
    y_sub = np.ascontiguousarray(y[::subsample_step])
    if len(y_sub) < 50:
        return float("nan")
    if np.std(y_sub) == 0:
        return 0.0
    return float(antropy.sample_entropy(y_sub, order=int(m), metric="chebyshev"))


def fitness_permutation_entropy(df: pd.DataFrame, order: int = 5,
                                delay: int = 1) -> float:
    """Normalized permutation entropy of raw_y (Bandt & Pompe 2002)."""
    try:
        import antropy
    except ImportError:
        return float("nan")
    y = get_objective(df)
    if len(y) < 2 * math.factorial(order):
        return float("nan")
    return float(antropy.perm_entropy(y, order=order, delay=delay,
                                      normalize=True))


def fitness_autocorrelation(df: pd.DataFrame) -> float:
    """Lag-1 autocorrelation of raw_y (Weinberger 1990)."""
    y = get_objective(df)
    if len(y) < 3:
        return 0.0
    y_c = y - np.mean(y)
    var = np.dot(y_c, y_c)
    if var == 0:
        return 0.0
    return float(np.dot(y_c[:-1], y_c[1:]) / var)


def fitness_lempel_ziv_complexity(df: pd.DataFrame) -> float:
    """Normalized Lempel-Ziv complexity of binarized fitness changes
    (Lempel & Ziv 1976). Binary: 1 if f improves, 0 otherwise."""
    try:
        import antropy
    except ImportError:
        return float("nan")
    y = get_objective(df)
    if len(y) < 3:
        return 0.0
    binary = (np.diff(y) < 0).astype(int)
    return float(antropy.lziv_complexity(binary, normalize=True))



# ---------------------------------------------------------------------
# Category 4: Adapted DynamoRep -- Early/Late Dynamics
# (Inspired by Cenikj et al. 2023, adapted for (1+1)-ES)
# ---------------------------------------------------------------------


def x_spread_early(df: pd.DataFrame) -> float:
    """Mean per-dimension std of x-coordinates in first 25% of evaluations."""
    X = get_coordinates(df)
    cutoff = max(1, len(X) // 4)
    return float(np.mean(np.std(X[:cutoff], axis=0)))


def x_spread_late(df: pd.DataFrame) -> float:
    """Mean per-dimension std of x-coordinates in last 25% of evaluations."""
    X = get_coordinates(df)
    cutoff = max(1, len(X) // 4)
    return float(np.mean(np.std(X[-cutoff:], axis=0)))


def spread_ratio(df: pd.DataFrame) -> float:
    """Ratio of late to early x-spread. <1 = convergence, >1 = divergence."""
    early = x_spread_early(df)
    if early == 0:
        return 0.0
    return float(x_spread_late(df) / early)


def centroid_drift(df: pd.DataFrame) -> float:
    """Euclidean distance between centroid of first 25% and last 25%."""
    X = get_coordinates(df)
    cutoff = max(1, len(X) // 4)
    c_early = np.mean(X[:cutoff], axis=0)
    c_late = np.mean(X[-cutoff:], axis=0)
    return float(np.linalg.norm(c_late - c_early))


def f_range_early(df: pd.DataFrame) -> float:
    """Range (max-min) of fitness in first 25% of evaluations."""
    y = get_objective(df)
    cutoff = max(1, len(y) // 4)
    return float(np.ptp(y[:cutoff]))


def f_range_late(df: pd.DataFrame) -> float:
    """Range (max-min) of fitness in last 25% of evaluations."""
    y = get_objective(df)
    cutoff = max(1, len(y) // 4)
    return float(np.ptp(y[-cutoff:]))


def f_range_ratio(df: pd.DataFrame) -> float:
    """Ratio of late to early fitness range. <1 = narrowing, >1 = widening."""
    early = f_range_early(df)
    if early == 0:
        return 0.0
    return float(f_range_late(df) / early)


# ---------------------------------------------------------------------
# Category 5: Novel Features
# ---------------------------------------------------------------------


def improvement_spatial_correlation(df: pd.DataFrame) -> float:
    """Pearson correlation between step size and improvement magnitude,
    on improving steps only. Positive = big jumps yield big improvements."""
    X = get_coordinates(df)
    y = get_objective(df)
    if len(y) < 3:
        return 0.0
    best_so_far = np.minimum.accumulate(y)
    improving = np.zeros(len(y), dtype=bool)
    improving[1:] = y[1:] < best_so_far[:-1]
    idx = np.where(improving)[0]
    if len(idx) < 3:
        return 0.0
    steps = np.linalg.norm(X[idx] - X[idx - 1], axis=1)
    imps = best_so_far[idx - 1] - y[idx]
    if np.std(steps) == 0 or np.std(imps) == 0:
        return 0.0
    return float(np.corrcoef(steps, imps)[0, 1])


def improvement_burstiness(df: pd.DataFrame) -> float:
    """CV of inter-improvement intervals. High = bursty improvements,
    low = steady improvement rate. Inspired by Barabasi (2005)."""
    y = get_objective(df)
    best = y[0]
    imp_idx = []
    for t in range(1, len(y)):
        if y[t] < best:
            imp_idx.append(t)
            best = y[t]
    if len(imp_idx) < 3:
        return 0.0
    intervals = np.diff(imp_idx).astype(float)
    mu = np.mean(intervals)
    if mu == 0:
        return 0.0
    return float(np.std(intervals) / mu)


def dimension_convergence_heterogeneity(df: pd.DataFrame) -> float:
    """Std across dimensions of per-dimension range shrinkage (late/early).
    High = uneven convergence across dimensions."""
    X = get_coordinates(df)
    n, d = X.shape
    if n < 4 or d < 2:
        return 0.0
    cutoff = max(1, n // 4)
    shrinkages = []
    for j in range(d):
        r_early = np.ptp(X[:cutoff, j])
        r_late = np.ptp(X[-cutoff:, j])
        shrinkages.append(r_late / r_early if r_early > 0 else 0.0)
    return float(np.std(shrinkages))


def step_size_autocorrelation(df: pd.DataFrame) -> float:
    """Lag-1 autocorrelation of step sizes (not fitness).
    Captures search momentum."""
    steps = _consecutive_step_sizes(df)
    if len(steps) < 3:
        return 0.0
    s_c = steps - np.mean(steps)
    var = np.dot(s_c, s_c)
    if var == 0:
        return 0.0
    return float(np.dot(s_c[:-1], s_c[1:]) / var)


def fitness_plateau_fraction(df: pd.DataFrame) -> float:
    """Fraction of consecutive evaluations with negligible fitness change.
    Uses raw fitness (not best-so-far), capturing flat landscape regions."""
    y = get_objective(df)
    if len(y) < 2:
        return 0.0
    y_range = np.ptp(y)
    if y_range == 0:
        return 1.0
    eps = 1e-8 * y_range
    return float(np.mean(np.abs(np.diff(y)) < eps))


def half_convergence_time(df: pd.DataFrame) -> float:
    """Fraction of budget when best-so-far reaches 50% of total improvement.
    Low = fast early convergence, high = slow or late convergence."""
    y = get_objective(df)
    bsf = np.minimum.accumulate(y)
    total_imp = bsf[0] - bsf[-1]
    if total_imp <= 0:
        return 1.0
    half_target = bsf[0] - 0.5 * total_imp
    idx = np.searchsorted(-bsf, -half_target)
    return float(idx / max(1, len(y) - 1))


# ---------------------------------------------------------------------
# Unified summary function
# ---------------------------------------------------------------------


def compute_behavior_metrics(
    df: pd.DataFrame,
    *,
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    radius: Optional[float] = None,
    disp_samples: int = 10_000,
):
    """Compute all scalar behavior metrics and return them in a dictionary."""

    # default bounds and radius
    if bounds is None:
        d = len([c for c in df.columns if c.startswith("x")])
        bounds = [(-5.0, 5.0)] * d
    if radius is None:
        radius = 0.1 * (bounds[0][1] - bounds[0][0])

    # time‑series based exploration/exploitation percentages (averaged)
    avg_exploration_pct, avg_exploitation_pct = avg_exploration_exploitation_chunked(df)

    # improvement stats
    avg_imp, success_rate = improvement_statistics(df)

    metrics = {
        # Exploration & diversity
        "avg_nearest_neighbor_distance": average_nearest_neighbor_distance(df),
        "dispersion": coverage_dispersion(df, bounds, disp_samples),
        "avg_exploration_pct": avg_exploration_pct,
        # Exploitation
        "avg_distance_to_best": average_distance_to_best_so_far(df),
        "intensification_ratio": intensification_ratio(df, radius),
        "avg_exploitation_pct": avg_exploitation_pct,
        # Convergence
        "average_convergence_rate": average_convergence_rate(df),
        "avg_improvement": avg_imp,
        "success_rate": success_rate,
        # Stagnation
        "longest_no_improvement_streak": longest_no_improvement_streak(df),
        "last_improvement_fraction": last_improvement_fraction(df),
        # Cat 1: Step-size dynamics
        "step_size_mean": step_size_mean(df),
        "step_size_std": step_size_std(df),
        "step_size_trend": step_size_trend(df),
        "directional_persistence": directional_persistence(df),
        # Cat 2: Information-theoretic
        "fitness_sample_entropy": fitness_sample_entropy(df),
        "fitness_permutation_entropy": fitness_permutation_entropy(df),
        "fitness_autocorrelation": fitness_autocorrelation(df),
        "fitness_lempel_ziv_complexity": fitness_lempel_ziv_complexity(df),
        # Cat 4: Adapted DynamoRep (early/late dynamics)
        "x_spread_early": x_spread_early(df),
        "x_spread_late": x_spread_late(df),
        "spread_ratio": spread_ratio(df),
        "centroid_drift": centroid_drift(df),
        "f_range_early": f_range_early(df),
        "f_range_late": f_range_late(df),
        "f_range_ratio": f_range_ratio(df),
        # Cat 5: Novel features
        "improvement_spatial_correlation": improvement_spatial_correlation(df),
        "improvement_burstiness": improvement_burstiness(df),
        "dimension_convergence_heterogeneity": dimension_convergence_heterogeneity(df),
        "step_size_autocorrelation": step_size_autocorrelation(df),
        "fitness_plateau_fraction": fitness_plateau_fraction(df),
        "half_convergence_time": half_convergence_time(df),
    }
    return metrics
