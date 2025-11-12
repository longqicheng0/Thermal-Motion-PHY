"""Analysis functions for thermal motion tracking.

Provides computations for relative positions, step displacements, radial
displacement, mean squared displacement (MSD) and a simple 2D diffusion
coefficient estimator (MSD ~ 4 D t).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict
import math

from scipy import stats


def relative_positions(positions: np.ndarray) -> np.ndarray:
    """Return positions relative to the initial position (set initial as zero).

    Args:
        positions: (N,2) array of raw X,Y positions

    Returns:
        rel: (N,2) positions shifted so the first row is at the origin
    """
    if positions.size == 0:
        return positions.reshape(-1, 2)
    origin = positions[0]
    return positions - origin


def step_displacements(rel_positions: np.ndarray) -> np.ndarray:
    """Return per-frame displacement vectors (differences between consecutive frames).

    Returns an array of shape (N-1, 2).
    """
    if rel_positions.shape[0] < 2:
        return np.zeros((0, 2))
    return np.diff(rel_positions, axis=0)


def step_lengths(steps: np.ndarray) -> np.ndarray:
    """Euclidean length for each step vector.
    """
    if steps.size == 0:
        return np.zeros((0,))
    return np.linalg.norm(steps, axis=1)


def radial_displacement(rel_positions: np.ndarray) -> np.ndarray:
    """Distance from the origin for each frame.
    """
    if rel_positions.size == 0:
        return np.zeros((0,))
    return np.linalg.norm(rel_positions, axis=1)

def mean_squared_displacement_time(
    rel_positions: np.ndarray,
    dt: float,
    max_lag: Optional[int] = None,
    dim: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute MSD vs time for 1D or 2D positions assuming constant time step dt.

    Args:
        rel_positions: (N,) array for 1D or (N,2) array for 2D relative positions
        dt: time step between consecutive frames
        max_lag: maximum lag (in frames) to compute; if None, uses N-1
        dim: dimensionality (1 or 2)

    Returns:
        t: time differences (Δt) corresponding to each lag
        msd: mean squared displacement for each Δt
        msd_se: standard error of MSD for each Δt
    """
    n = rel_positions.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 positions to compute MSD")

    if max_lag is None:
        max_lag = n - 1
    max_lag = min(max_lag, n - 1)

    lags = np.arange(1, max_lag + 1)
    msd = np.zeros_like(lags, dtype=float)
    msd_se = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        if dim == 1:
            diffs = rel_positions[lag:] - rel_positions[:-lag]
            sq = diffs**2
        elif dim == 2:
            diffs = rel_positions[lag:] - rel_positions[:-lag]
            sq = np.sum(diffs**2, axis=1)
        else:
            raise ValueError("dim must be 1 or 2")

        if sq.size:
            msd[i] = np.mean(sq)
            msd_se[i] = np.std(sq, ddof=1) / np.sqrt(len(sq)) if len(sq) > 1 else 0.0

    # convert from frame lag → physical time difference
    t = lags * dt
    return t, msd, msd_se


def estimate_diffusion_coefficient_time(
    t: np.ndarray,
    msd: np.ndarray,
    msd_se: Optional[np.ndarray] = None,
    dim: int = 1,
) -> Tuple[float, float, float, float]:
    """
    Estimate diffusion coefficient D from MSD(t) = 2 * dim * D * t + intercept.

    Returns:
        D: estimated diffusion coefficient
        slope: fitted slope of MSD vs time
        intercept: fitted intercept
        D_unc: uncertainty of D from slope covariance
    """
    if msd_se is not None and np.any(msd_se > 0):
        mask = np.isfinite(msd_se) & (msd_se > 0)
        if np.count_nonzero(mask) >= 2:
            w = 1.0 / msd_se[mask]
            coeffs, cov = np.polyfit(t[mask], msd[mask], 1, w=w, cov=True)
        else:
            coeffs, cov = np.polyfit(t, msd, 1, cov=True)
    else:
        coeffs, cov = np.polyfit(t, msd, 1, cov=True)

    slope, intercept = coeffs
    slope_unc = np.sqrt(cov[0, 0])
    D = slope / (2 * dim)
    D_unc = slope_unc / (2 * dim)
    return D, slope, intercept, D_unc


def fit_gaussian(data: np.ndarray) -> Dict[str, float]:
    """Fit a Gaussian distribution to 1D data and return mu and sigma.

    Uses sample mean and sample standard deviation (MLE / unbiased as needed).
    Also returns the number of points.
    """
    data = np.asarray(data)
    n = data.size
    if n == 0:
        return {'n': 0, 'mu': 0.0, 'sigma': 0.0}
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=0))
    return {'n': n, 'mu': mu, 'sigma': sigma}


def normality_tests(data: np.ndarray) -> Dict[str, float]:
    """Perform simple normality tests on 1D data.

    Returns a dict containing test names and p-values. If SciPy is not
    available the function returns an empty dict.
    """
    res: Dict[str, float] = {}
    data = np.asarray(data)
    if data.size < 3:
        return res
    try:
        # D'Agostino-Pearson test (normaltest) and Kolmogorov-Smirnov against fitted normal
        k2, p_norm = stats.normaltest(data)
        res['normaltest_pvalue'] = float(p_norm)
    except Exception:
        pass
    try:
        # KS test against a normal with sample mean/std
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=0))
        if sigma <= 0:
            res['ks_pvalue'] = 0.0
        else:
            _, pks = stats.kstest(data, 'norm', args=(mu, sigma))
            res['ks_pvalue'] = float(pks)
    except Exception:
        pass
    try:
        # Shapiro-Wilk (limited to n<=5000 typically)
        w, psh = stats.shapiro(data)
        res['shapiro_pvalue'] = float(psh)
    except Exception:
        pass
    return res


def fit_bivariate_gaussian(positions: np.ndarray) -> Dict[str, object]:
    """Fit a bivariate Gaussian to (N,2) positions and return mean and covariance.

    Returns dict with keys: 'n', 'mean' (length-2 list), and 'cov' (2x2 list).
    """
    pos = np.asarray(positions)
    if pos.size == 0:
        return {'n': 0, 'mean': [0.0, 0.0], 'cov': [[0.0, 0.0], [0.0, 0.0]]}
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions must be (N,2)")
    xm = float(pos[:, 0].mean())
    ym = float(pos[:, 1].mean())
    cov = np.cov(pos[:, 0], pos[:, 1], bias=True)
    return {'n': int(pos.shape[0]), 'mean': [xm, ym], 'cov': cov.tolist()}


if __name__ == "__main__":
    # quick smoke test
    import numpy as _np
    p = _np.array([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=float)
    rel = relative_positions(p)
    steps = step_displacements(rel)
    print("rel:\n", rel)
    print("steps:\n", steps)
    lags, msd, msd_se = mean_squared_displacement(rel)
    print("msd:", msd)
    print("msd_se:", msd_se)