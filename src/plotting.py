"""Plotting helpers for thermal motion analysis.

Creates and saves common diagnostic plots.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional
import os
from scipy.stats import multivariate_normal

def plot_xy_time_series(times: np.ndarray, positions: np.ndarray, outpath: str):
    """Plot X and Y as functions of time (no markers, thin lines).
    """
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4))
    ax[0].plot(times, positions[:, 0], linestyle='-', linewidth=0.8)
    ax[0].set_ylabel('X (pixels)')
    ax[1].plot(times, positions[:, 1], linestyle='-', linewidth=0.8)
    ax[1].set_ylabel('Y (pixels)')
    ax[1].set_xlabel('Frame')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_radial(times: np.ndarray, radial: np.ndarray, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(times, radial, linestyle='-', linewidth=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Radial displacement (pixels)')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_step_histogram(step_lengths: np.ndarray, outpath: str, bins: int = 30):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(step_lengths, bins=bins, density=True, alpha=0.7)
    ax.set_xlabel('Step length (pixels)')
    ax.set_ylabel('Probability density')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_hist_with_gaussian(data: np.ndarray, outpath: str, bins: int = 30, label: str = "Value"):
    """Plot histogram of `data` with an overlaid Gaussian PDF computed from sample mean/std."""
    data = np.asarray(data)
    if data.size == 0:
        return
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=0))
    fig, ax = plt.subplots(figsize=(5, 3))
    counts, edges, _ = ax.hist(data, bins=bins, density=True, alpha=0.6, label='data')
    # overlay Gaussian
    xs = np.linspace(edges[0], edges[-1], 200)
    if sigma > 0:
        ys = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
        ax.plot(xs, ys, '-', linewidth=0.9, label=f'Gaussian fit (mu={mu:.2f}, sigma={sigma:.2f})')
    ax.set_xlabel(label or 'Value')
    ax.set_ylabel('Probability density')
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_msd(msd_lags: np.ndarray, msd: np.ndarray, slope: float, outpath: str, dt: float = 1.0):
    t = msd_lags * dt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(t, msd, linestyle='-', linewidth=0.8, label='MSD')
    ax.plot(t, slope * t, linestyle='-', linewidth=0.8, label=f'Fit: slope={slope:.3f}')
    ax.set_xlabel('Lag time (frames)')
    ax.set_ylabel('MSD (pixels^2)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_polar_trajectory(positions: np.ndarray, outpath: str):
    """Plot trajectory in polar coordinates (theta, r) connected by a thin line.

    Positions are expected to be relative positions (origin at 0,0). Theta is
    computed as arctan2(y, x) and r as sqrt(x^2 + y^2).
    """
    x = positions[:, 0]
    y = positions[:, 1]
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='polar')
    # Use thin solid line, no markers
    ax.plot(theta, r, linestyle='-', linewidth=0.8)
    ax.set_title('Polar trajectory')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_overlay_trajectories(list_of_positions: Sequence[np.ndarray], labels: Sequence[str], outpath: str):
    """Overlay multiple XY trajectories on a single Cartesian plot.

    Each element of list_of_positions is an (N_i, 2) array. Labels is a list of
    same-length strings for the legend.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for pos, lab in zip(list_of_positions, labels):
        ax.plot(pos[:, 0], pos[:, 1], linestyle='-', linewidth=0.6, label=lab)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_aspect('equal', 'box')
    if labels:
        ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_bivariate_surface(
    positions: np.ndarray,
    outpath: str,
    grid_size: int = 100,
    cmap: str = 'viridis',
    overlay_trajectories: Optional[Sequence[np.ndarray]] = None,
    end_points: Optional[np.ndarray] = None,
    alpha: float = 0.6,
):
    """Create a 3D surface (and 2D contour) of the fitted bivariate Gaussian over X-Y.

    Positions should be relative positions (N,2). The function fits a Gaussian
    (mean and covariance) and evaluates the pdf on a grid for plotting.
    """
    if positions.size == 0:
        return
    x = positions[:, 0]
    y = positions[:, 1]
    xm = x.mean()
    ym = y.mean()
    cov = np.cov(np.vstack([x, y]), bias=True)
    # convert covariance to a plain Python nested list to satisfy type checkers
    # that may not accept numpy.ndarray for the `cov` parameter
    cov_list = cov.tolist()

    # grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    # expand a bit
    dx = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    dy = (ymax - ymin) * 0.1 if ymax > ymin else 1.0
    xs = np.linspace(xmin - dx, xmax + dx, grid_size)
    ys = np.linspace(ymin - dy, ymax + dy, grid_size)
    Xg, Yg = np.meshgrid(xs, ys)

    # compute pdf values
   
    rv = multivariate_normal(mean=[xm, ym], cov=cov_list)
    Z = rv.pdf(np.dstack((Xg, Yg)))

    # 3D surface (with transparency)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xg, Yg, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('pdf')

    # overlay end points if provided
    if end_points is not None and end_points.size:
        ep = np.asarray(end_points)
        ax.scatter(ep[:, 0], ep[:, 1], np.max(Z) * 1.05, c='red', s=20, label='end points')
        try:
            ax.legend()
        except Exception:
            pass

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

    # 2D contour + trajectories overlay (XY plane)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    # contour levels
    levels = np.linspace(np.min(Z), np.max(Z), 8)
    cs = ax2.contourf(Xg, Yg, Z, levels=levels, cmap=cmap, alpha=alpha)
    # add colorbar for the density contour
    try:
        fig2.colorbar(cs, ax=ax2)
    except Exception:
        pass
    # overlay trajectories if provided
    if overlay_trajectories:
        for pos in overlay_trajectories:
            ax2.plot(pos[:, 0], pos[:, 1], '-', linewidth=0.6, color='k', alpha=0.8)
    # overlay end points
    if end_points is not None and end_points.size:
        ep = np.asarray(end_points)
        ax2.scatter(ep[:, 0], ep[:, 1], c='red', s=20)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal', 'box')
    fig2.tight_layout()
    # save with a suffix
    out2 = os.path.splitext(outpath)[0] + '_xy_contour.png'
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)



if __name__ == "__main__":
    import numpy as _np
    t = _np.arange(10, dtype=float)
    pos = _np.cumsum(_np.random.randn(10, 2), axis=0)
    plot_xy_time_series(t, pos, 'tmp_xy.png')
    plot_radial(t, _np.linalg.norm(pos - pos[0], axis=1), 'tmp_radial.png')
    plot_step_histogram(_np.linalg.norm(_np.diff(pos, axis=0), axis=1), 'tmp_hist.png')