"""Main runner for analyzing tracking data.

Usage: python main.py --input <path/to/file> [--outdir results/analysis]
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from src.io import load_two_column_file
from src.analysis import (
    relative_positions,
    step_displacements,
    step_lengths,
    radial_displacement,
    mean_squared_displacement,
    estimate_diffusion_coefficient,
    fit_gaussian,
    normality_tests,
    fit_bivariate_gaussian,
)
from src.plotting import (
    plot_xy_time_series,
    plot_radial,
    plot_step_histogram,
    plot_msd,
    plot_polar_trajectory,
    plot_overlay_trajectories,
    plot_hist_with_gaussian,
    plot_bivariate_surface,
)

from scipy.stats import multivariate_normal

def main():
    p = argparse.ArgumentParser(description="Thermal motion analysis for 2D tracking data")
    p.add_argument("--input", required=False, help="Path to two-column tracking file (optional)")
    p.add_argument("--indir", default="results", help="Directory to scan for .txt tracking files")
    p.add_argument("--outdir", default="analysis", help="Directory to write results (default: analysis)")
    p.add_argument("--dt", type=float, default=0.5, help="Time between frames (seconds). Default 0.5s")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine files to process: either a single input file, or all .txt in indir
    files = []
    if args.input:
        files = [args.input]
    else:
        import glob
        pattern = os.path.join(args.indir, '*.txt')
        files = sorted(glob.glob(pattern))

    if not files:
        print(f"No input files found (input={args.input}, indir={args.indir}). Exiting.")
        return

    all_rel_positions = []
    all_labels = []

    summaries = {}
    for filepath in files:
        arr = load_two_column_file(filepath)
        if arr.shape[0] == 0:
            print(f"Skipping empty or invalid file: {filepath}")
            continue
        rel = relative_positions(arr)
        steps = step_displacements(rel)
        lengths = step_lengths(steps)
        radial = radial_displacement(rel)

        # Stats
        stats = {
            'n_frames': int(arr.shape[0]),
            'mean_step': float(np.mean(lengths)) if lengths.size else 0.0,
            'median_step': float(np.median(lengths)) if lengths.size else 0.0,
            'std_step': float(np.std(lengths)) if lengths.size else 0.0,
            'mean_radial': float(np.mean(radial)) if radial.size else 0.0,
        }

        # MSD and diffusion estimate
        lags, msd = mean_squared_displacement(rel)
        D_est, slope = estimate_diffusion_coefficient(lags, msd, dt=args.dt) if lags.size else (0.0, 0.0)
        stats['D_est_pixels2_per_frame'] = float(D_est)
        stats['msd_slope'] = float(slope)

        # Use numeric label for each trial (order independent)
        numeric_label = len(all_rel_positions) + 1
        label = str(numeric_label)
        summaries[label] = stats

        # Print per-file summary
        print(f"Summary for {label}:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # save per-file outputs
        file_outdir = os.path.join(args.outdir, label)
        os.makedirs(file_outdir, exist_ok=True)
        import json
        with open(os.path.join(file_outdir, 'summary.json'), 'w') as fh:
            json.dump(stats, fh, indent=2)

        frames = np.arange(arr.shape[0], dtype=float)
        plot_xy_time_series(frames, rel, os.path.join(file_outdir, 'xy_time_series.png'))
        plot_radial(frames, radial, os.path.join(file_outdir, 'radial.png'))
        plot_step_histogram(lengths, os.path.join(file_outdir, 'step_hist.png'))
        # Also analyze step components for Gaussianity (x and y)
        step_x = steps[:, 0] if steps.size else np.zeros((0,))
        step_y = steps[:, 1] if steps.size else np.zeros((0,))
        # Save histograms with Gaussian overlays
        gx = fit_gaussian(step_x)
        gy = fit_gaussian(step_y)
        nx = normality_tests(step_x)
        ny = normality_tests(step_y)
        # store fit/test results
        stats['step_x_fit'] = gx
        stats['step_y_fit'] = gy
        stats['step_x_tests'] = nx
        stats['step_y_tests'] = ny
        plot_hist_with_gaussian(step_x, os.path.join(file_outdir, 'step_x_hist_gauss.png'), label='step_x')
        plot_hist_with_gaussian(step_y, os.path.join(file_outdir, 'step_y_hist_gauss.png'), label='step_y')
        if lags.size:
            plot_msd(lags, msd, slope, os.path.join(file_outdir, 'msd.png'), dt=args.dt)
        # polar trajectory for each file
        plot_polar_trajectory(rel, os.path.join(file_outdir, 'polar_trajectory.png'))

        all_rel_positions.append(rel)
        all_labels.append(label)

    # Combined overlays saved in outdir
    if all_rel_positions:
        plot_overlay_trajectories(all_rel_positions, all_labels, os.path.join(args.outdir, 'all_trajectories.png'))
        # For polar overlay, we will plot each trajectory separately on the same polar axes
        # create a simple polar overlay by plotting theta/r lines
        # We reuse plot_overlay_trajectories but in polar form we create a figure here
        try:
            # create polar overlay
            import matplotlib.pyplot as _plt
            fig = _plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='polar')
            for rel, lab in zip(all_rel_positions, all_labels):
                theta = np.arctan2(rel[:, 1], rel[:, 0])
                r = np.hypot(rel[:, 0], rel[:, 1])
                ax.plot(theta, r, linestyle='-', linewidth=0.6, label=lab)
            ax.set_title('All polar trajectories')
            ax.legend(fontsize='small')
            fig.tight_layout()
            figpath = os.path.join(args.outdir, 'all_trajectories_polar.png')
            fig.savefig(figpath, dpi=150)
            _plt.close(fig)
        except Exception:
            # if plotting fails for some reason, continue
            pass

        # Aggregate positions across all trials (all times) and analyze distribution
        try:
            agg = np.vstack(all_rel_positions)
            agg_x = agg[:, 0]
            agg_y = agg[:, 1]
            agg_x_fit = fit_gaussian(agg_x)
            agg_y_fit = fit_gaussian(agg_y)
            agg_x_tests = normality_tests(agg_x)
            agg_y_tests = normality_tests(agg_y)
            # Save histograms with Gaussian overlays for aggregated positions
            plot_hist_with_gaussian(agg_x, os.path.join(args.outdir, 'agg_positions_x_hist_gauss.png'), label='X position')
            plot_hist_with_gaussian(agg_y, os.path.join(args.outdir, 'agg_positions_y_hist_gauss.png'), label='Y position')
            # bivariate fit
            bivar = fit_bivariate_gaussian(agg)
            # compute average log-likelihood and information criteria

            cov_arr = np.asarray(bivar['cov'], dtype=float)
            rv = multivariate_normal(mean=bivar['mean'], cov=cov_arr) # type: ignore
            logpdfs = rv.logpdf(agg)
            ll = float(np.sum(logpdfs))
            
            n_pts = int(agg.shape[0])
            k_params = 5  # 2 means + 3 unique cov entries
            aic = 2 * k_params - 2 * ll
            bic = k_params * np.log(n_pts) - 2 * ll
            agg_summary = {
                'n_points': n_pts,
                'x_fit': agg_x_fit,
                'y_fit': agg_y_fit,
                'x_tests': agg_x_tests,
                'y_tests': agg_y_tests,
                'bivariate_fit': bivar,
                'log_likelihood': ll,
                'aic': aic,
                'bic': bic,
            }
            # Save aggregate summary
            import json
            with open(os.path.join(args.outdir, 'aggregate_positions_summary.json'), 'w') as fh:
                json.dump(agg_summary, fh, indent=2)
            # save bivariate surface plot (pass overlay trajectories and end_points)
            # end points: final relative position from each trial
            end_pts = np.vstack([rel[-1] for rel in all_rel_positions if rel.size]) if all_rel_positions else None
            plot_bivariate_surface(
                agg,
                os.path.join(args.outdir, 'agg_positions_bivariate_surface.png'),
                overlay_trajectories=all_rel_positions,
                end_points=end_pts,
                alpha=0.45,
            )
            print(f"Aggregate position analysis saved to: {args.outdir}/aggregate_positions_summary.json")
        except Exception as e:
            print(f"Warning: failed to compute aggregate positions analysis: {e}")

    # Save combined summaries
    try:
        import json
        with open(os.path.join(args.outdir, 'summaries.json'), 'w') as fh:
            json.dump(summaries, fh, indent=2)
    except Exception:
        pass

    print(f"Plots and summaries written to: {args.outdir}")


if __name__ == '__main__':
    main()
