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

    # Physical constants and instrumentation parameters (user-provided)
    # Viscosity mu and uncertainty
    MU_VAL = 0.0941192
    MU_UNC = 0.0047 
    # particle radius in micrometers and uncertainty
    R_VAL_UM = 0.8
    R_UNC_UM = 0.05
    # pixel-to-length scale: 0.19 um per 20 pixels

    PIXEL_SCALE_M_PER_PX = 0.1204 * 1e-6  # meters per pixel
    PIXEL_SCALE_M2_PER_PX = PIXEL_SCALE_M_PER_PX ** 2
    # Boltzmann constant and temperature (K)
    K_B = 1.380649e-23
    T_K = 296.5

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

        # MSD and diffusion estimate (includes per-lag MSD standard errors)
        lags, msd, msd_se = mean_squared_displacement(rel)
        if lags.size:
            # pass dt and per-lag errors to the estimator for a properly
            # scaled fit and uncertainty. lags are in frames; estimator
            # will convert to time using dt.
            D_est_pix2_per_s, slope, D_unc_pix2_per_s = estimate_diffusion_coefficient(lags, msd, dt=args.dt, msd_se=msd_se, dim=1)
        else:
            raise ValueError("No lags computed for MSD; cannot estimate diffusion coefficient.")
        stats['D_est_pixels2_per_s'] = float(D_est_pix2_per_s) 
        stats['D_est_unc_pixels2_per_s'] = float(D_unc_pix2_per_s)
        stats['msd_slope_pixels2_per_s'] = float(slope)

        # Convert pixel^2/s to m^2/s using pixel scale
        scale_m = PIXEL_SCALE_M2_PER_PX
        scale_final = scale_m 
        D_est_final_s = float(D_est_pix2_per_s * scale_final)
        D_unc_final_s = float(D_unc_pix2_per_s * scale_final)
        stats['D_est_m2_s'] = D_est_final_s
        stats['D_est_unc_m2_s'] = D_unc_final_s

        # Compute k from D = k * T / gamma  => k = D * gamma / T
        mu_val = MU_VAL
        mu_unc = MU_UNC
        r_val = R_VAL_UM * 1e-6
        r_unc = R_UNC_UM * 1e-6
        gamma = 6.0 * np.pi * mu_val * r_val
        k_est = D_est_final_s * gamma / T_K
        # Propagate uncertainties (assume independent): contributions from D, mu, r
        dk_dD = gamma / T_K
        dk_dmu = D_est_final_s * 6.0 * np.pi * r_val / T_K
        dk_dr = D_est_final_s * 6.0 * np.pi * mu_val / T_K
        sigma_D = D_unc_final_s
        sigma_mu = mu_unc
        sigma_r = r_unc
        k_unc = float(np.sqrt((dk_dD * sigma_D) ** 2 + (dk_dmu * sigma_mu) ** 2 + (dk_dr * sigma_r) ** 2))

        stats['k_est_J_per_K'] = float(k_est)
        stats['k_est_unc_J_per_K'] = float(k_unc)

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
        plot_msd(lags, msd, slope, os.path.join(file_outdir, 'msd.png'), dt=args.dt, dim=1, msd_se=msd_se)
        # polar trajectory for each file
        plot_polar_trajectory(rel, os.path.join(file_outdir, 'polar_trajectory.png'))

        all_rel_positions.append(rel)
        all_labels.append(label)

    # After processing all files, prepare and save a results table (CSV) containing D and k estimates
    try:
        import csv
        csvpath = os.path.join(args.outdir, 'k_estimates_table.csv')
        with open(csvpath, 'w', newline='') as csvf:
            writer = csv.writer(csvf)
            header = [
                'label', 'n_frames',
                'D_m2_s', 'D_unc_m2_s',
                'k_J_per_K', 'k_unc_J_per_K'
            ]
            writer.writerow(header)
            print('\nCalculated D and k estimates:')
            print(', '.join(header))
            for label, st in summaries.items():
                row = [
                    label,
                    st.get('n_frames', ''),
                    st.get('D_est_m2_s', ''),
                    st.get('D_est_unc_m2_s', ''),
                    st.get('k_est_J_per_K', ''),
                    st.get('k_est_unc_J_per_K', ''),
                ]
                writer.writerow(row)
                # print a concise table line
                print(', '.join([str(x) for x in row]))

            # print avg k
            k_values = [st['k_est_J_per_K'] for st in summaries.values() if 'k_est_J_per_K' in st]
            if k_values:
                avg_k = sum(k_values) / len(k_values)
                print(f"Average k over all trials: {avg_k} J/K")
        print(f"Saved D and k estimates to: {csvpath}")
    except Exception as e:
        print(f"Warning: failed to write k estimates table: {e}")

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
            # Theoretical diffusion calculation from D = k_B T / (6 pi mu r)
            # Use user-provided viscosity and radius with uncertainties (assumed units: mu in Pa*s, r in micrometers)
            # Values provided by user:

            r_val_um = 0.8
            r_unc_um = 0.05
            # convert radius to meters
            r_val = r_val_um * 1e-6
            r_unc = r_unc_um * 1e-6
            k_B = 1.380649e-23
            T = 298.15
            gamma = 6.0 * np.pi * MU_VAL * r_val
            D_theory = k_B * T / gamma
            # uncertainty propagation: relative uncertainties add in quadrature for mu and r
            # dD/D = sqrt((dmu/mu)^2 + (dr/r)^2)
            rel_unc = np.sqrt((MU_UNC / MU_VAL) ** 2 + (r_unc / r_val) ** 2)
            D_theory_unc = D_theory * rel_unc
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
                'D_theory_m2_s': float(D_theory),
                'D_theory_unc_m2_s': float(D_theory_unc),
                'mu_used_Pa_s': MU_VAL,
                'mu_unc_Pa_s': MU_UNC,
                'r_used_m': r_val,
                'r_unc_m': r_unc,
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
