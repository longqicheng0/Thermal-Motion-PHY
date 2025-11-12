# Thermal Motion Analysis — Methods & Math

This toolbox analyzes 2D particle/object tracking for thermal (Brownian-like) motion. It loads two-column X,Y files, sets the initial position to the origin, computes step and radial displacements, mean-squared displacement (MSD), fits Gaussian models to step/position distributions, and visualizes results—including a 3D bivariate Gaussian surface and XY contour with trajectory overlays.

---

## Quick Usage

Install dependencies and run the analysis:

```bash
python3 -m pip install -r requirements.txt
python3 main.py --indir results --outdir analysis
````

Default parameters:

* `dt = 0.5` seconds (time between frames)

Outputs are written to `analysis/`, including per-trial folders with PNGs and `summary.json`, as well as aggregate summaries and plots (`aggregate_positions_summary.json` and `agg_positions_bivariate_surface.png`).

---

## Methods & Math

Let tracked positions be 

$$
\mathbf{p}_i = (x_i, y_i)
$$

 for frames 
$$
i = 0 \dots N-1
$$

.

### Relative Positions

Set the initial position to the origin:

$$
\mathbf{r}_i = \mathbf{p}_i - \mathbf{p}_0
$$

### Step Displacements

$$
\Delta\mathbf{r}*i = \mathbf{r}*{i+1} - \mathbf{r}_i, \quad i = 0 \dots N-2
$$

Step length:

$$
s_i = |\Delta \mathbf{r}_i| = \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}
$$

Step components are fitted with a univariate Gaussian:

$$
f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Parameters are estimated as sample mean and population std (`ddof=0`, MLE).

### Radial Displacement

$$
\rho_i = |\mathbf{r}_i|
$$

### Mean-Squared Displacement (MSD)

$$
\mathrm{MSD}(\tau) = \langle |\mathbf{r}_{t+\tau} - \mathbf{r}_t|^2 \rangle_t
$$

For isotropic 3D diffusion:

$$
\mathrm{MSD}(t) = 6 D t + C
$$

Estimate diffusion coefficient $D$ by fitting a line to MSD vs. time:
$$
\hat D = \frac{\text{slope}}{4}
$$

### Gaussian / Normality Testing

For 1D distributions (step components, X/Y positions) we report:

* Sample mean & std
* D'Agostino-Pearson omnibus test (`normaltest`) p-value
* Kolmogorov-Smirnov test (`kstest`) p-value
* Shapiro-Wilk p-value (if available)

### Bivariate Gaussian (Aggregate Positions)

Bivariate PDF:

$$
f(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = \frac{1}{2\pi\sqrt{|\Sigma|}} \exp\left(-\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu}) \right)
$$

* $$
  \boldsymbol{\mu} = (\bar x, \bar y)$$, $$\Sigma$$ = sample covariance (3 independent parameters)
  $$
* Log-likelihood:

$$
\ell = \sum_{i=1}^{n} \log f(\mathbf{x}_i; \hat\mu, \hat\Sigma)
$$

Model selection metrics:

* Akaike Information Criterion (AIC): 
  $$
  \mathrm{AIC} = 2k - 2\ell
  $$
* Bayesian Information Criterion (BIC): 
  $$
  \mathrm{BIC} = k \log n - 2\ell
  $$

where 

$$
k=5
$$

 (2 means + 3 covariance parameters) and $$n$$ is the number of points.

### 3D Bivariate Surface & XY Contours

Produces:

* 3D surface of the bivariate PDF
* 2D XY contour with optional trajectory and end-point overlays

---

## Outputs

* `analysis/<trial>/summary.json` — per-trial statistics (means, D, Gaussian fit, normality tests)
* `analysis/<trial>/*.png` — time-series, radial, histograms, MSD, polar plots
* `analysis/agg_positions_x_hist_gauss.png`, `agg_positions_y_hist_gauss.png` — aggregate univariate histograms
* `analysis/aggregate_positions_summary.json` — aggregate fit parameters, log-likelihood, AIC/BIC, p-values
* `analysis/agg_positions_bivariate_surface.png` and corresponding `*_xy_contour.png` — bivariate surface & contour with overlays

---

## Notes & Caveats

* Population std (`ddof=0`) is used for Gaussian overlay (MLE). Change to `ddof=1` for unbiased reporting.
* Normality tests are sample-size sensitive; p-values are indicators, not proofs.

Optional: CSV or LaTeX-friendly tables for all fit parameters can be generated for manuscripts.

---

## CLI Example

```bash
python main.py --input results/t2d.txt --outdir analysis --dt 0.5
```
