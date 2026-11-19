# Generative Nonparametric Conditional Density Estimation

A benchmark comparing traditional statistical methods and generative modeling approaches for conditional density estimation.

## Installation

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/chiatsewang/generative-nonparametric-cde.git
cd generative-nonparametric-cde

# Install the package
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install FlexCode (optional, for FlexCode experiments)
pip install -e ".[flexcode]"

# Setup pre-commit hooks (optional)
pre-commit install
```

### Requirements

- Python >= 3.11
- PyTorch >= 1.11.0
- NumPy, SciPy, matplotlib
- See [pyproject.toml](pyproject.toml) for full dependency list

## Available Methods

- **HY** - Hall--Yao conditional distribution estimation
- **FlexCode** - Flexible conditional density estimation
- **DeepCDE** - Deep Conditional Density Estimation
- **GCDS** - Generative Conditional Density Sampling
- **DDPM** - Denoising Diffusion Probabilistic Models
- **SRM** - Neural conditional generator (PyTorch; CUDA is used automatically when available)
- **CTMTF** - Conditional transformation models with transformation forests (R: `trtf`, `tram`, `mlt`)
- **DRF** - Distributional random forests (R: `drf`)

### R setup (CTMTF and DRF)

CTMTF and DRF run through `Rscript`. The required R packages are listed in
`R/install.R`, and `make install-r` installs the ones that are missing from
CRAN. R 4.5 or newer is needed. `mlt` 1.9-0 requires R 4.6, so on R 4.5 use
`mlt` 1.8-4. `mlt` depends on `nloptr`, which some systems need to install
separately because it can fail to compile from source.

`reticulate` uses the Python interpreter that runs the Python wrapper scripts.
Each wrapper passes `sys.executable` to the R scripts, which call
`reticulate::use_python(..., required = TRUE)`, so run `make` from the shell
where the project environment is active.

## Project Structure

```text
.
├── condgen_benchmark/    # Main package
│   ├── data/             # Data loading and generation
│   └── methods/          # CDE methods (ctmtf, ddpm, deepcde, drf, gcds, hy, srm)
├── scripts/              # Training and evaluation scripts
├── datasets/             # Generated datasets
└── workspaces/           # Model checkpoints and results
```

## License

GPL-2.0 - see [LICENSE](LICENSE)

## HY (Hall--Yao implementation)

The `hy` method uses
batched PyTorch/CUDA Adam theta estimation, automatic
**per-seed** selection of `(h, H)` using validation Energy Score (the 1D CRPS), and a shared
conditional sampler for tuning and final sampling.

```bash
make generate DATA_MODEL=U7 SEED=230 TRAIN=2000 VALID=600 TEST=1000
make train-hy DATA_MODEL=U7 SEED=230
make sample-hy DATA_MODEL=U7 SEED=230 SAMPLES=300
make eval-hy DATA_MODEL=U7 SEED=230
make conclude-hy
```

To bypass the automatic 21-pair search, specify both bandwidths:

```bash
make train-hy DATA_MODEL=U7 SEED=230 HY_H=0.7 HY_H_CDF=0.8
```

Tuning controls (Makefile defaults): `HY_N_SUB_VAL=200` validation points,
`HY_NUM_Y_GRID_TUNING=50` CDF grid points. The default six theta fits
use batched Adam on CUDA when available, otherwise on CPU. Tuning evaluates the Energy
Score directly from the estimated inverse CDF without Monte Carlo draws.
Final sampling uses `SAMPLES` for draws per test predictor and
`HY_NUM_Y_GRID=200` for CDF grid points (`--num-samples`, `--ny` in Python).
The grid is computed from the *response* `Y_train`, and the CDF is made
nondecreasing with a cumulative maximum before inverse sampling.
Training saves `theta.npz` (including the selected bandwidths), `config.yaml`,
and `tuning.csv` if tuning was automatic, under `workspaces/hy/...`.

The `hy` theta estimator uses PyTorch/Adam with automatic CPU/CUDA selection. The `hy` conditional
sampler still uses Numba on CPU; evaluation retains its existing CPU path.

### HY tuning speed and timing update

The HY theta objective now uses sparse CSR multiplication for its ball
membership matrix. When several candidate `H` values share the same `h`,
tuning also reuses the response order and the training/validation projections
onto the fitted theta, rather than rebuilding them for each candidate.
The same conditional-sampling core remains in use for tuning and final
sampling; no statistical rule or bandwidth grid has been changed in this update.

The training output records `t_opt` (wall-clock duration of the six theta
fits, which may now run concurrently) and `t_total` (entire fit/tuning time) in
`theta.npz`, and corresponding
`theta_fit_seconds`/`total_fit_seconds` fields in `config.yaml`.

For the new HY implementation, `fit_thetas()` returns
`(fitted_theta_for_each_h, theta_fit_seconds, theta_fit_diagnostics)`:
`fitted_theta_for_each_h[h]` is the fitted theta vector for that `h`, and
`theta_fit_diagnostics` records `h`, `epochs`, and `best_S_theta`.
The tuning result uses `bandwidth_energy_scores` for its 21 `(h, H, EnergyScore)`
entries. The YAML diagnostics key is `theta_fit_diagnostics`; the saved `theta.npz` keys `theta`, `h`, `H`, `t_opt`,
`t_total`, `success`, and the `tuning.csv` format are unchanged.

### 2026-10-20: Original U7 restored as O7; HY CDF sampling adjustment

The synthetic model previously called `U7` (before its October 2026
replacement) is available as `O7`: `X ~ N(0,1)` and
`Y = exp(sin(20 X + epsilon))`, `epsilon ~ N(0,1)`, with the original
Bessel-series `mean_std()` implementation. The current three-predictor `U7`
remains unchanged. `O7` is available in the generator's `--data-name` choices:

```bash
make generate DATA_MODEL=O7 SEED=230 TRAIN=1000 VALID=300 TEST=500
```

`O7` was **not** added to `ALL_MODELS` or method applicability rules in
`skip_rules.mk`: it is an opt-in historical dataset, not part of default
benchmark sweeps. The Makefile was not changed.

The new `hy` sampler no longer discards a valid estimated CDF merely because
it has fewer than ten distinct levels. Inverse-CDF interpolation now uses
all valid monotone CDF values after the existing cumulative-maximum correction.
The safeguard for a nonfinite or near-zero normalization denominator remains.
Theta fitting is already cached by `h`: the 21 `(h,H)` candidates invoke
the GPU theta estimator for only the six distinct `h` values. The existing sampler
cache also reuses sorting and projections for each fixed `h`.

### FlexCode: inherit the training grid at sampling time

FlexCode training uses `--ny=200` by default and stores the response grid in
`model.pth` and `model_tuned.pth`. `sample-flex` now reads the grid from the
selected checkpoint and passes its length to `model.predict`; it also verifies
that the returned density columns and grid coordinates match the saved grid.
The Makefile setting `FLEX_NUM_Y_GRID` (default 200) is passed as `--ny`
to `train-flex`, not to `sample-flex`. The sampling `--ny` option is removed,
so the sampling command cannot redefine training-time choices. The number
of generated responses still comes from `SAMPLES` / `--num-samples`.
Checkpoints without a stored response grid now raise an explicit error rather
than deriving the response support from test labels.

```bash
make train-flex DATA_MODEL=U1 SEED=230 FLEX_NUM_Y_GRID=200
make sample-flex DATA_MODEL=U1 SEED=230 SAMPLES=300
```

### 2026-10-21: HY theta estimation on CUDA with batched Adam

The default `make train-hy` now estimates theta for the six independent `h` values
together on **PyTorch float32 autograd and Adam**, automatically using CUDA if available and CPU otherwise. The full
leave-two-out Hall--Yao objective is retained (with a float32 numerical guard
near singular local-linear denominators). The ball-incidence matrix remains
sparse in GPU memory. `X`, `Y`, and other fixed tensors are transferred once;
all Adam iterations remain on the selected device. It learns one parameter
vector per `h` via `theta=w/sqrt(sum(w*w)+1e-5)`. Each row has an independent
best parameter and early stopping counter; finished rows are removed from
subsequent objective evaluations.

Defaults: `HY_ADAM_LR=0.01`, `HY_ADAM_MAX_EPOCHS=300`,
`HY_ADAM_PATIENCE=30`. Device selection is automatic (CUDA if available, otherwise CPU). Any strictly lower finite loss
resets that theta's patience.

The **21-pair Energy Score search, CPU conditional sampler, evaluation and
output file formats** are unchanged. There is no need to retrain other methods.
The optimized theta and bandwidths continue to be saved to
`workspaces/hy/<data-name>/seed_<seed>/theta.npz`. Per-h epoch counts and
lowest objective values appear under `theta_fit_diagnostics` in `config.yaml`.

The HY training CLI has no device flag. For a CPU-only comparison on a CUDA machine, hide CUDA from the process, e.g. `CUDA_VISIBLE_DEVICES="" make train-hy DATA_MODEL=U1 SEED=230`.

### 2026-10-21: HY GPU-only cleanup

The adopted `hy` implementation now contains only the PyTorch/Adam theta
optimizer. Its obsolete Numba/SLSQP theta criterion,
optimizer-selection flags, and CPU-only regression tests have been removed.
The GPU fixed-term preparation is independent of the legacy CPU estimator.
The CPU conditional sampler, Energy Score tuning, and benchmark evaluation were not changed in that update.


### HY source structure (2026-10-21)

`condgen_benchmark/methods/hy/loss.py` contains `HallYaoLoss`, the
differentiable Hall--Yao objective. `fit.py` contains `fit_thetas`, the
batched Adam optimizer for the six candidate theta parameters.
`tuning.py` selects `(h,H)` via Energy Score, and `sampler.py` retains the
existing CPU conditional sampler. This is a naming and module split only;
Torch sparse CSR computation and warning behavior are unchanged.

### 2026-10-21: Automatic device selection for HY and SRM

HY theta estimation and SRM sampling now both detect CUDA automatically,
falling back to CPU if unavailable. No device-related Makefile or Python CLI
arguments are needed. Other method algorithms and defaults are unchanged.
