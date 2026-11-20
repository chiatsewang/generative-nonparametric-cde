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

## Running experiments

Every experiment runs through `make` from the repository root. A run has five
steps. `generate` simulates a dataset, `train-<algo>` fits a method,
`sample-<algo>` draws responses from the fitted conditional distribution at
each test predictor, `eval-<algo>` scores those draws, and `conclude` collects
the scores of all seeds into an Excel workbook. The method names used in
targets are `ddpm`, `gcds`, `hy`, `flex`, `deepcde`, `srm`, `ctmtf` and `drf`.

These commands run DDPM on one model and one seed.

```bash
make generate DATA_MODEL=U1 SEED=230
make train-ddpm DATA_MODEL=U1 SEED=230
make sample-ddpm DATA_MODEL=U1 SEED=230
make eval-ddpm DATA_MODEL=U1 SEED=230
make conclude-ddpm
```

When `DATA_MODEL` or `SEED` is left off the command line, the `train-<algo>`,
`sample-<algo>`, `eval-<algo>` and `generate-all` targets loop over every
model in `ALL_MODELS` and every seed in `ALL_SEEDS`. Each method target skips
the models it does not apply to (see [Methods](#methods)). The full sweep for
one method runs as follows.

```bash
make generate-all
make train-hy
make sample-hy
make eval-hy
make conclude-hy
```

To run some seeds rather than all 50, override `ALL_SEEDS` on the command
line. For example, the following runs DRF on M2 for the first three seeds and
writes the summary to `workspaces/drf_M2.xlsx`.

```bash
make generate-all DATA_MODEL=M2 ALL_SEEDS="230 460 690"
make train-drf DATA_MODEL=M2 ALL_SEEDS="230 460 690"
make sample-drf DATA_MODEL=M2 ALL_SEEDS="230 460 690"
make eval-drf DATA_MODEL=M2 ALL_SEEDS="230 460 690"
make conclude-drf DATA_MODEL=M2
```

`conclude` summarizes whichever `seed_*` folders it finds, so it needs no seed
list.

`make eval` evaluates every method in `ALL_ALGOS`, or only the one named by
`ALGO`. `make conclude` summarizes all methods and models, and accepts `ALGO`
and `DATA_MODEL` as filters. `make help` lists the targets.

### Key variables

| Variable | Default | Meaning |
|---|---|---|
| `DATA_MODEL` | all of `ALL_MODELS` | Model to generate, train, sample and evaluate (`U1`-`U7`, `M1`-`M4`; `generate` also accepts `O7`) |
| `SEED` | all of `ALL_SEEDS` | Dataset seed; `ALL_SEEDS` holds the 50 seeds `230`, `460`, ..., `11500` |
| `ALGO` | all of `ALL_ALGOS` | Method for `eval` and `conclude` |
| `TRAIN`, `VALID`, `TEST` | `1000`, `300`, `500` | Training, validation and test sample sizes |
| `ORACLE` | `500` | Fresh responses drawn per test predictor for the oracle Energy Score |
| `SAMPLES` | `300` | Responses drawn per test predictor at sampling time |
| `OUTDIR` | `datasets/synthetic` | Folder that holds the generated datasets |

The single-run `generate` target uses `DATA_MODEL=U1` and `SEED=230` when they
are not given. Method settings such as `DDPM_EPOCHS`, `GCDS_LR`, `SRM_BATCH`
and `HY_RADIUS` are Makefile variables with defaults in the `Makefile`, and any
of them can be overridden on the command line.

### Where outputs land

`generate` writes `train.npz`, `valid.npz` and `test.npz` to
`$(OUTDIR)/<model>/seed_<seed>/`. Each file holds the predictors `X` and the
responses `Y`, and `test.npz` also holds the oracle score `ES_oracle`.
`make clean` removes `$(OUTDIR)`.

Each method writes to `workspaces/<algo>/<model>/seed_<seed>/`, relative to the
directory `make` runs in.

| File | Written by | Contents |
|---|---|---|
| `config.yaml` | `train-<algo>` | Arguments of the training run |
| model files | `train-<algo>` | Method-specific, for example `model_last.pth` and `model_best.pth` (DDPM, DeepCDE), `generator.pth` (GCDS, SRM), `model.pth` and `model_tuned.pth` (FlexCode), `theta.npz` (HY), `model.rds` (CTMTF, DRF) |
| `sampled.npz` | `sample-<algo>` | `Y_hat` (draws, shape `(n_test, SAMPLES, response_dim)`), `X` and `Y_true` |
| `eval.npz` | `eval-<algo>` | The metrics described in [Evaluation](#evaluation) |

`conclude` writes one workbook under `workspaces/`. Its name follows the
filters given.

- `make conclude` writes `workspaces/all.xlsx`
- `make conclude ALGO=ddpm` writes `workspaces/ddpm.xlsx`
- `make conclude DATA_MODEL=U1` writes `workspaces/U1.xlsx`
- `make conclude ALGO=ddpm DATA_MODEL=U1` writes `workspaces/ddpm_U1.xlsx`

`conclude-<algo>` is a shortcut for `make conclude ALGO=<algo>`.

## Methods

The PyTorch methods (DDPM, GCDS, DeepCDE, SRM and the HY theta fit) use CUDA
when it is available and the CPU otherwise. No script has a device flag. To
force the CPU on a CUDA machine, hide the GPU from the process, for example
`CUDA_VISIBLE_DEVICES="" make train-hy DATA_MODEL=U1 SEED=230`. CTMTF and DRF
run in R (see [R setup](#r-setup-ctmtf-and-drf)).

**DDPM** is a conditional denoising diffusion model. An MLP conditioned on the
predictors learns to remove noise from the response over `DDPM_TIME_STEP`
diffusion steps (default `300`, linear noise schedule), and sampling runs the
reverse diffusion from Gaussian noise.

**GCDS** trains a conditional generator against a discriminator. The generator
maps the predictors and a three-dimensional noise vector to a response draw.

**DeepCDE** expands the conditional density in a basis (`DEEPCDE_BASIS`,
default `cosine`, with `DEEPCDE_NUM_BASIS` terms, default `31`) on the
response rescaled to its training range. A neural network predicts the basis
coefficients. Sampling evaluates the density on a grid of
`DEEPCDE_NUM_YGRID` points (default `200`) and draws from it.

**FlexCode** also uses a basis expansion (`FLEX_BASIS`, default `cosine`, up to
`FLEX_MAX_BASIS` terms, default `31`), but fits the coefficients by regression
(`FLEX_REG_METHOD`, `rf` for random forests by default or `knn`). Training
also tunes a bump threshold and a sharpening exponent on the validation set
and saves the result as `model_tuned.pth`. Sampling uses the tuned model
because `FLEX_TUNED_MODEL` defaults to `--tuned-model`. See
[FlexCode sampling grid](#flexcode-sampling-grid).

**HY** is the Hall--Yao single-index estimator. It estimates one direction
`theta` so that the response depends on the predictors through `theta^T X`,
then estimates the conditional distribution function by local-linear smoothing
on that projection and samples by inverting it. HY needs a scalar response.
See [HY details](#hy-details).

**SRM** is a neural conditional generator trained by minimizing a Monte Carlo
Energy Score. During training it draws `SRM_TRAIN_SAMPLES` responses (default
`30`) per training point to estimate the score.

**CTMTF** fits a conditional transformation model with a transformation
forest (`traforest` from `trtf`, on a Bernstein basis of order 5, 100 trees).
Sampling draws random probability levels and takes the predicted conditional
quantiles at those levels.

**DRF** fits a distributional random forest (`drf`, splitting rule
`FourierMMD` by default). For each test predictor the forest gives weights
over the training points, and sampling resamples the training responses with
those weights.

### Which models each method runs on

The applicability rules live in `skip_rules.mk`, which both the `Makefile` and
`scripts/conclude_simulation.py` read. The model groups are
`UNIVARIATE_MODELS` (`U1`-`U7`), `MULTIVARIATE_MODELS` (`M1`-`M4`) and
`ALL_MODELS` (both).

| Method | Rule in `skip_rules.mk` | U1 | U2 | U3 | U4 | U5 | U6 | U7 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DDPM | `DDPM_APPLY_MODELS := $(ALL_MODELS)` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| GCDS | `GCDS_APPLY_MODELS := $(ALL_MODELS)` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| DeepCDE | `DEEPCDE_APPLY_MODELS := $(UNIVARIATE_MODELS)` | yes | yes | yes | yes | yes | yes | yes | - | - | - | - |
| FlexCode | `FLEX_APPLY_MODELS := $(UNIVARIATE_MODELS)` | yes | yes | yes | yes | yes | yes | yes | - | - | - | - |
| HY | `HY_APPLY_MODELS := U1 U2 U3 U4 U7` | yes | yes | yes | yes | - | - | yes | - | - | - | - |
| SRM | `SRM_APPLY_MODELS := $(ALL_MODELS)` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| CTMTF | `CTMTF_APPLY_MODELS := $(UNIVARIATE_MODELS)` | yes | yes | yes | yes | yes | yes | yes | - | - | - | - |
| DRF | `DRF_APPLY_MODELS := $(ALL_MODELS)` | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |

Method targets skip every model outside the method's rule, and `conclude`
marks those pairs as `not applicable`.

## Models

The synthetic models are defined in `condgen_benchmark/data/synthetic.py`.
Unless stated otherwise, the predictors are independent standard normal and
`eps` is standard normal noise independent of `X`.

| Model | `X` dim | `Y` dim | Conditional distribution |
|---|---|---|---|
| `U1` | 4 | 1 | `Y = sin(X)^T theta + eps`, with `theta = (0.5, 0.5, 0.5, 0.5)` normalized to unit length |
| `U2` | 4 | 1 | `Y = 0.5 sum_j sin(Z_j X_j) + eps`, with random signs `Z_j` in {-1, +1} |
| `U3` | 4 | 1 | As `U2`, with `Z_j ~ Uniform(0, 1)` |
| `U4` | 5 | 1 | `Y = X1^2 + exp(X2 + X3/3) + X4 - X5 + (0.5 + X2^2/2 + X5^2/2) eps`, heteroscedastic Gaussian noise |
| `U5` | 30 | 1 | `Y = m(X) exp(0.25 eps)`, with `m(X) = 5 + X1^2/3 + X2^2 + X3^2 + X4 + X5` and `eps` a mixture of N(-1, 1) and N(1, 1) whose weight depends on `X1` through a logistic function |
| `U6` | 30 | 1 | `Y = m(X) exp(0.5 eps)`, with the same `m(X)` as `U5` and `eps` an equal mixture of N(-2, 1) and N(2, 1) |
| `U7` | 3 | 1 | Single-index Gaussian model, `Z = beta^T X` with `beta = (1, 2, -1)/sqrt(6)` and `Y = sin(1.5 Z) + (0.7 + 0.2 tanh(Z)) eps` |
| `M1` | 5 | 7 | `Y = (X1^2, ..., X5^2, exp(X2 + X5/3), sin(X4 + X5)) + eps`, with `eps ~ N(0, I_7)` |
| `M2` | 30 | 5 | `X ~ Uniform(0, 1)^30` and `Y` Gaussian with mean zero, unit variances and all correlations equal to `X1` |
| `M3` | 30 | 2 | Bivariate extension of `U5` with correlated mixture errors and two location functions |
| `M4` | 5 | 10 | `X ~ Uniform(-1, 1)^5` and `Y` an eight-component Gaussian mixture whose weights, location and rotation depend on `X` |

`O7` is a generator-only model. It has one predictor `X ~ N(0, 1)` and
`Y = exp(sin(20 X + eps))`, and its conditional mean and standard deviation
come from a truncated modified-Bessel series. `O7` is not in `ALL_MODELS` or
any `*_APPLY_MODELS` rule, so sweeps and `conclude` leave it out. It can be
generated explicitly with `make generate`.

```bash
make generate DATA_MODEL=O7 SEED=230
```

## Evaluation

`eval-<algo>` loads `sampled.npz` and draws, at the same test predictors, as
many fresh responses from the true conditional distribution as there are
method draws. It then computes the Energy Score of the method's draws `Yhat`
against the true draws `Y`, averaged over test predictors,

```text
ES = E|Yhat - Y| - 1/2 E|Yhat - Yhat'|
```

where `|.|` is the Euclidean norm, the first term averages over all method and
true pairs, and the second averages over distinct pairs of method draws (a
U-statistic). Lower is better.

The oracle baseline `ES_oracle` is the Energy Score of the true distribution
itself, `1/2 E|Y - Y'|`. `generate` computes it once per dataset from `ORACLE`
fresh responses at each test predictor and stores it in `test.npz`, so every
method on that dataset is compared with the same baseline. The headline metric
is the relative score

```text
rEEScore = (ES - ES_oracle) / ES_oracle
```

which is near zero for a method that matches the true conditional
distribution. Evaluation stops with an error if `test.npz` has no
`ES_oracle`, in which case the dataset needs to be generated again.

Both scores are computed in chunks of 32 test predictors at a time, on CUDA
when it is available, so peak memory depends on the chunk size and not on the
number of test points.

`eval.npz` also stores the mean and median squared errors of the conditional
mean and standard deviation (`mu_mse`, `mu_medse`, `sigma_mse`,
`sigma_medse`), and the mean Wasserstein-1 distance `w_mean` (sliced over 100
projections when the response is multivariate), together with `escore_mean`,
`ES_oracle` and `reescore`.

The workbook written by `conclude` has two sheets. The `summary` sheet has one
row per model and method, and gives each metric as `mean (sd)` over the seeds
found. The `details` sheet gives the same means and standard deviations as
separate numeric columns, with a status for each pair (`ok`, `missing`,
`not applicable` or `read error`) and the number of seeds read.

## HY details

### Bandwidth tuning

HY uses two bandwidths, `h` for estimating `theta` and `H` for the conditional
distribution function. By default `train-hy` selects them separately for each
model and seed from a grid of 21 pairs. The grid has six values
`h = 0.1, 0.3, ..., 1.1`, and for each `h` the values `H = h + 0.1, h + 0.3,
..., 1.2`. For each pair, tuning scores the validation data with the Energy
Score (the CRPS in one dimension) and keeps the pair with the lowest score.

Tuning uses `HY_N_SUB_VAL` validation points (default `200`), drawn at random
with the dataset seed, and a distribution-function grid of
`HY_NUM_Y_GRID_TUNING` points (default `50`). The Energy Score is computed
exactly from the estimated inverse distribution function, without Monte Carlo
draws. Since `theta` depends only on `h`, the six `theta` fits are shared
across the 21 pairs, and the projections and response ordering are computed
once for each `h`.

To skip the search, give both bandwidths on the command line.

```bash
make train-hy DATA_MODEL=U7 SEED=230 HY_H=0.7 HY_H_CDF=0.8
```

### Theta fitting

The criterion compares local-linear estimates of the conditional distribution
function with empirical distribution functions over a fixed set of balls in
predictor space. The ball centers form the grid {-1.5, 0, 1.5} in each
predictor coordinate, with radius `HY_RADIUS` (default `1.0`). The criterion is
the full leave-two-out Hall--Yao objective, evaluated with an Epanechnikov
kernel, and the ball-membership matrix is stored as a sparse CSR tensor.

All `h` values are fitted together in one batched float32 PyTorch run with
Adam, on CUDA when available. Each `h` has its own parameter vector `w`, with
`theta = w / sqrt(sum(w^2) + 1e-5)`, and all start from the same random unit
vector drawn from the seed. Each `h` keeps its own best value and its own
early-stopping counter, which any strictly lower finite objective resets, and
an `h` whose counter reaches the patience leaves the batch. The settings are
`HY_ADAM_LR` (default `0.01`), `HY_ADAM_MAX_EPOCHS` (default `300`) and
`HY_ADAM_PATIENCE` (default `30`). Where a local-linear denominator falls below `1e-8` in
absolute value, the objective uses the empirical distribution function
instead. Each fitted `theta` is signed so that
its first nonzero entry is positive.

### Sampling

`sample-hy` reads `theta` and `H` from `theta.npz`. For each test predictor it
estimates the conditional distribution function on a grid of `HY_NUM_Y_GRID`
points (default `200`) spanning the training responses, clips it to [0, 1]
and makes it nondecreasing with a cumulative maximum. It then draws `SAMPLES`
responses by linear interpolation of the inverse at uniform draws, which are
clipped to the smallest and largest estimated probability levels. When the local-linear
weights cannot be normalized, the draws equal the mean training response.
Tuning and final sampling share this sampler, which runs on the CPU with
NumPy. To score a candidate during tuning, it computes the Energy Score of
each estimated distribution exactly, without drawing samples, in a function
compiled with Numba.

### Saved files

`train-hy` writes to `workspaces/hy/<model>/seed_<seed>/`.

| File | Contents |
|---|---|
| `theta.npz` | `theta`, the selected `h` and `H`, `t_opt` (wall-clock seconds of the batched theta fit), `t_total` (seconds of the whole fit and tuning) and `success` |
| `config.yaml` | The training arguments, `selected_h`, `selected_H`, `mode` (`tuned` or `manual`), `theta_fit_diagnostics` (the `h`, `epochs` and `best_S_theta` of each fit), `n_val_used`, `theta_fit_seconds` and `total_fit_seconds` |
| `tuning.csv` | One row per pair with columns `h,H,EnergyScore`, written only when the bandwidths are tuned |

The code is in `condgen_benchmark/methods/hy/`. `loss.py` holds the objective
`HallYaoLoss`, `fit.py` holds the batched Adam fit `fit_thetas`, `tuning.py`
holds the bandwidth search, and `sampler.py` holds the shared conditional
sampler.

## FlexCode sampling grid

`train-flex` evaluates the density on a response grid of `FLEX_NUM_Y_GRID`
points (default `200`, passed as `--ny`) and stores that grid in both
`model.pth` and `model_tuned.pth`. `sample-flex` reads the grid from the
selected checkpoint, predicts on it, and checks that the predicted density
and grid match the saved grid. The sampling command has no grid option, so a
sampling run cannot change a choice made at training time, and `SAMPLES`
only sets the number of draws. A checkpoint without a saved grid is rejected
with an error that asks for retraining.

```bash
make train-flex DATA_MODEL=U1 SEED=230 FLEX_NUM_Y_GRID=200
make sample-flex DATA_MODEL=U1 SEED=230 SAMPLES=300
```

## License

GPL-2.0 - see [LICENSE](LICENSE)
