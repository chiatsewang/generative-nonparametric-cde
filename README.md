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

- **HyAlg** - Kernel-based method (Hu & Yao algorithm)
- **FlexCode** - Flexible conditional density estimation
- **DeepCDE** - Deep Conditional Density Estimation
- **GCDS** - Generative Conditional Density Sampling
- **DDPM** - Denoising Diffusion Probabilistic Models

## Project Structure

```text
.
├── condgen_benchmark/    # Main package
│   ├── data/             # Data loading and generation
│   ├── methods/          # CDE methods (ddpm, deepcde, gcds, hyalg)
│   └── models/           # Neural network architectures
├── scripts/              # Training and evaluation scripts
├── datasets/             # Generated datasets
└── workspaces/           # Model checkpoints and results
```

## License

GPL-2.0 - see [LICENSE](LICENSE)
