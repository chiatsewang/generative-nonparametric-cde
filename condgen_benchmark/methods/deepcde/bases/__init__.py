# Basis functions for DeepCDE density estimation
from condgen_benchmark.methods.deepcde.bases.cosine import CosineBasis
from condgen_benchmark.methods.deepcde.bases.tensor import TensorBasis
from condgen_benchmark.methods.deepcde.bases.wavelet import WaveletBasis

__all__ = [
    "CosineBasis",
    "WaveletBasis",
    "TensorBasis",
]
