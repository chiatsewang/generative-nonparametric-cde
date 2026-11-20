"""Shared train/validation/test data loading for the HY scripts."""

import os

import numpy as np


def load_split(data_dir, data_name, seed, split):
    path = os.path.join(data_dir, data_name, f"seed_{seed}", f"{split}.npz")
    with np.load(path) as data:
        X = np.asarray(data["X"], dtype=np.float64)
        Y = np.asarray(data["Y"], dtype=np.float64)
    if X.ndim != 2 or len(X) != len(Y) or Y.reshape(len(X), -1).shape[1] != 1:
        raise ValueError("X must have shape (n, p), and Y must be scalar")
    return X, Y.reshape(-1)
