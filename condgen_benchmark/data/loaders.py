import os
from typing import Optional, Tuple, Union

import numpy as np
import torch


def load_synthetic_dataset(
    data_dir: str,
    data_name: str,
    seed: int,
    split_type: str,
    num_X: int,
    num_Y: Optional[int] = None,
    to_torch: bool = False,
) -> Tuple[
    Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]
]:
    """
    Load synthetic dataset from .npz file with X, Y as P(Y|X) data.

    Args:
        data_dir: Root directory containing the data
        data_name: Name of the dataset
        seed: Random seed for dataset generation
        split_type: Type of data split (e.g., 'train', 'validation', 'test')
        num_X: Number of X (predictor) samples to load
        num_Y: Number of Y (response) samples to load. If None, Y is
            returned as None
        to_torch: Whether to convert arrays to PyTorch tensors

    Returns:
        Tuple of (X, Y) where:
        - X: Predictor data as numpy array or torch tensor
        - Y: Response data as numpy array or torch tensor, or None if
            num_Y is None

    Raises:
        ValueError: If requested number of samples exceeds available data
        FileNotFoundError: If the data file doesn't exist
    """
    data_path = os.path.join(
        data_dir, "synthetic", data_name, f"seed_{seed}", f"{split_type}.npz"
    )

    data = np.load(data_path)
    X, Y_data = data["X"], data["Y"]

    if X.shape[0] < num_X:
        raise ValueError(
            f"Predictor data (X) insufficient: need {num_X}, got {X.shape[0]}"
        )
    X = (
        X[:num_X]
        if not to_torch
        else torch.tensor(X[:num_X], dtype=torch.float32)
    )

    Y = None
    if num_Y:
        if Y_data.shape[0] < num_Y:
            raise ValueError(
                f"Response data (Y) insufficient: need {num_Y}, "
                f"got {Y_data.shape[0]}"
            )

        Y = (
            Y_data[:num_Y]
            if not to_torch
            else torch.tensor(Y_data[:num_Y], dtype=torch.float32)
        )

    return X, Y  # predictor, response
