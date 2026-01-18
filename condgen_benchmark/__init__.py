# condgen_benchmark/__init__.py
"""
Conditional Generation Benchmark Framework

A unified framework for evaluating conditional density estimation algorithms.
"""

__version__ = "1.0.0"
__author__ = "Chiatse Wang"
__email__ = "chiatsewang@stat.sinica.edu.tw"

from condgen_benchmark import data, methods, models

__all__ = [
    "data",
    "models",
    "methods",
]
