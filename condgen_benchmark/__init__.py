"""
Conditional Generation Benchmark Framework

A unified framework for evaluating conditional density estimation algorithms.
"""

__version__ = "1.1.1"
__author__ = "Chiatse Wang"
__email__ = "chiatsewang@stat.sinica.edu.tw"

from condgen_benchmark import data, methods

__all__ = [
    "data",
    "methods",
]
