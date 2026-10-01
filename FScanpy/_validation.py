"""Shared validation for prediction and plotting parameters."""
import math
from numbers import Integral, Real


def probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite number between 0 and 1')
    return float(value)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)
