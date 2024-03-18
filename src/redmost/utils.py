#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:38:29 2023.

@author: Maurizio D'Addona
"""
import os
from typing import Optional, Tuple

import numpy as np

from scipy.signal.windows import general_gaussian   # type: ignore


def get_data_file(filename: str) -> str:
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'ui',
        filename
    )


def smooth_fft(
    data: np.ndarray,
    m: float = 1.0,
    sigma: float = 25.0,
    axis: int = -1,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return a smoothed version of an array.

    :param data:
        The input array to be smoothed.
    :param m:
        parameter to be passed to the function general_gaussian().
        The default value is 1.0.
    :param sigma:
        Parameter to be passed to the function general_gaussian().
        The default value is 25.0.
    :param axis:
        The axis along with perform the smoothing. The default value is -1.
    :param mask:
        An optional array containing a boolean mask of values that should be
        masked during the smoothing process, were a True means that the
        corresponding value in the input array is masked.
    :return: The smoothed array.
    """
    data = np.copy(data)
    if mask is None:
        actual_mask: np.ndarray = np.zeros_like(data, dtype=bool)
    else:
        actual_mask = mask

    actual_mask |= ~np.isfinite(data)

    if len(data.shape) > 1:
        for j in range(data.shape[0]):
            data[j, actual_mask[j]] = np.interp(
                np.flatnonzero(actual_mask[j]),
                np.flatnonzero(~actual_mask[j]),
                data[j, ~actual_mask[j]]
            )
    else:
        data[actual_mask] = np.interp(
            np.flatnonzero(actual_mask),
            np.flatnonzero(~actual_mask),
            data[~actual_mask]
        )

    xx = np.hstack((data, np.flip(data, axis=axis)))
    win = np.roll(
        general_gaussian(xx.shape[axis], m, sigma),
        xx.shape[axis]//2
    )
    fxx = np.fft.fft(xx, axis=axis)
    xxf = np.real(np.fft.ifft(fxx*win))[..., :data.shape[axis]]
    xxf[actual_mask] = np.nan
    return xxf


def separate_continuum(
    data: np.ndarray,
    m: float = 1.0,
    sigma: float = 10.0,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a numpy array in a smoothed continuum and a residual.

    :param data:
        The input array to be smoothed.
    :param m:
        parameter to be passed to the function general_gaussian().
        The default value is 1.0.
    :param sigma:
        Parameter to be passed to the function general_gaussian().
        The default value is 25.0.
    :param mask:
        An optional array containing a boolean mask of values that should be
        masked during the smoothing process, were a True means that the
        corresponding value in the input array is masked.

    :return continuum: The continuum.
    :return residuals: The continuum subtracted spectrum.
    """
    continuum = smooth_fft(data, m, sigma, mask=mask)
    residuals = data - continuum
    return continuum, residuals
