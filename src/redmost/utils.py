#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:38:29 2023.

@author: Maurizio D'Addona
"""
import os
from typing import Optional, Tuple, Union
from urllib import request
import json
from multiprocessing.pool import Pool

import numpy as np

from scipy.signal.windows import general_gaussian   # type: ignore

# See https://stackoverflow.com/questions/28774852/pypi-api-how-to-get-stable-package-version
try:
    from packaging import version
except ImportError:
    from pip._vendor.packaging import version

import redmost

DEVICE_DICT = {}

try:
    import tensorflow as tf
except Exception:
    HAS_TENSORFLOW = False
    TF_DEVICES = {}
    DEVICE_DICT['tf'] = TF_DEVICES
else:
    HAS_TENSORFLOW = True

    TF_DEVICES = {
        device.name: device
        for device in tf.config.list_physical_devices()
    }
    DEVICE_DICT['tf'] = TF_DEVICES


try:
    import torch
except Exception:
    HAS_TORCH = False
    TORCH_DEVICES = {}
    DEVICE_DICT['torch'] = TORCH_DEVICES
else:
    HAS_TORCH = True
    TORCH_DEVICES = {}
    if torch.cuda.is_available():
        for j in range(torch.cuda.device_count()):
            dev_id = f"{j}:{torch.cuda.get_device_name(j)}"
            TORCH_DEVICES[dev_id] = f'cuda:{j}'
    DEVICE_DICT['torch'] = TORCH_DEVICES

    def smooth_fft_torch(
            data: np.ndarray,
            sigma: float = 25.0,
            axis: int = -1,
            mask: Optional[np.ndarray] = None,
            device: Optional[str] = None
    ) -> np.ndarray:
        """
        Return a smoothed version of an array.

        :param data:
            The input array to be smoothed.
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

        if device is None:
            device = torch.get_default_device()

        data = np.copy(data).astype(np.float32)
        if mask is None:
            actual_mask: np.ndarray = np.zeros_like(data, dtype=bool)
        else:
            actual_mask = mask

        actual_mask |= ~np.isfinite(data)

        data[actual_mask] = 0

        data = torch.from_numpy(data).to(device=device)

        xx = torch.hstack((data, torch.flip(data, dims=[axis, ])))

        gauss = torch.signal.windows.gaussian(xx.shape[axis], std=sigma,
                                              device=device)

        win = torch.roll(gauss, xx.shape[axis] // 2)

        fxx = torch.fft.fft(xx, dim=axis)
        xxf = torch.real(torch.fft.ifft(fxx * win)).cpu().numpy()
        xxf = xxf[..., :data.shape[axis]]
        xxf[actual_mask] = np.nan
        return xxf

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def find_exp(number: float) -> int:
    base10 = np.log10(abs(number))
    return int(np.floor(base10))

def check_updates() -> Tuple[bool, str]:
    """Check for new a version on pypi.python.org."""

    try:
        cur_ver_str = '.'.join([str(x) for x in redmost.version_tuple[:3]])
        current_version = version.parse(cur_ver_str)
    except Exception:
        return False, 'none'

    req = request.Request(redmost.PYPI_REPO_API_URL, method='GET')
    try:
        with request.urlopen(req) as response:
            response_data = response.read()
    except Exception as exc:
        return False, str(exc)

    pypi_version = version.parse('0')
    j = json.loads(response_data)
    releases = j.get('releases', [])
    for release in releases:
        ver = version.parse(release)
        if not ver.is_prerelease:
            pypi_version = max(pypi_version, ver)

    is_outdate = current_version < pypi_version
    return is_outdate, str(pypi_version)


def get_data_file(
    file_name: str,
) -> Union[str, None]:
    """
    Get the path of a data file.

    :param file_name: the name of the file.
    :return: the path of the file.
    """
    data_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 'ui', f"{file_name}"
    )
    if os.path.isfile(data_file):
        return  data_file


def get_icon(
    icon_name: str,
    theme: Optional[str] = 'feather'
) -> Union[str, None]:
    """
    Load icons from an existing theme.

    :param icon_name: the name of the icon.
    :param theme: the name of the theme. Default value is 'feather'
    :return: the path of the icon file.
    """
    for ext in ["png", "svg"]:
        icon_file = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'ui', 'icons',
            theme, f"{icon_name}.{ext}"
        )
        if os.path.isfile(icon_file):
            return icon_file

def smooth_fft_np(
    data: np.ndarray,
    sigma: float = 25.0,
    axis: int = -1,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Return a smoothed version of an array.

    :param data:
        The input array to be smoothed.
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
    data[actual_mask] = 0

    xx = np.hstack((data, np.flip(data, axis=[axis, ])))
    win = np.roll(
        general_gaussian(xx.shape[axis], 1, sigma),
        xx.shape[axis]//2
    )
    fxx = np.fft.fft(xx, axis=axis)
    xxf = np.real(np.fft.ifft(fxx*win))[..., :data.shape[axis]]
    xxf[actual_mask] = np.nan
    return xxf


def smooth_fft(
    data: np.ndarray,
    sigma: float = 25.0,
    axis: int = -1,
    mask: Optional[np.ndarray] = None,
    use_gpu: Union[str, bool] = False,
    device: Optional[str] = None
) -> np.ndarray:

    if sigma <= 0:
        return data

    if use_gpu == 'torch' and HAS_TORCH:
        return smooth_fft_torch(
            data, sigma, axis=axis, mask=mask, device=device
        )
    else:
        return smooth_fft_np(
            data, sigma, axis=axis, mask=mask
        )


def continuum_fit_chebyshev_np(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    order: int = 5
) -> np.ndarray:
    """
    Fit the spectrum continuum with a Chebyshev polynomial.

    :param data: numpy.ndarray
        It can be a 1D, 2D or 3D array. The first dimension is treated as the
        spectral axis (i.e. wavelengths)
    """
    if len(data.shape) == 1:
        pix_coords = np.arange(data.shape[0])
        continuum = np.zeros_like(data)
        if mask is not None:
            spec_f = data[~mask]
            spec_x = pix_coords[~mask]
        else:
            spec_f = data
            spec_x = pix_coords

        try:
            mu_spectra = np.polynomial.Chebyshev.fit(
                spec_x,
                spec_f,
                order
            )
        except ValueError:
            continuum[...] = np.nan
            return continuum

        if mask is not None:
            continuum[~mask] = mu_spectra(spec_x)
            continuum[mask] = np.nan
        else:
            continuum = mu_spectra(spec_x)
    elif len(data.shape) == 2:
        continuum = np.zeros_like(data, dtype=np.float32)
        for j in range(data.shape[1]):
            continuum[:, j] = continuum_fit_chebyshev_np(
                data[:, j],
                mask=mask[:, j],
                order=order
            )
    else:
        flattened_data = data.reshape(data.shape[0], -1)
        flattened_mask = mask.reshape(mask.shape[0], -1)
        flattened_continuum = continuum_fit_chebyshev_np(
            flattened_data,
            mask=flattened_mask,
            order=order
        )
        continuum = flattened_continuum.reshape(flattened_data)

    return continuum


def continuum_fit_chebyshev(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    order: int = 5,
    use_gpu: Union[str, bool] = False,
    device: Optional[str] = None
) -> np.ndarray:
    if use_gpu == 'torch' and HAS_TORCH:
        # TODO: implement chebyshev fitting with torch
        raise NotImplementedError()
    else:
        return continuum_fit_chebyshev_np(data, mask, order)

def separate_continuum(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    fit_mode: str = 'chebyshev',
    fft_sigma: float = 10.0,
    chebyshev_order: int = 5,
    use_gpu: Union[str, bool] = False,
    device: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a numpy array in a smoothed continuum and a residual.

    :param data:
        The input array to be smoothed.
    :param sigma:
        Parameter to be passed to the function general_gaussian().
        The default value is 25.0.
    :param mask:
        An optional array containing a boolean mask of values that should be
        masked during the smoothing process, were a True means that the
        corresponding value in the input array is masked.
    :param fit_mode:
        Specify which function to use to fit the continuum. It can be either
        'chebyshev' or 'fft'.
    :param use_gpu:
        Whether to use the GPU version of the continuum fitting function.
    :device:
        The GPU device to use
    :return continuum: The continuum.
    :return residuals: The continuum subtracted spectrum.
    """
    if fit_mode not in ['chebyshev', 'fft']:
        raise ValueError(
            "'mode' can be either 'chebyshev' or 'fft'"
        )
    elif fit_mode == 'chebyshev':
        continuum = continuum_fit_chebyshev(
            data,
            order=chebyshev_order,
            mask=mask,
            use_gpu=use_gpu,
            device=device
        )
    else:
        continuum = smooth_fft(
            data,
            fft_sigma,
            mask=mask,
            use_gpu=use_gpu,
            device=device
        )
    residuals = data - continuum
    return continuum, residuals
