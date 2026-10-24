# Copyright (c) 2023-2026 Maurizio D'Addona <mauritiusdadd@gmail.com>
#
# This file is part of Redmost.
#
# Redmost is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Redmost is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Redmost.  If not, see <https://www.gnu.org/licenses/>.
"""Spectrum loaders and format identifiers."""

import os
import copy
from pathlib import Path
from typing import Any


import numpy as np

from astropy.nddata import VarianceUncertainty  # type: ignore
from astropy.nddata import StdDevUncertainty  # type: ignore
from astropy.nddata import InverseVariance  # type: ignore
from astropy.io import fits, registry
from astropy.table import Table  # type: ignore
from astropy import wcs  # type: ignore
from astropy import units  # type: ignore
from specutils.io.registers import identify_spectrum_format  # type: ignore
from specutils.io.registers import data_loader  # type: ignore
from specutils.manipulation import LinearInterpolatedResampler
try:
    from specutils import Spectrum
except ImportError:
    # The Spectrum1D class is deprecated and may be removed in a future version
    from specutils import Spectrum1D as Spectrum

from redmost import utils


KNOWN_SPEC_EXT_NAMES = [
    'spec', 'spectrum', 'flux', 'fluxes', 'data', 'sci', 'science',
    'primary'
]
KNOWN_VARIANCE_EXT_NAMES = [
    'stat', 'stats', 'var', 'variance', 'spec_var',
    'spectrum_var', 'spec_variance', 'spectrum_variance',
    'variance_spec', 'variance_spectrum'
]
KNOWN_RMS_EXT_NAMES = [
    'noise', 'err', 'rms', 'error_spectrum', 'spectrum_error',
    'rms_spectrum', 'spectrum_rms', 'noise_spectrum', 'spectrum_noise',
]
KNOWN_INVAR_EXT_NAMES = [
    'ivar', 'ivariance', 'spec_ivar', 'spectrum_ivariance', 'spec_ivariance',
    'inverse_variance'
]
KNOWN_MASK_EXT_NAMES = ['mask', 'platemask', 'footprint', 'dq', 'nan_mask']
KNOWN_WAVE_EXT_NAMES = [
    "lam", "lambda", "wave", "wavelength", "waves", "wavelengths"
]
KNOWN_RCURVE_EXT_NAMES = ['r', 'reso', 'resolution', 'rcurve', 'wd']
KNOWN_RGB_EXT_NAMES = ['r', 'g', 'b', 'red', 'green', 'blue']


def getHDU(
        hdul: fits.HDUList,
        valid_names: list[str],
        index: str | int | None = None
):
    if index is None:
        for hdu in hdul:
            if hdu.name.lower() in valid_names and hdu.data is not None:
                return hdu
        return None
    else:
        return hdul[index]


def isFits(filename: str | os.PathLike[str]) -> bool:
    """
    Check if the file is a valid FITS file.

    Parameters
    ----------
    filename: str or os.PathLike
        The path of the file to check.

    Returns
    -------
    bool:
        True if the file is a valid FITS file, False otherwise.
    """
    try:
        _ = fits.getheader(filename)
    except (OSError, FileNotFoundError):
        return False
    return True


def identifySpecexFits(origin, *args, **kwargs) -> bool:
    """Identify spectra extracted with python-specex."""
    if not isFits(args[0]):
        return False

    with fits.open(args[0]) as hdul:
        flux_hdu = getHDU(hdul, KNOWN_SPEC_EXT_NAMES)
        if flux_hdu is None:
            return False

        if flux_hdu.data is None:
            return False


        try:
            specexFitsLoader(args[0])
        except Exception:
            return False
        return True


def identifyClashVlt1dFits(origin, *args, **kwargs) -> bool:
    """Identify spectra extracted with python-specex."""
    if not isFits(args[0]):
        return False

    with fits.open(args[0]) as hdul:
        if hdul[0].header.get('NAXIS') != 1:
            return False

        instrument = hdul[0].header.get('INSTRUME')
        ctype1 = hdul[0].header.get('CTYPE1')
        tunit = hdul[0].header.get('TUNIT')

        if len(hdul[0].data.shape) != 1:
            return False

        if (instrument is None) or (instrument.strip() != 'VIMOS'):
            return False

        if (tunit is None) or (ctype1 is None):
            return False

        if ctype1.strip() != 'WAVE':
            return False

    return True


def isAscii(filename: str | os.PathLike[str], sample_size: int = 4096) -> bool:
    """
    Check if the file is a valid ASCII text file.

    Parameters
    ----------
    filename : str or os.PathLike
        The path of the file to check.
    sample_size : int, optional
        Number of bytes to sample from the beginning of the file.

    Returns
    -------
    bool
        True if the sampled bytes can be decoded as ASCII and don't contain null bytes.
    """
    try:
        with open(filename, 'rb') as f:
            chunk = f.read(sample_size)
        if not chunk:
            return False
        if b'\x00' in chunk:
            return False
        chunk.decode('ascii')
    except (OSError, UnicodeDecodeError):
        return False
    return True


def identify1DAsciiTable(origin, *args, **kwargs) -> bool:
    if not args or not Path(str(args[0])).is_file():
        return False
    if isFits(args[0]):

        return False
    if not isAscii(args[0]):
        return False

    with open(args[0]) as f:
        for l in f:
            if l.strip().startswith('#'):
                return True

    return False


@data_loader(
    label="ascii-table",
    identifier=identify1DAsciiTable,
    extensions=['txt', 'ascii', 'dat']
)
def asciiTableLoader(
   file_name: str,
   resolution: int | float = 1,
    **kwargs
) -> Spectrum:
    t = Table.read(file_name, format='ascii')
    try:
        t_sky = Table.read(f"{file_name}_sky", format='ascii')
    except OSError:
        t_sky = None
    
    flux = None
    var = None
    lam = None
    mask = None
    reso = None

    flux_units = units.Unit("Angstrom-1 cm-2 erg s-1")
    for col in t.colnames:
        if col.lower() in KNOWN_SPEC_EXT_NAMES:
            flux = t[col] * flux_units
            if t_sky is not None:
                var = t_sky[col]**2
        elif col.lower() in KNOWN_VARIANCE_EXT_NAMES:
            var = t[col]
        elif col.lower() in KNOWN_RMS_EXT_NAMES:
            var = t[col]**2
        elif col.lower() in KNOWN_INVAR_EXT_NAMES:
            var = 1/t[col]
        elif col.lower() in KNOWN_MASK_EXT_NAMES:
            mask = t[col]
        elif col.lower() in KNOWN_WAVE_EXT_NAMES:
            lam = t[col] * units.AA
        elif col.lower() in KNOWN_RCURVE_EXT_NAMES:
            reso = t[col]

    if flux is None:
        raise ValueError(f"Flux column not found in {file_name}")

    if lam is None:
        raise ValueError(f"Wavelength column not found in {file_name}")

    if mask is None:
        mask = np.full_like(flux.value, False, dtype=bool)

    mask |= ~np.isfinite(flux.value)

    if var is not None:
        unc = VarianceUncertainty(var * (flux_units**2))
    else:
        unc = None
    
    meta = {}
    sp = Spectrum(
        flux=flux,
        spectral_axis=lam,
        uncertainty=unc,
        meta=meta
    )

    if reso is not None:
        rc = reso.copy()
    else:
        # If now wavelenght dispersion information is present, then
        # compute it using the wavelenght
        delta_lambda = np.ones_like(lam)
        delta_lambda[1:] = (lam[1:] - lam[:-1])
        delta_lambda = delta_lambda.to('Angstrom').value
        rc = resolution / delta_lambda
        rc[0] = rc[1]
    rc[rc < 1e-3] = 2.

    sp.mask = mask
    sp.rc = rc
    return sp


def identifyFORS21dFits(origin, *args, **kwargs) -> bool:
    if not isFits(args[0]):
        return False

    with fits.open(args[0]) as hdul:
        if hdul[0].header.get('NAXIS') != 2:
            return False

        if hdul[0].header.get('NAXIS2') != 1:
            return False

        instrument = hdul[0].header.get('INSTRUME')
        ctype1 = hdul[0].header.get('CTYPE1')
        ctype2 = hdul[0].header.get('CTYPE2')

        if (instrument is None) or (instrument.strip() != 'FORS2'):
            return False

        if ctype1.strip() != 'WAVE':
            return False

        if ctype2.strip() != 'OBJECT':
            return False

    return True


@data_loader(
    label="FORS2-1d",
    identifier=identifyFORS21dFits,
    extensions=['fits']
)
def fors2FitsLoader(
   file_name: str,
    flux_hdu_index: str | int | None = None,
    resolution: int | float = 1,
    **kwargs
) -> Spectrum:
    data, header = fits.getdata(file_name, header=True)
    flux_units = units.Unit("Angstrom-1 cm-2 erg s-1")

    flux = data[0] * flux_units
    flux_wcs = wcs.WCS(header)

    pixel = np.arange(len(flux))
    lam = flux_wcs.spectral.pixel_to_world(pixel).to(units.Angstrom)
    var = np.ones_like(data[0]) * (flux_units**2)


    meta = {'header': header}
    sp = Spectrum(
        flux=flux,
        spectral_axis=lam,
        wcs=flux_wcs,
        uncertainty=VarianceUncertainty(var),
        meta=meta
    )
    sp.mask = ~np.isfinite(data[0])
    return sp

@data_loader(
    label="specex-1d",
    identifier=identifySpecexFits,
    extensions=['fits']
)
def specexFitsLoader(
    file_name: str,
    flux_hdu_index: str | int | None = None,
    resolution: int | float = 1,
    **kwargs
) -> Spectrum:
    """
    Load a python-specex spectrum.

    :param file_name:
        The path of the FITS file to read.
    :param flux_hdu_index:
        The extension name or index from which to read the spectrum
    :param resolution:
        The spectral resolution parameter
    :return: A Spectrum object.
    """
    with fits.open(file_name, **kwargs) as hdulist:
        flux_hdu = getHDU(hdulist, KNOWN_SPEC_EXT_NAMES, index=flux_hdu_index)
        var_hdu = getHDU(hdulist, KNOWN_VARIANCE_EXT_NAMES)
        ivar_hdu = getHDU(hdulist, KNOWN_INVAR_EXT_NAMES)
        rms_hdu = getHDU(hdulist, KNOWN_RMS_EXT_NAMES)
        mask_hdu = getHDU(hdulist, KNOWN_MASK_EXT_NAMES)
        rc_hdu = getHDU(hdulist, KNOWN_RCURVE_EXT_NAMES)

        valid_id_keys: list[str] = [
            f"{i}{j}"
            for i in ['', 'OBJ', 'OBJ_', 'TARGET', 'TARGET_']
            for j in ['ID', 'NUMBER', 'UID', 'UUID']
        ]

        obj_id = None
        for key in valid_id_keys:
            try:
                obj_id = hdulist[0].header[key]
            except KeyError:
                continue
            else:
                break

        if flux_hdu is None:
            raise ValueError("Cannot find flux data")
        flux = flux_hdu.data

        if var_hdu is not None:
            var = var_hdu.data
        elif rms_hdu is not None:
            var = rms_hdu.data ** 2
        elif ivar_hdu is not None:
            var = 1.0 / ivar_hdu.data
        else:
            var = np.ones_like(flux)


        if mask_hdu is None:
            mask = None
        else:
            mask = mask_hdu.data

        if rc_hdu is None:
            rc = None
        else:
            rc = rc_hdu.data

        try:
            flux_wcs = wcs.WCS(flux_hdu.header)
        except wcs.wcs.InconsistentAxisTypesError as exc:
            crpix1 = flux_hdu.header.get('CRPIX1')
            crval1 = flux_hdu.header.get('CRVAL1') 
            cdelt1 = flux_hdu.header.get('CDELT1') 

            if any([crpix1 is None, crval1 is None, cdelt1 is None]):
                raise exc

            # Fixing broken WCS
            flux_hdu.header['CTYPE1'] = 'WAVE'
            flux_hdu.header['CUNIT1'] = 'Angstrom'
            
            flux_wcs = wcs.WCS(flux_hdu.header)


        if flux.shape != var.shape:
            raise ValueError(
                "Spectral data invalid or corruptede: Flux data shape "
                "do not match variance data one!"
            )

        # NOTE: Wavelenghts must be in Angstrom units
        pixel = np.arange(len(flux))
        if flux_wcs.has_spectral:
            lam = flux_wcs.pixel_to_world(pixel).Angstrom
        else:
            try:
                coeff0 = flux_hdu.header["COEFF0"]
                coeff1 = flux_hdu.header["COEFF1"]
            except KeyError:
                raise ValueError("Catton determine wavelength mapping")
            lam = 10**(coeff0 + coeff1*pixel)
        flux = flux.astype('float32')

        flux_not_nan_mask = ~np.isnan(flux)

        if mask is not None:
            flux_not_nan_mask &= ~(mask.astype(bool))

        try:
            flux_units = units.Unit(flux_hdu.header['BUNIT'])
        except KeyError:
            flux_units = units.Unit(flux_hdu.header['TUNIT'])

            if flux_units.physical_type == 'dimensionless':
                flux_units = flux_units * units.Unit(
                    "Angstrom-1 cm-2 erg s-1"
                )

        wave_units = units.Unit(flux_hdu.header['CUNIT1'])

        flux = flux.copy() * flux_units
        var = var.copy() * (flux_units**2)
        lam = lam.copy() * wave_units

        if rc is not None:
            rc = rc.copy()
        else:
            # If now wavelenght dispersion information is present, then
            # compute it using the wavelenght
            delta_lambda = np.ones_like(lam)
            delta_lambda[1:] = (lam[1:] - lam[:-1])
            delta_lambda = delta_lambda.to('Angstrom').value
            rc = resolution / delta_lambda
            rc[0] = rc[1]
        rc[rc < 1e-3] = 2.

        meta = {'header': hdulist[0].header}
        uncertainty = VarianceUncertainty(var)

        sp: Spectrum = Spectrum(
            flux=flux,
            spectral_axis=lam,
            wcs=flux_wcs,
            uncertainty=uncertainty,
            meta=meta
        )

        sp.mask = flux_not_nan_mask
        sp.rc = rc
        sp.obj_id = obj_id

    return sp


@data_loader(
    label="clash-vlt-1d",
    identifier=identifyClashVlt1dFits,
    extensions=['fits']
)
def clashVlt1DFitsLoader(
        file_name: str,
        flux_hdu_index: str | int | None = None,
        resolution: int | float = 1,
        **kwargs
) -> Spectrum:
    """
    Load a python-specex spectrum.

    :param file_name:
        The path of the FITS file to read.
    :param flux_hdu_index:
        The extension name or index from which to read the spectrum
    :param resolution:
        The spectral resolution parameter
    :return: A Spectrum object.
    """

    with fits.open(file_name, **kwargs) as hdulist:
        flux_hdu = hdulist[0]
        noise_hdu = getHDU(hdulist, ['noise',])

        grism = flux_hdu.header.get('HIERARCH ESO INS ADF GRISM NAME')
        if grism == 'LR_Blue':
            # cut wavelengths <3600A and >6840A
            w_min = units.Quantity(3600, 'angstrom')
            w_max = units.Quantity(6800, 'angstrom')
        else:
            w_min = units.Quantity(4750, 'angstrom')
            w_max = units.Quantity(9800, 'angstrom')
            # cut wavelengths <4700A and >9900A

        if flux_hdu is None:
            raise ValueError("Cannot find flux data")

        flux = flux_hdu.data

        if noise_hdu is None:
            var = np.ones_like(flux)
        else:
            var = noise_hdu.data**2

        flux_wcs = wcs.WCS(flux_hdu.header)

        if flux.shape != var.shape:
            raise ValueError(
                "Spectral data invalid or corruptede: Flux data shape "
                "do not match variance data one!"
            )

        # NOTE: Wavelengths must be in Angstrom units
        pixel = np.arange(len(flux))
        lam = flux_wcs.pixel_to_world(pixel).Angstrom
        flux = flux.astype('float32')

        flux_not_nan_mask = ~np.isnan(flux)

        try:
            flux_units = units.Unit(flux_hdu.header['BUNIT'])
        except KeyError:
            flux_units = units.Unit(flux_hdu.header['TUNIT'])

            if flux_units.physical_type == 'dimensionless':
                flux_units = flux_units * units.Unit(
                    "Angstrom-1 cm-2 erg s-1"
                )

        wave_units = units.Unit(flux_hdu.header['CUNIT1'])

        flux = flux.copy() * flux_units
        var = var.copy() * (flux_units ** 2)
        lam = lam.copy() * wave_units

        # If now wavelength dispersion information is present, then
        # compute it using the wavelength
        delta_lambda = np.ones_like(lam)
        delta_lambda[1:] = (lam[1:] - lam[:-1])
        delta_lambda = delta_lambda.to('Angstrom').value
        rc = resolution / delta_lambda
        rc[0] = rc[1]
        rc[rc < 1e-3] = 2.

        lam_mask = np.ones(len(flux), dtype=bool)

        if w_min is not None:
            lam_mask &= lam >= w_min

        if w_max is not None:
            lam_mask &= lam <= w_max

        flux = flux[lam_mask]
        var = var[lam_mask]
        lam = lam[lam_mask]
        flux_not_nan_mask = flux_not_nan_mask[lam_mask]
        rc = rc[lam_mask]

        meta = {'header': hdulist[0].header}
        uncertainty = VarianceUncertainty(var)

        sp: Spectrum = Spectrum(
            flux=flux,
            spectral_axis=lam,
            wcs=flux_wcs,
            uncertainty=uncertainty,
            meta=meta
        )

        sp.mask = flux_not_nan_mask
        sp.rc = rc
        sp.obj_id = None

    return sp


def identifyJWSTStacked(origin, *args, **kwargs) -> bool:
    """Identify spectra extracted with python-specex."""
    if not isFits(args[0]):
        return False

    with fits.open(args[0]) as hdul:
        if len(hdul) < 2:
            return False

        if hdul[0].header.get('NAXIS') != 0:
            return False

        for hdu in hdul[1:]:
            if hdu.name != 'SCI':
                return False

            for key in ['WMIN', 'WMAX', 'DLAM']:
                if key not in hdu.header:
                    return False

    return True


@data_loader(
    label="jwst-stacked-1d",
    identifier=identifyJWSTStacked,
    extensions=['fits']
)
def JWSTStackedLoader(
    file_name: str | os.PathLike[str], *args, **kwargs
) -> Spectrum:
    chunks = []

    rms_file: Path = Path(
        str(file_name).replace('.fits', '_rms.fits')
    )

    with fits.open(file_name) as hdul:
        for j, hdu in enumerate(hdul):
            if j == 0:
                continue

            flux = hdu.data * units.dimensionless_unscaled

            try:
                rms = StdDevUncertainty(fits.getdata(rms_file, ext=j))
            except (OSError, FileNotFoundError):
                rms = None

            header = hdu.header.copy()
            w_min = header['WMIN']
            w_max = header['WMAX']
            d_lam = header['DLAM'] / 10000
            # lam = np.arange(w_min, w_max, step=d_lam) * units.micrometer
            lam = np.linspace(
                w_min, w_max, num=len(flux), endpoint=True
            ) * units.micrometer

            sp = Spectrum(flux=flux, spectral_axis=lam, uncertainty=rms)
            chunks.append(sp)

    new_spectral_axis = np.sort(
        np.concatenate([x.spectral_axis.value for x in chunks])
    ) * units.micrometer
    new_spectral_axis = new_spectral_axis.to(units.angstrom)

    resampler = LinearInterpolatedResampler(
        extrapolation_treatment='zero_fill'
    )

    resampled_chunks = [resampler(x, new_spectral_axis) for x in chunks]

    return sum(resampled_chunks)


def read_sdss_extra(
    file: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        myt = Table.read(file, hdu='COADD')
    except KeyError:
        return None, None

    try:
        wd = myt['wdisp']
    except KeyError:
        wd = None

    try:
        sky = myt['sky']
    except KeyError:
        sky = None

    return sky, wd


class Spectrum2D:
    def __init__(
        self,
        flux: np.ndarray | units.Quantity | None = None,
        spectral_axis: units.Quantity | None = None,
        uncertainty: None | (
            StdDevUncertainty | VarianceUncertainty | InverseVariance
        ) = None
    ) -> None:
        self.wcs: wcs.WCS | None = None
        self.spectral_axis: units.Quantity | None = spectral_axis
        self.flux: np.ndarray | None = None

        if flux is not None:
            if isinstance(flux, units.Quantity):
                self.flux = flux
            else:
                self.flux = flux * units.dimensionless_unscaled

        self.uncertainty = uncertainty
        self.meta: dict[Any, Any] = {}

    @classmethod
    def read(
        cls, file_path: str, ext: int | str | None = None,
        format: str | None = None
    ) -> "Spectrum2D":
        data = None
        header = None
        skip_cards = {
            'SIMPLE', 'XTENSION', 'BITPIX', 'NAXIS',
            'NAXIS1', 'NAXIS2', 'PCOUNT', 'GCOUNT', 'EXTEND'
        }
        if ext is not None:
            data, header = fits.getdata(
                filename=file_path, ext=ext, header=True
            )
        else:
            with fits.open(file_path) as hdul:
                # First check PrimaryHDU (ext 0)
                h0_data = hdul[0].data
                if h0_data is not None and getattr(h0_data, 'ndim', 0) == 2:
                    data = h0_data
                    header = hdul[0].header.copy()
                else:
                    # Look for 2D extension matching known names or any 2D
                    for hdu in hdul:
                        h_ndim = getattr(hdu.data, 'ndim', 0)
                        if (
                            hdu.data is not None
                            and h_ndim == 2
                            and hdu.name.lower() in KNOWN_SPEC_EXT_NAMES
                        ):
                            data = hdu.data
                            header = hdu.header.copy()
                            # Inherit primary header cards if missing
                            for card in hdul[0].header.cards:
                                if (
                                    card.keyword not in header
                                    and card.keyword not in skip_cards
                                ):
                                    header[card.keyword] = (
                                        card.value, card.comment
                                    )
                            break
                    if data is None:
                        for hdu in hdul:
                            h_ndim = getattr(hdu.data, 'ndim', 0)
                            if hdu.data is not None and h_ndim == 2:
                                data = hdu.data
                                header = hdu.header.copy()
                                for card in hdul[0].header.cards:
                                    if (
                                        card.keyword not in header
                                        and card.keyword not in skip_cards
                                    ):
                                        header[card.keyword] = (
                                            card.value, card.comment
                                        )
                                break

        if data is None or header is None:
            data, header = fits.getdata(filename=file_path, ext=0, header=True)


        try:
            flux_units = units.Unit(header['BUNIT'])
        except KeyError:
            try:
                flux_units = units.Unit(header['TUNIT'])
            except KeyError:
                flux_units = 1 * units.dimensionless_unscaled

        sp_2d = cls(flux=data * flux_units)

        try:
            sp_wcs = wcs.WCS(header)
        except Exception:
            sp_wcs = None
        else:
            if not sp_wcs.has_spectral:
                # Try to fix wrong WCSs
                for j in range(sp_wcs.naxis):
                    axis_key = f'CTYPE{j+1:d}'
                    unit_key = f'CUNIT{j+1:d}'

                    axis_type = header.get(axis_key)
                    axis_unit = header.get(unit_key)

                    if axis_unit is None:
                        continue
                    else:
                        try:
                            actual_unit = units.Unit(axis_unit)
                        except Exception:
                            actual_unit = None
                            continue
                        else:
                            try:
                                _ = actual_unit.to(units.angstrom)
                            except Exception:
                                # Not a wavelength
                                continue

                    if axis_type.strip().lower() in KNOWN_WAVE_EXT_NAMES:
                        header[axis_key] = 'WAVE'
                        break
                sp_wcs = wcs.WCS(header)

            sp_2d.wcs = sp_wcs
            if sp_wcs.has_spectral:
                pixel = np.arange(data.shape[1])
                lam = sp_wcs.spectral.pixel_to_world(pixel).Angstrom
                sp_2d.spectral_axis = lam * units.angstrom
            sp_2d.meta['header'] = copy.deepcopy(header)
        return sp_2d


def identifyClashVlt2dFits(origin, *args, **kwargs) -> bool:
    """Identify VIMOS 2D spectra."""
    if not (
        isinstance(args[0], (str, Path)) and
        Path(str(args[0])).suffix.lower() == '.fits'
    ):
        return False

    with fits.open(args[0]) as hdul:
        instrument = hdul[0].header.get('INSTRUME')
        prodcatg = str(hdul[0].header.get('PRODCATG', '')).strip().upper()

        if (instrument is None) or (instrument.strip() != 'VIMOS'):
            return False

        if hdul[0].header.get('NAXIS') == 2 and hdul[0].data is not None:
            if hdul[0].data.shape[0] == 1:
                return False
            return True

        # Check for 2D extension in ESO Phase3 or multi-extension VIMOS
        if prodcatg == 'ANCILLARY.2DSPECTRUM' or 'SPEC_1D' in hdul[0].header:
            for hdu in hdul[1:]:
                if (
                    hdu.header.get('NAXIS') == 2
                    and hdu.data is not None
                    and hdu.data.shape[0] > 1
                ):
                    return True

    return False




@data_loader(
    label="clash-vlt-2d",
    identifier=identifyClashVlt2dFits,
    extensions=['fits']
)
def clashVlt2DFitsLoader(
        file_name: str,
        flux_hdu_index: str | int | None = None,
        resolution: int | float = 1,
        **kwargs
) -> Spectrum2D:
    """
    Load a python-specex spectrum.

    :param file_name:
        The path of the FITS file to read.
    :param flux_hdu_index:
        The extension name or index from which to read the spectrum
    :param resolution:
        The spectral resolution parameter
    :return: A Spectrum object.
    """

    return Spectrum2D.read(file_name, ext=flux_hdu_index)


def identify_format(filename, dtype=Spectrum):
    """
    Reimplemented from specutils
    """
    # check for valid string input
    if (
        not isinstance(filename, (str, Path)) or
        not Path(filename).is_file()
    ):
        raise ValueError(f'{filename} is not a valid string path to a file')

    # identify the file format
    valid_format = registry.identify_format(
        'read', dtype, filename, None, {}, {}
    )

    return valid_format


def read(file: str) -> Spectrum | Spectrum2D:
    sp_formats = identify_format(file)

    spec_class: type[Spectrum] | type[Spectrum2D]
    for spec_format in sp_formats:
        if spec_format == 'clash-vlt-2d':
            spec_class = Spectrum2D
            break
        elif spec_format == 'clash-vlt-1d':
            spec_class = Spectrum
            break
        elif spec_format == 'specex-1d':
            spec_class = Spectrum
            break
    else:
        spec_class = Spectrum
        spec_format = None

    orig_sp: Spectrum | Spectrum2D = spec_class.read(
        file, format=spec_format
    )

    sp: Spectrum | Spectrum2D = type(orig_sp)(
        flux=orig_sp.flux,
        spectral_axis=orig_sp.spectral_axis.to('AA')  # Convert to Angstrom
    )

    sp.uncertainty = orig_sp.uncertainty
    sp.wd = None
    sp.sky = None
    sp.meta = orig_sp.meta

    if sp.meta is None:
        sp.meta = {}

    p_file = Path(file)
    sp.meta['file_name'] = p_file.name
    sp.meta['full_path'] = str(p_file.resolve())


    # Try to read auxiliary data
    if 'SDSS-V spec' in sp_formats:
        sky, wd = read_sdss_extra(file)

        if sky is not None:
            sp.sky = sky * sp.flux.unit
        if wd is not None:
            sp.wd = ((10**wd)*units.Unit('Angstrom')).to(sp.spectral_axis.unit)

    return sp
