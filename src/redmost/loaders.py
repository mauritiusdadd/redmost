import os
from typing import Optional, Union, List, Tuple

import numpy as np

from astropy.nddata import VarianceUncertainty  # type: ignore
from astropy.io import fits  # type: ignore
from astropy.table import Table  # type: ignore
from astropy import wcs  # type: ignore
from astropy import units  # type: ignore
from specutils import Spectrum1D  # type: ignore
from specutils.io.registers import identify_spectrum_format  # type: ignore
from specutils.io.registers import data_loader  # type: ignore

KNOWN_SPEC_EXT_NAMES = ['spec', 'spectrum', 'flux', 'data', 'sci', 'science']
KNOWN_VARIANCE_EXT_NAMES = ['stat', 'stats', 'var', 'variance', 'noise', 'err']
KNOWN_INVAR_EXT_NAMES = ['ivar', 'ivariance']
KNOWN_MASK_EXT_NAMES = ['mask', 'platemask', 'footprint', 'dq', 'nan_mask']
KNOWN_WAVE_EXT_NAMES = ['wave', 'wavelenght', 'lambda', 'lam']
KNOWN_RCURVE_EXT_NAMES = ['r', 'reso', 'resolution', 'rcurve', 'wd']
KNOWN_RGB_EXT_NAMES = ['r', 'g', 'b', 'red', 'green', 'blue']


def identifySpecexFits(origin, *args, **kwargs) -> bool:
    """Identify spectra extracted with python-specex."""
    if not (
        isinstance(args[0], str) and
        os.path.splitext(args[0].lower())[1] == '.fits'
    ):
        return False

    valid_id_keys = [
        f"{i}{j}"
        for i in ['', 'OBJ', 'OBJ_', 'TARGET', 'TARGET_']
        for j in ['ID', 'NUMBER', 'UID', 'UUID']
    ]

    with fits.open(args[0]) as hdul:
        for hdu in hdul:
            for key in valid_id_keys:
                try:
                    _ = hdu.header[key]
                except KeyError:
                    continue
                else:
                    return True
    return False


@data_loader(
    label="specex-1d",
    identifier=identifySpecexFits,
    extensions=['fits']
)
def specexFitsLoader(
    file_name: str,
    flux_hdu_index: Optional[Union[str, int]] = None,
    resolution: Union[int, float] = 1,
    **kwargs
) -> Spectrum1D:
    """
    Load a python-specex spectrum.

    :param file_name:
        The path of the FITS file to read.
    :param flux_hdu_index:
        The extension name or index from which to read the spectrum
    :param resolution:
        The spectral resolution parameter
    :return: A Spectrum1D object.
    """
    def getHDU(
        hdul: fits.HDUList,
        valid_names: List[str],
        index: Optional[Union[str, int]] = None
    ):
        if index is None:
            for hdu in hdul:
                if hdu.name.lower() in valid_names:
                    return hdu
            else:
                return None
        else:
            return hdul[index]

    with fits.open(file_name, **kwargs) as hdulist:
        flux_hdu = getHDU(hdulist, KNOWN_SPEC_EXT_NAMES, index=flux_hdu_index)
        var_hdu = getHDU(hdulist, KNOWN_VARIANCE_EXT_NAMES)
        mask_hdu = getHDU(hdulist, KNOWN_MASK_EXT_NAMES)
        rc_hdu = getHDU(hdulist, KNOWN_RCURVE_EXT_NAMES)

        valid_id_keys: List[str] = [
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

        if var_hdu is None:
            var = np.ones_like(flux)
        else:
            var = var_hdu.data

        if mask_hdu is None:
            mask = None
        else:
            mask = mask_hdu.data

        if rc_hdu is None:
            rc = None
        else:
            rc = rc_hdu.data

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

        flux_units = units.Unit(flux_hdu.header['BUNIT'])
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

        sp: Spectrum1D = Spectrum1D(
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

def read_sdss_extra(
    file: str
) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
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

def read(file: str) -> Spectrum1D:
    sp_formats = identify_spectrum_format(file)
    sp = Spectrum1D.read(file)
    sp.wd = None
    sp.sky = None

    # Try to read auxiliary data
    if 'SDSS-V spec' in sp_formats:
        sky, wd = read_sdss_extra(file)

        if sky is not None:
            sp.sky = sky * sp.flux.unit
        if wd is not None:
            sp.wd = ((10**wd)*units.Unit('Angstrom')).to(sp.spectral_axis.unit)

    return sp
