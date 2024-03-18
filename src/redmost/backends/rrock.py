import os
import uuid
import pickle
import argparse
from typing import Tuple, List, Dict, Optional, Any, Callable

import numpy as np

from scipy import sparse  # type: ignore

from astropy.nddata import VarianceUncertainty  # type: ignore
from astropy.nddata import StdDevUncertainty  # type: ignore
from astropy.nddata import InverseVariance  # type: ignore

from specutils import Spectrum1D  # type: ignore

try:
    import redrock
except (ImportError, ModuleNotFoundError):
    HAS_REDROCK = False

    def get_templates(*args, **kwargs):
        raise NotImplementedError()

    def get_template_types(*args, **kwargs):
        raise NotImplementedError()


    def build_redrock_targets(*args, **kwargs):
        raise NotImplementedError()


    def run_redrock(*args, **kwargs):
        raise NotImplementedError()

else:
    HAS_REDROCK = True

    from redrock.utils import elapsed, get_mp
    from redrock.targets import Spectrum, Target, DistTargetsCopy
    from redrock.templates import load_dist_templates, find_templates, Template
    from redrock.zfind import zfind

    def get_templates(
        template_types: Optional[List[str]] = None,
        filepath: bool = False,
        templates: Optional[str] = None
    ) -> List[Template]:
        """
        Get avilable templates.

        :param template_types:
            List of template types to retrieve. If it's empty all available
            templates will be returned.
            The default is [].
        :param filepath:
            If it's true then return the file paths instead of actual templates.
        :param templates:
            The path of a template file or of a directory containing templates
            files. If None, templates are searched in the default redrock path.
            The default value is None.

        :return available_templates:
            The available templates or the corresponding file paths.

        """

        if templates is not None and os.path.isfile(templates):
            return [Template(templates), ]

        available_templates = []
        for t in find_templates(templates):
            templ = Template(t)
            if (not template_types) or (templ.template_type in template_types):
                if filepath:
                    available_templates.append(t)
                else:
                    available_templates.append(templ)

        return available_templates


    def get_template_types() -> List[str]:
        """
        Get the available types of templates.

        :return types: List of types of available templates.
        """
        templates = [
            t.template_type
            for t in get_templates()
        ]
        types = list(set(templates))
        return types


    def build_redrock_targets(
        spectra: Dict[uuid.UUID, Spectrum1D],
        lambda_min_ang: float = 3500.0,
        lambda_max_ang: float = 10000.0,
        progress_callback: Optional[Callable] = None
    ) -> List[Target]:

        target_dict = {}

        targets: List[Target] = []
        sp: Spectrum1D
        sp_uuid: uuid.UUID
        for j, (sp_uuid, sp) in enumerate(spectra.items()):
            target_id = f"{j:06d}"
            if progress_callback is not None:
                progress_callback(j + 1)

            flux: np.ndarray = sp.flux.value.astype('float32')
            wave: np.ndarray = sp.spectral_axis.value.astype('float32')

            wd: np.ndarray
            try:
                wd = sp.wd.value
            except AttributeError:
                delta_lambda = np.ones_like(wave)
                delta_lambda[1:] = (wave[1:] - wave[:-1])
                wd = 2.0 / delta_lambda
                wd[0] = wd[1]

            ivar: np.ndarray
            if isinstance(sp.uncertainty, VarianceUncertainty):
                ivar = 1 / sp.uncertainty.array
            elif isinstance(sp.uncertainty, InverseVariance):
                ivar = sp.uncertainty.array
            elif isinstance(sp.uncertainty, StdDevUncertainty):
                ivar = 1 / (sp.uncertainty.array ** 2)
            else:
                ivar = np.ones_like(flux)

            not_nan_mask = np.isfinite(flux) | np.isfinite(ivar)

            flux = flux[not_nan_mask]
            ivar = ivar[not_nan_mask]
            wave = wave[not_nan_mask]
            wd = wd[not_nan_mask]

            wd[wd < 1e-3] = 2.

            # clip to template minimum and maximum range
            imin = abs(wave - lambda_min_ang).argmin()
            imax = abs(wave - lambda_max_ang).argmin()

            wave = wave[imin:imax]
            flux = flux[imin:imax]
            ivar = ivar[imin:imax]
            wd = wd[imin:imax]

            ndiag = int(4 * np.ceil(wd.max()) + 1)
            nbins = wd.shape[0]

            ii = np.arange(wave.shape[0])
            di = ii - ii[:, None]
            di2 = di ** 2

            # build resolution from wdisp
            reso = np.zeros([ndiag, nbins])

            for idiag in range(ndiag):
                offset = ndiag // 2 - idiag
                d = np.diagonal(di2, offset=offset)
                if offset < 0:
                    reso[idiag, :len(d)] = np.exp(-d / 2 / wd[:len(d)] ** 2)
                else:
                    reso[idiag, nbins - len(d):nbins] = np.exp(
                        -d / 2 / wd[nbins - len(d):nbins] ** 2
                    )

            reso /= np.sum(reso, axis=0)
            offsets = ndiag // 2 - np.arange(ndiag)
            nwave = reso.shape[1]
            reso = sparse.dia_matrix((reso, offsets), (nwave, nwave))

            rrspec = Spectrum(wave, flux, ivar, reso, None)
            target = Target(target_id, [rrspec])
            target_dict[target_id] = sp_uuid
            targets.append(target)

        return targets, target_dict

    def run_redrock(targets: List[Target]) -> Tuple[Any, Any]:
        start = elapsed(None, "", comm=None)

        dtargets = DistTargetsCopy(targets, comm=None, root=0)

        # Get the dictionary of wavelength grids
        dwave = dtargets.wavegrids()

        _ = elapsed(
            start,
            "Distribution of {} targets".format(len(dtargets.all_target_ids)),
            comm=None
        )

        # Read the template data
        dtemplates = load_dist_templates(
            dwave,
            templates=None,
            comm=None,
            mp_procs=1,
            # use_gpu=True,
            # gpu_mode=True
        )

        opt_zfind_args = {}
        # TODO: do more tests before enabling this
        # opt_zfind_args['use_gpu'] = False

        # Compute the redshifts, including both the coarse scan and the
        # refinement.  This function only returns data on the rank 0 process.
        start = elapsed(None, "", comm=None)
        scandata, zfit = zfind(
            dtargets,
            dtemplates,
            mp_procs=1,
            nminima=3,
            archetypes=None,
            priors=None,
            chi2_scan=None,
            **opt_zfind_args
        )

        _ = elapsed(start, "Computing redshifts took", comm=None)

        return (scandata, zfit)


def dump_targets(targets: List[Any], pickle_file: str):
    with open(pickle_file, 'wb') as f:
        pickle.dump(targets, f)


def run_redrock_on_pickled_targets(pickle_file: str):
    try:
        with open(pickle_file, 'rb') as f:
            targets: List[Any] = pickle.load(f)
    except Exception:
        return

    return run_redrock(targets)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'in_pickle_file'
    )

    parser.add_argument(
        'out_pickle_file'
    )

    args = parser.parse_args()
    result = run_redrock_on_pickled_targets(args.in_pickle_file)

    with open(args.out_pickle_file, "wb") as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    main()