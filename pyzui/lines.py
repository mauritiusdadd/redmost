from typing import Optional, Union, List, Tuple

from astropy.modeling import Fittable1DModel, Parameter
from scipy.stats import median_abs_deviation
from scipy.signal import find_peaks_cwt

import numpy as np

from pyzui import utils

# Some important lines with corresponding wavelenghts in Angstrom
RESTFRAME_LINES = [
    (10320, '[SII]', 'E'),
    (8863.0, 'TiO', 'A'),
    (8430.0, 'TiO', 'A'),
    (8195.0, 'NaI', 'A'),
    (8183.0, 'NaI', 'A'),
    (7590.0, 'TiO', 'A'),
    (7065.2, 'HeI', 'AE'),
    (6725.0, '[SII]', 'E'),
    (6562.8, 'Halpha', 'AEB'),
    (6159.0, 'TiO', 'A'),
    (5892.5, 'NaD', 'A'),
    (5603.0, 'TiO', 'A'),
    (5269.0, 'Ca,Fe', 'A'),
    (5175.4, 'MgI', 'A'),
    (5006.8, '[OIII]', 'E'),
    (4958.9, '[OIII]', 'E'),
    (4861.3, 'Hbeta', 'AEB'),
    (4340.4, 'Hgamma', 'AE'),
    (4304.4, 'Gband', 'A'),
    (4216.0, 'CN', 'A'),
    (4101.7, 'Hdelta', 'AE'),
    (4000.0, 'Balmer_Break', 'Break'),
    (4072.0, '[SII]', 'E'),
    (3968.5, 'CaII_H', 'A'),
    (3933.7, 'CaII_K', 'A'),
    (3889.1, 'Hksi,CN(H8)', 'AE'),
    (3869.0, '[NeIII]', 'E'),
    (3797.9, 'Hteta', 'AE'),
    (3770.6, 'H11', 'AE'),
    (3727.5, '[OII]', 'E'),
    (3581.0, 'FeI', 'A'),
    (3425.8, '[NeV]', 'E'),
    (3345.9, '[NeV]', 'E'),
    (2964.0, 'FeII_bump', 'E'),
    (2799.0, 'MgII', 'AEB'),
    (2626.0, 'FeII', 'E'),
    (2600.0, 'FeII', 'A'),
    (2586.7, 'FeII', 'A'),
    (2382.0, 'FeII', 'A'),
    (2374.0, 'FeII', 'A'),
    (2344.2, 'FeII', 'A'),
    (2260.0, 'FeII', 'A'),
    (2142.0, '[NII]', 'E'),
    (1909.0, '[CIII]', 'EB'),
    (1856.0, 'AlIII', 'A'),
    (1670.8, 'AlII', 'A'),
    (1666.1497, 'OIII]', 'E'),
    (1640.0, 'HeII', 'AE'),
    (1608.5, 'FeII', 'A'),
    (1660.8092, 'OIII]', 'E'),
    (1549.0, 'CIV', 'AEB'),
    (1526.7, 'SiII', 'A'),
    (1397.0, 'SiIV+OIV', 'AEB'),
    (1334.5, 'CII', 'AE'),
    (1303.0, 'OI', 'AE'),
    (1260.4, 'SiII', 'A'),
    (1240.0, 'NV', 'AE'),
    (1215.7, 'LyA', 'AEB'),
    (1033.0, 'OVI', 'AE'),
    (1025.6, 'LyB', 'AE'),
    (972.5, 'LyG', 'AE'),
]


def _normal(x, mu, sigma):
    return np.exp(-((x - mu)**2)/(2*sigma)) / (sigma * np.sqrt(2 * np.pi))


class Emission1D(Fittable1DModel):
    """Simple model for a flat spectrum with emission lines."""

    redshift = Parameter()

    def __init__(self, lines_identifications, sigma=3, redshift=0, **kwargs):
        self.line_candidates = lines_identifications
        self.sigma = sigma
        super().__init__(redshift=redshift, **kwargs)

    def evaluate(self, lam: np.ndarray, redshift: float) -> np.ndarray:
        """
        Evaluate the model.

        Parameters
        ----------
        lam : np.ndarray
            Array of wavelength.
        redshift : float, optional
            The redshift of the spectrum.
            The default value is 0.

        Returns
        -------
        result : np.ndarray
            The model values.

        """
        result = np.zeros_like(lam)
        for candidate in self.line_candidates:
            mu = candidate[1] / (1 + redshift)
            result += _normal(lam, mu, self.sigma) * candidate[3]
        return result


def get_spectrum_lines(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    var: Optional[Union[np.ndarray, float]] = None,
    sigma_threshold: float = 10.0,
    smoothing_sigma: float = 10,
) -> Union[None, List[Tuple[int, int, int, int]]]:
    """
    Identify the position of clear emission or absorption lines.

    Parameters
    ----------
    wavelengths : np.ndarray
        The wavelengths corresponding to each flux value.
    flux : numpy.ndarray
        The spectrum itself.
    var : numpy.ndarray, optional
        The variance of the spectrum itself.
        The default value is None.
    sigma_threshold : float, optional
        The threshold to use for line identification.
        The default value is 10.0
    smoothing_sigma : float, optional
        Parameter to be passed to the smoothing function.
        The default value is 10.0.

    Returns
    -------
    identifications : list or None
        A list of tuple. Each tuple ha the form of (k, w, l, h) and contains
        the index k for the wavelength w of the line, the approximate max width
        l of the line and a height h of the line. Note that l and h are not
        actual physical quantities and should be used with caution when
        comparing to other values from a different spectrum.
    """
    if np.isnan(flux).all():
        return None
    else:
        flux = np.ma.array(flux.copy(), mask=np.isnan(flux))

    if var is None:
        var = 1.0
    elif isinstance(var, np.ndarray):
        var = np.ma.array(var.copy(), mask=np.isnan(var))

    smoothed_spec = utils.smooth_fft(flux, sigma=smoothing_sigma)
    smoothed_spec = np.ma.array(smoothed_spec, mask=np.isnan(smoothed_spec))

    # Subtract the smoothed spectrum to the spectrum itself to get a
    # crude estimation of the noise, then square it and divide for the variance
    # and then go back with a square root
    norm_noise = ((flux - smoothed_spec)**2) / var
    norm_noise = np.ma.sqrt(norm_noise)

    # Get the median value of the noise. The median is more robust against the
    # presence of lines with respect to the mean
    noise_median = np.ma.median(norm_noise)

    # Get the NMAD of the noise. We assume here that the noise has a
    # unimodal distribution (eg. gaussian like), and this is a good assumption
    # if the noise is due only to the random fluctuations
    noise_nmad = median_abs_deviation(norm_noise, scale='normal')

    norm_noise_deb = np.abs(norm_noise - noise_median)

    # Get the possible lines
    outlier = norm_noise_deb >= (sigma_threshold * noise_nmad)

    # Delete identification with lenght 1 (almost all are fake)
    for k, v in enumerate(outlier):
        if (k == 0) or (k == len(outlier)-1):
            continue
        if v and ((outlier[k-1] == 0) and (outlier[k+1] == 0)):
            outlier[k] = 0

    # Get position, width and height of the identifications
    identifications = []
    c_start = None
    c_wstart = None
    c_end = None
    for k, v in enumerate(outlier):
        if v:
            if c_start is None:
                c_start = k
                c_wstart = wavelengths[k]
                c_end = None
        elif c_start is not None:
            if c_end is None:
                c_end = k
                c_wh = np.ma.max(norm_noise_deb[c_start: c_end])
                c_wh /= noise_nmad
                c_max_pos = np.ma.argmax(norm_noise_deb[c_start: c_end])
                c_pos_idx = c_start + c_max_pos
                c_wpos = wavelengths[c_pos_idx]
                c_wlen = wavelengths[k] - c_wstart
                identifications.append((c_pos_idx, c_wpos, c_wlen, c_wh))
                c_start = None

    # Sort by height
    identifications.sort(key=lambda a: -a[3])
    return identifications


def get_lines(name=None, line_type=None, wrange=None, z=0):
    """
    Return line data according to the given line name and types.

    Parameters
    ----------
    name : str or None, optional
        The name of the line (eg. CaII_H or FeI, etc...). If None, the lines
        are selected only by type. If both name and type are None, all lines
        are returned.
    line_type : str or None, optional
        Type of the line, can be 'A' (absorption), 'E' (emission) 'B' (Broad).
        If None, then all the line types are returned.
        The default is None.
    wrange : tuple/list/np.ndarray of floats or None, optional
        The wavelength ragne in which lines should be. If None, no selection
        according to the line wavelenght is made.
        The default is None.
    z : float, optional
        The redshit of the lines. The default value is 0.

    Returns
    -------
    selected_lines : list
        List of line data. Each element of the list is a 3-tuple in the form
        (wavelenght in Angstrom, Line name, Line type).

    """
    if name is None:
        selected_lines = RESTFRAME_LINES[:]
    else:
        selected_lines = [
            line
            for line in RESTFRAME_LINES
            if name.lower() == line[1].lower()
        ]
    if line_type is not None:
        selected_lines = [
            line
            for line in selected_lines
            if line_type.lower() in line[2].lower()
        ]

    selected_lines = [
        ((1 + z) * line[0], line[1], line[2])
        for line in selected_lines
    ]

    if wrange is not None:
        w_min = np.nanmin(wrange)
        w_max = np.nanmax(wrange)
        selected_lines = [
            line
            for line in selected_lines
            if w_min <= line[0] <= w_max
        ]

    return selected_lines


def get_redshift_from_lines(
    identifications: Union[tuple, list, np.ndarray],
    z_max: float = 6.0,
    z_min: float = 0.0,
    z_points: Optional[Union[float, None]] = None,
    tol: Optional[float] = None
):
    """
    Get the redshift of a set of line identifications.

    Parameters
    ----------
    identifications : Union[tuple, list, np.ndarray]
        A list of identification generated by get_spectrum_lines().
    z_max : Optional[float], optional
        The maximum redshift. The default is 6.
    z_min : Optional[float], optional
        The minimum redshift. The default is 0.
    z_points : Optional[Union[float, None]], optional
        Number of redshift values between z_min and z_max to test.
        If None, then z_points = 1000*(z_max - z_min).
        The default is None.
    tol : Optional[float], optional
        The tolerance. If None, it is computed automatically.
        The default is None.

    Returns
    -------
    z_values
        Best estimations of the redshift sorted from the most probable to the
        least probable.
    z_probs
        Pseudo-probabilities of the redshift estimations (the higer the better)
    """
    if len(identifications) < 2:
        return None

    if z_points is None:
        z_points = 1000 * (z_max - z_min)

    if tol is None:
        tol = np.mean([x[2] for x in identifications])

    mymodel = Emission1D(identifications, tol, redshift=0)

    z_values = np.linspace(z_min, z_max, z_points)
    prob_values = np.zeros_like(z_values)
    for j, z in enumerate(z_values):
        rest_lines_lam = [
            x[0] for x in get_lines(z=z)
        ]

        prob_values[j] = np.sum(mymodel(rest_lines_lam))

    peak_indices = find_peaks_cwt(prob_values, 1)
    z_values_p = z_values[peak_indices]
    z_prob_p = prob_values[peak_indices]

    z_prob_p_sorted_ind = np.argsort(z_prob_p)[::-1]
    z_values_p = z_values_p[z_prob_p_sorted_ind]
    z_prob_p = z_prob_p[z_prob_p_sorted_ind]

    mean_prob = np.median(z_prob_p)
    std_prob = np.std(z_prob_p)

    plausible_mask = z_prob_p >= mean_prob + std_prob

    return (z_values_p[plausible_mask], z_prob_p[plausible_mask])