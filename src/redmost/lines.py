from typing import Optional, Union, List, Tuple

from astropy.modeling import Fittable1DModel, Parameter  # type: ignore
from scipy.stats import median_abs_deviation  # type: ignore
from scipy.signal import find_peaks_cwt  # type: ignore

import numpy as np

from redmost import utils

# Some important lines with corresponding wavelengths in Angstrom
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
    x = np.array(x, dtype='float64')
    return np.exp(-((x - mu)**2)/(2*sigma)) / (sigma * np.sqrt(2 * np.pi))


class Emission1D(Fittable1DModel):
    """Simple model for a flat spectrum with emission lines."""
    n_inputs = 1
    n_outputs = 1

    redshift = Parameter()

    def __init__(
        self,
        lines_identifications: List[float],
        sigma: float = 1.0,
        redshift: float = 0.0,
        **kwargs
    ):
        self.line_candidates = lines_identifications
        self.sigma = sigma
        super().__init__(redshift=redshift, **kwargs)

    def evaluate(self, x: np.ndarray, redshift: float = 0.0) -> np.ndarray:
        """
        Evaluate the model.

        :param x: Array of wavelength.
        :param redshift: The redshift of the spectrum. The default value is 0.
        :return result: The model values.

        """
        result = np.zeros_like(x, dtype='float64')
        for candidate_lam in self.line_candidates:
            mu = candidate_lam / (1 + redshift)
            result += _normal(x, mu, self.sigma)
        return result


def get_spectrum_lines(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    var: Optional[Union[np.ndarray, float]] = None,
    sigma_threshold: float = 10.0,
    smoothing_sigma: float = 10,
) -> List[Tuple[int, float, float, float]]:
    """
    Identify the position of clear emission or absorption lines.

    :param wavelengths: The wavelengths corresponding to each flux value.
    :param flux: The spectrum itself.
    :param var: The variance of the spectrum itself. The default value is None.
    :param sigma_threshold: The threshold to use for line identification.
                            The default value is 10.0
    :param smoothing_sigma: Parameter to be passed to the smoothing function.
                            The default value is 10.0.
    :return identifications:
        A list of tuple. Each tuple ha the form of (k, w, l, h) and contains
        the index k for the wavelength w of the line, the approximate max width
        l of the line and a height h of the line. Note that l and h are not
        actual physical quantities and should be used with caution when
        comparing to other values from a different spectrum.
    """
    if np.isnan(flux).all():
        return []
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
    identifications: List[Tuple[int, float, float, float]] = []
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
                identifications.append(
                    (int(c_pos_idx), float(c_wpos), float(c_wlen), float(c_wh))
                )
                c_start = None

    # Sort by height
    identifications.sort(key=lambda a: -a[3])
    return identifications


def get_lines(
    name: Optional[str] = None,
    line_type: Optional[str] = None,
    wrange: Optional[Union[np.ndarray, List[float]]] = None,
    z: float = 0
) -> List[Tuple[float, str, str]]:
    """
    Return line data according to the given line name and types.

    :param name:
        The namxe of the line (eg. CaII_H or FeI, etc...). If None, the lines
        are selected only by type. If both name and type are None, all lines
        are returned.
    :param line_type:
        Type of the line, can be 'A' (absorption), 'E' (emission) 'B' (Broad).
        If None, then all the line types are returned.
        The default is None.
    :param wrange:
        The wavelength ragne in which lines should be. If None, no selection
        according to the line wavelenght is made.
        The default is None.
    :param z:
        The redshit of the lines. The default value is 0.
    :return selected_lines:
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
    identifications: List[float],
    z_max: float = 6.0,
    z_min: float = 0.0,
    z_points: Optional[int] = None,
    tol: float = 1.0
) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """
    Get the redshift of a set of line identifications.

    :param identifications:
        A list of identification generated by get_spectrum_lines().
    :param z_max:
        The maximum redshift. The default is 6.
    :param z_min:
        The minimum redshift. The default is 0.
    :param z_points:
        Number of redshift values between z_min and z_max to test.
        If None, then z_points = int(1000*(z_max - z_min)).
        The default is None.
    :param tol:
        The tolerance. The default is 0.1.

    :return z_values:
        Best estimations of the redshift sorted from the most probable to the
        least probable.
    :return z_probs:
        Pseudo-probabilities of the estimation (the higher the better).
    """
    if len(identifications) < 2:
        return None

    if z_points is None:
        z_points = int(1000 * (z_max - z_min) / tol)

    mymodel = Emission1D(identifications, sigma=tol, redshift=0)

    z_values = np.linspace(
        start=z_min, stop=z_max, num=z_points
    )

    if len(z_values) <= 1:
        return None

    prob_values = np.zeros_like(z_values)
    for j, z in enumerate(z_values):
        rest_lines_lam = np.array([
            x[0] for x in get_lines(z=z)
        ])

        # We compute the matching at object restframe so redshift=0
        prob_values[j] = np.sum(mymodel.evaluate(rest_lines_lam))

    peak_indices = find_peaks_cwt(prob_values, 1)
    z_values_p = z_values[peak_indices]
    z_prob_p = prob_values[peak_indices]

    z_prob_p_sorted_ind = np.argsort(z_prob_p)[::-1]
    z_values_p = z_values_p[z_prob_p_sorted_ind]
    z_prob_p = z_prob_p[z_prob_p_sorted_ind]

    mean_prob = np.median(z_prob_p)
    std_prob = np.std(z_prob_p)

    plausible_mask = z_prob_p >= mean_prob + std_prob

    best_z_values = z_values_p[plausible_mask]
    best_z_probs = z_prob_p[plausible_mask]

    return best_z_values, best_z_probs
