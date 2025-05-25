"""Spectral features functions"""

import numpy as np
import scipy
from features_extraction.features_functions import calc_fft, filterbank, wavelet
from spectral_utils.utils import calc_fft, spectral_spread, spectral_centroid

def spectral_distance(signal: np.ndarray, fs: int) -> float:
    """Computes the signal spectral distance.

    Distance of the signal's cumulative sum of the FFT elements to
    the respective linear regression.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        spectral distance

    """
    f, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag)

    # Computing the linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    return np.sum(points_y - cum_fmag)

def fundamental_frequency(signal: np.ndarray, fs: int) -> float:
    """Computes fundamental frequency of the signal.

    The fundamental frequency integer multiple best explain
    the content of the signal spectrum.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : int
        Sampling frequency

    Returns
    -------
    f0: float
       Predominant frequency of the signal

    """
    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

    # Finding big peaks, not considering noise peaks with low amplitude

    bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

    # # Condition for offset removal, since the offset generates a peak at frequency zero
    bp = bp[bp != 0]
    if not list(bp):
        f0 = 0
    else:
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0

def max_power_spectrum(signal: np.ndarray, fs: int) -> float:
    """Computes maximum power spectrum density of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : scalar
        Sampling frequency

    Returns
    -------
    nd-array
        Max value of the power spectrum density

    """
    if np.std(signal) == 0:
        return float(max(scipy.signal.welch(signal, fs, nperseg=len(signal))[1]))
    else:
        return float(max(scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))[1]))

def max_frequency(signal: np.ndarray, fs: int) -> float:
    """Computes maximum frequency of the signal.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]

def median_frequency(signal: np.ndarray, fs: int) -> float:
    """Computes median frequency of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median frequency is computed
    fs: int
        Sampling frequency

    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]

    return f_median

def spectral_decrease(signal: np.ndarray, fs: int) -> float:
    """Represents the amount of decreasing of the spectra amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral decrease is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral decrease

    """
    f, fmag = calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag) + 1)

    # Sum of numerator
    soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    if not np.sum(fmag_band):
        return 0
    else:
        # Sum of denominator
        soma_den = 1 / np.sum(fmag_band)

        # Spectral decrease computing
        return soma_den * soma_num

def spectral_kurtosis(signal: np.ndarray, fs: int) -> float:
    """Measures the flatness of a distribution around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral kurtosis is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Kurtosis

    """
    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)

def spectral_slope(signal: np.ndarray, fs: int) -> float:
    """Computes the spectral slope.

    Spectral slope is computed by finding constants m and b of the function aFFT = mf + b, obtained by linear regression
    of the spectral amplitude.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral slope is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Slope

    """
    f, fmag = calc_fft(signal, fs)
    sum_fmag = fmag.sum()
    dot_ff = (f * f).sum()
    sum_f = f.sum()
    len_f = len(f)

    if not ([f]) or (sum_fmag == 0):
        return 0
    else:
        if not (len_f * dot_ff - sum_f ** 2):
            return 0
        else:
            num_ = (1 / sum_fmag) * (len_f * np.sum(f * fmag) - sum_f * sum_fmag)
            denom_ = (len_f * dot_ff - sum_f ** 2)
            return num_ / denom_

def spectral_variation(signal: np.ndarray, fs: int) -> float:
    """Computes the amount of variation of the spectrum along time.

    Spectral variation is computed from the normalized cross-correlation between two consecutive amplitude spectra.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral variation is computed.
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Variation

    """
    f, fmag = calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:] ** 2)
    sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1 - (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5)))

    return variation

def spectral_roll_off(signal: np.ndarray, fs: int) -> float:
    """Computes the spectral roll-off of the signal.

    The spectral roll-off corresponds to the frequency where 95% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-off is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-off

    """
    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.95 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]

def spectral_roll_on(signal: np.ndarray, fs: int) -> float:
    """Computes the spectral roll-on of the signal.

    The spectral roll-on corresponds to the frequency where 5% of the signal magnitude is contained
    below of this value.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-on is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral roll-on

    """
    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.05 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]

def human_range_energy(signal: np.ndarray, fs: int) -> float:
    """Computes the human range energy ratio.

    The human range energy ratio is given by the ratio between the energy
    in frequency 0.6-2.5Hz and the whole energy band.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which human range energy ratio is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Human range energy ratio

    """
    f, fmag = calc_fft(signal, fs)

    allenergy = np.sum(fmag ** 2)

    if allenergy == 0:
        # For handling the occurrence of Nan values
        return 0.0

    hr_energy = np.sum(fmag[np.argmin(np.abs(0.6 - f)):np.argmin(np.abs(2.5 - f))] ** 2)

    ratio = hr_energy / allenergy

    return ratio

def power_bandwidth(signal: np.ndarray, fs: int) -> float:
    """Computes power spectrum density bandwidth of the signal.

    It corresponds to the width of the frequency band in which 95% of its power is located.

    Description in article:
    Power Spectrum and Bandwidth Ulf Henriksson, 2003 Translated by Mikael Olofsson, 2005

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the power bandwidth computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Occupied power in bandwidth

    """
    # Computing the power spectrum density
    if np.std(signal) == 0:
        freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
    else:
        freq, power = scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))

    if np.sum(power) == 0:
        return 0.0

    # Computing the lower and upper limits of power bandwidth
    cum_power = np.cumsum(power)
    f_lower = freq[np.where(cum_power >= cum_power[-1] * 0.95)[0][0]]

    cum_power_inv = np.cumsum(power[::-1])
    f_upper = freq[np.abs(np.where(cum_power_inv >= cum_power[-1] * 0.95)[0][0] - len(power) + 1)]

    # Returning the bandwidth in terms of frequency

    return np.abs(f_upper - f_lower)

def spectral_entropy(signal: np.ndarray, fs: int) -> float:
    """Computes the spectral entropy of the signal based on Fourier transform.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which spectral entropy is computed
    fs : int
        Sampling frequency

    Returns
    -------
    float
        The normalized spectral entropy value

    """
    # Removing DC component
    sig = signal - np.mean(signal)

    f, fmag = calc_fft(sig, fs)

    power = fmag ** 2

    if power.sum() == 0:
        return 0.0

    prob = np.divide(power, power.sum())

    prob = prob[prob != 0]

    # If probability all in one value, there is no entropy
    if prob.size == 1:
        return 0.0

    return -np.multiply(prob, np.log2(prob)).sum() / np.log2(prob.size)

def wavelet_entropy(signal: np.ndarray, fs: int, function: callable = scipy.signal.ricker, widths: np.ndarray = np.arange(1, 10)) -> float:
    """Computes CWT entropy of the signal.

    Implementation details in:
    https://dsp.stackexchange.com/questions/13055/how-to-calculate-cwt-shannon-entropy
    B.F. Yan, A. Miyamoto, E. Bruhwiler, Wavelet transform-based modal parameter identification considering uncertainty

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    float
        wavelet entropy

    """
    if np.sum(signal) == 0:
        return 0.0

    cwt = wavelet(signal, function, widths)
    energy_scale = np.sum(np.abs(cwt), axis=1)
    t_energy = np.sum(energy_scale)
    prob = energy_scale / t_energy
    w_entropy = -np.sum(prob * np.log(prob))

    return w_entropy


#######################################################################################################################
# Spectral features functions with multiple features which numbers depending on parameters 
#######################################################################################################################

''' 
https://fr.wikipedia.org/wiki/Cepstre 
return num_ceps dimensions in a tuple
'''

def mfcc(signal: np.ndarray, fs: int, pre_emphasis: float = 0.97, nfft: int = 512, nfilt: int = 40, num_ceps: int = 12, cep_lifter: int = 22) -> tuple:
    """Computes the MEL cepstral coefficients.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which MEL coefficients is computed
    fs : int
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters
    num_ceps: int
        Number of cepstral coefficients
    cep_lifter: int
        Filter length

    Returns
    -------
    tuple
        MEL cepstral coefficients

    """
    filter_banks = filterbank(signal, fs, pre_emphasis, nfft, nfilt)

    mel_coeff = scipy.fft.dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_ceps + 1)]  # Keep 2-13

    mel_coeff -= (np.mean(mel_coeff, axis=0) + 1e-8)

    # liftering
    ncoeff = len(mel_coeff)
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)  # cep_lifter = 22 from python_speech_features library

    mel_coeff *= lift

    return tuple(mel_coeff)


def fft_mean_coeff(signal: np.ndarray, fs: int, nfreq: int = 10) -> tuple:
    """Computes the mean value of each spectrogram frequency.

    nfreq can not be higher than half signal length plus one.
    When it does, it is automatically set to half signal length plus one.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which fft mean coefficients are computed
    fs : int
        Sampling frequency
    nfreq : int
        The number of frequencies. Cannot be higher than half of acquisition frequency.

    Returns
    -------
    tuple
        The mean value of each spectrogram frequency

    """
    if nfreq > len(signal) // 2 + 1:
        nfreq = len(signal) // 2 + 1

    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)

    return tuple(fmag_mean)


def lpcc(signal: np.ndarray, n_coeff: int = 12) -> tuple:
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction cepstral coefficients are computed
    n_coeff : int
        Number of coefficients. https://tsfel.readthedocs.io/en/latest/_modules/tsfel/feature_extraction/features_utils.html

    Returns
    -------
    tuple
        Linear prediction cepstral coefficients

    """
    # 12-20 cepstral coefficients are sufficient for speech recognition
    lpc_coeffs = lpc(signal, n_coeff)

    if np.sum(lpc_coeffs) == 0:
        return tuple(np.zeros(len(lpc_coeffs)))

    # Power spectrum
    powerspectrum = np.abs(np.fft.fft(lpc_coeffs)) ** 2
    lpcc_coeff = np.fft.ifft(np.log(powerspectrum))

    return tuple(np.abs(lpcc_coeff))

def wavelet_abs_mean(signal: np.ndarray, function: callable = scipy.signal.ricker, widths: np.ndarray = np.arange(1, 10)) -> tuple:
    """Computes CWT absolute mean value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT absolute mean value

    """
    return tuple(np.abs(np.mean(wavelet(signal, function), axis=1)))


def wavelet_std(signal: np.ndarray, function: callable = scipy.signal.ricker, widths: np.ndarray = np.arange(1, 10)) -> tuple:
    """Computes CWT std value of each wavelet scale.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT std

    """
    return tuple((np.std(wavelet(signal, function, widths), axis=1)))

def wavelet_energy(signal: np.ndarray, function: callable = scipy.signal.ricker, widths: np.ndarray = np.arange(1, 10)) -> tuple:
    """Computes CWT energy of each wavelet scale.

    Implementation details:
    https://stackoverflow.com/questions/37659422/energy-for-1-d-wavelet-in-python

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Input from which CWT is computed
    function :  wavelet function
        Default: scipy.signal.ricker
    widths :  nd-array
        Widths to use for transformation
        Default: np.arange(1,10)

    Returns
    -------
    tuple
        CWT energy

    """
    cwt = wavelet(signal, function, widths)
    energy = np.sqrt(np.sum(cwt ** 2, axis=1) / np.shape(cwt)[1])

    return tuple(energy)