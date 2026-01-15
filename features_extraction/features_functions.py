
'''
As most of the frequential features are directly extracted from the TSFEL library (https://tsfel.readthedocs.io/en/latest/),
here is the offcial disclaimer
'''
''' 
BSD 3-Clause License

Copyright (c) 2020, Fraunhofer AICOS
All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "TSFEL"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import numpy as np
import pandas as pd
import scipy.signal
import pywt


#######################################################################################################################
# preprocessing functions for spectral features calculation
#######################################################################################################################

def compute_time(signal, fs):
        
    """Creates the signal correspondent time array.

    Parameters
    ----------
    signal: nd-array
        Input from which the time is computed.
    fs: int
        Sampling Frequency

    Returns
    -------
    time : float list
        Signal time

    """
    return np.arange(0, len(signal))/fs

def filterbank(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40):
    """Computes the MEL-spaced filterbank.

    It provides the information about the power in each frequency band.

    Implementation details and description on:
    https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial
    https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html#fnref:1

    Parameters
    ----------
    signal : nd-array
        Input from which filterbank is computed
    fs : int
        Sampling frequency
    pre_emphasis : float
        Pre-emphasis coefficient for pre-emphasis filter application
    nfft : int
        Number of points of fft
    nfilt : int
        Number of filters

    Returns
    -------
    nd-array
        MEL-spaced filterbank

    """

    # Signal is already a window from the original signal, so no frame is needed.
    # According to the references it is needed the application of a window function such as
    # hann window. However if the signal windows don't have overlap, we will lose information,
    # as the application of a hann window will overshadow the windows signal edges.

    # pre-emphasis filter to amplify the high frequencies

    emphasized_signal = np.append(np.array(signal)[0], np.array(signal[1:]) - pre_emphasis * np.array(signal[:-1]))

    # Fourier transform and Power spectrum
    mag_frames = np.absolute(np.fft.rfft(emphasized_signal, nfft))  # Magnitude of the FFT

    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    filter_bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):

        f_m_minus = int(filter_bin[m - 1])  # left
        f_m = int(filter_bin[m])  # center
        f_m_plus = int(filter_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - filter_bin[m - 1]) / (filter_bin[m] - filter_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (filter_bin[m + 1] - k) / (filter_bin[m + 1] - filter_bin[m])

    # Area Normalization
    # If we don't normalize the noise will increase with frequency because of the filter width.
    enorm = 2.0 / (hz_points[2:nfilt + 2] - hz_points[:nfilt])
    fbank *= enorm[:, np.newaxis]

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    return filter_banks  

def lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients

    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have a length >= n_coeff")

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, 'full')

    r = np.zeros(order+1, 'float32')
    # Assuring that works for all type of input lengths
    nx = np.min([order+1, len(signal)])
    r[:nx] = acf[len(signal)-1:len(signal)+order]

    smatrix = ff.create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order+1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.], lpc_coeffs)))

def calc_fft(signal, fs):
    """ This functions computes the fft of a signal.
    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : int
        Sampling frequency
    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)
    """

    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()

def spectral_spread(signal, fs):
    """Measures the spread of the spectrum around its mean value.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral spread is computed.
    fs : int
        Sampling frequency

    Returns
    -------
    float
        Spectral Spread

    """
    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5

def lpc(signal, n_coeff=12):
    """Computes the linear prediction coefficients.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
        Linear prediction coefficients

    """

    if signal.ndim > 1:
        raise ValueError("Only 1 dimensional arrays are valid")
    if n_coeff > signal.size:
        raise ValueError("Input signal must have a length >= n_coeff")

    # Calculate the order based on the number of coefficients
    order = n_coeff - 1

    # Calculate LPC with Yule-Walker
    acf = np.correlate(signal, signal, 'full')

    r = np.zeros(order+1, 'float32')
    # Assuring that works for all type of input lengths
    nx = np.min([order+1, len(signal)])
    r[:nx] = acf[len(signal)-1:len(signal)+order]

    smatrix = create_symmetric_matrix(r[:-1], order)

    if np.sum(smatrix) == 0:
        return tuple(np.zeros(order+1))

    lpc_coeffs = np.dot(np.linalg.inv(smatrix), -r[1:])

    return tuple(np.concatenate(([1.], lpc_coeffs)))

def create_symmetric_matrix(acf, order=11):
    """Computes a symmetric matrix.

    Implementation details and description in:
    https://ccrma.stanford.edu/~orchi/Documents/speaker_recognition_report.pdf

    Parameters
    ----------
    acf : nd-array
        Input from which a symmetric matrix is computed
    order : int
        Order

    Returns
    -------
    nd-array
        Symmetric Matrix

    """

    smatrix = np.empty((order, order))
    xx = np.arange(order)
    j = np.tile(xx, order)
    i = np.repeat(xx, order)
    smatrix[i, j] = acf[np.abs(i - j)]

    return smatrix

# def wavelet(signal, function=scipy.signal.ricker, widths=np.arange(1, 10)):
#     """Computes CWT (continuous wavelet transform) of the signal.

#     Parameters
#     ----------
#     signal : nd-array
#         Input from which CWT is computed
#     function :  wavelet function
#         Default: scipy.signal.ricker
#     widths :  nd-array
#         Widths to use for transformation
#         Default: np.arange(1,10)

#     Returns
#     -------
#     nd-array
#         The result of the CWT along the time axis
#         matrix with size (len(widths),len(signal))

#     """

#     if isinstance(function, str):
#         function = eval(function)

#     if isinstance(widths, str):
#         widths = eval(widths)

#     cwt = scipy.signal.cwt(signal, function, widths)

#     return cwt

# CHANGED WITH 

# def wavelet(signal, function = 'mexh', widths=np.arange(1, 10)):

#     if isinstance(widths, str):
#         widths = eval(widths)

#     cwt = pywt.cwt(signal, widths, function )
#     print("cqt = ",cwt)
#     print(cwt[0])

#     return cwt

###################"""""" 08-01-2026 ############
from scipy.signal import convolve

def ricker(points, a):
    """Retourne une ondelette de Ricker de longueur 'points' et de largeur 'a'."""
    x = np.linspace(-(a * np.sqrt(6)), a * np.sqrt(6), points)
    x = (2 / (np.sqrt(3 * a) * (np.pi ** 0.25))) * (1 - (x / a) ** 2) * np.exp(-(x ** 2) / (2 * a ** 2))
    return x

# def wavelet(signal, widths=np.arange(1, 10)):
#     cwt = np.zeros((len(widths), len(signal)))
#     for i, width in enumerate(widths):
#         psi = ricker(len(signal), width)
#         cwt[i, :] = np.convolve(signal, psi, mode='same')
#     return cwt


def wavelet(signal, function=ricker, widths=np.arange(1, 10)):
    """
    Calcule la transformée en ondelettes continues (CWT) avec une ondelette générique.

    Paramètres :
    - signal : tableau 1D, le signal d'entrée
    - function : fonction qui génère l'ondelette (doit prendre (points, a) en arguments)
    - widths : tableau de largeurs pour l'ondelette

    Retourne :
    - cwt : tableau 2D, la transformée en ondelettes
    """

    if isinstance(function, str):
        function = eval(function)

    if isinstance(widths, str):
        widths = eval(widths)

    cwt = np.zeros((len(widths), len(signal)))
    for i, width in enumerate(widths):
        psi = function(len(signal), width)
        cwt[i, :] = convolve(signal, psi, mode='same')
    return cwt

#######################################################################################################################
# fourier Fab
#######################################################################################################################

def calcul_3d_fourier(x,y,z):
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)

    fft_x = np.abs(np.fft.fft(x))
    fft_y = np.abs(np.fft.fft(y))
    fft_z = np.abs(np.fft.fft(z))

    return(np.linalg.norm([fft_x,fft_y,fft_z],2,axis = 0))

def get_df_freq(fft,freq_acq):
    
    ''' 
    Je retourne juste un df avec une colonne avec les fréquences renseignées
    L'intérêt est que c'est insensible au changement de taille d'epoch
    '''

    nb_intervalles = len(fft) / freq_acq
    nb_coef = len(fft)
    col_freq = np.arange(nb_coef) / (nb_intervalles)
    df_freq = pd.DataFrame(data = col_freq,columns=['freq'])
    df_freq['coef'] = fft / (freq_acq * 2)

    return(df_freq.iloc[0:nb_coef // 2])

def get_spectrum_features(df_freq,freq_acq,l_custom_freq = [0,5,8,12,25]):
    
    dict_features = dict()
    
    for i in range(freq_acq // 2):
        feature_name = 'F_' + str(i) + '_' + str(i+1)
        df_loc = df_freq[(df_freq['freq'] >= i) & (df_freq['freq'] < i + 1)]
        feature_value = df_loc.sum()['coef']
        dict_features[feature_name] = feature_value

    for i in range(len(l_custom_freq) - 1):
        low = l_custom_freq[i]
        up = l_custom_freq[i + 1]
        feature_name = 'F_custom_' + str(low) + '_' + str(up)
        df_loc = df_freq[(df_freq['freq'] >= low) & (df_freq['freq'] < up)]
        feature_value = df_loc.sum()['coef']
        dict_features[feature_name] = feature_value
    
    return(dict_features)



#######################################################################################################################
# temporal features function plus dictionary
#######################################################################################################################



def autocorr(signal):

    """Computes autocorrelation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which autocorrelation is computed

    Returns
    -------
    float
        Cross correlation of 1-dimensional sequence

    """
    signal = np.array(signal)
    return float(np.correlate(signal, signal))

def zero_crossing(signal):
    signal = signal - np.mean(signal)
    return len(np.where(np.diff(np.sign(signal)))[0])

def mean_abs_diff(signal):
    """Computes mean absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which mean absolute deviation is computed

   Returns
   -------
   float
       Mean absolute difference result

   """
    return np.mean(np.abs(np.diff(signal)))
def distance(signal):
    """Computes signal traveled distance.

    Calculates the total distance traveled by the signal
    using the hipotenusa between 2 datapoints.

   Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which distance is computed

    Returns
    -------
    float
        Signal distance

    """
    diff_sig = np.diff(signal).astype(float)
    return np.sum([np.sqrt(1 + diff_sig ** 2)])
def sum_abs_diff(signal):
    """Computes sum of absolute differences of the signal.

   Feature computational cost: 1

   Parameters
   ----------
   signal : nd-array
       Input from which sum absolute difference is computed

   Returns
   -------
   float
       Sum absolute difference result

   """
    return np.sum(np.abs(np.diff(signal)))
def slope(signal):
    """Computes the slope of the signal.

    Slope is computed by fitting a linear equation to the observed data.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which linear equation is computed

    Returns
    -------
    float
        Slope

    """
    t = np.linspace(0, len(signal) - 1, len(signal))

    return np.polyfit(t, signal, 1)[0]
def abs_energy(signal):
    """Computes the absolute energy of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        Absolute energy

    """
    return np.sum(np.abs(signal) ** 2)
def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which the area under the curve is computed

    Returns
    -------
    float
        peak to peak distance

    """
    return np.abs(np.max(signal) - np.min(signal))
def entropy(signal):
    
    ''' 
    fab je rajoute round de signal car mes valeurs sont continues
    '''
    signal = np.round(signal,2)
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed

    Returns
    -------
    float
        The normalized entropy value

    """

    value, counts = np.unique(signal, return_counts=True)
    p = counts / counts.sum()


    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return - np.sum(p * np.log2(p)) / np.log2(len(signal))

    return(np.mean(signal))

dict_time_domain_mvt_features = {'autocorr' : autocorr,
                                'zero_crossing' : zero_crossing,
                                'mean_abs_diff' : mean_abs_diff,
                                'distance' : distance,
                                'sum_abs_diff' : sum_abs_diff,
                                'slope' : slope,
                                'abs_energy' : abs_energy,
                                'pk_pk_distance' : pk_pk_distance,
                                'entropy' : entropy,
                                'max' : np.max,
                                'std' : np.std,
                                'mean' : np.mean
                                }

dict_time_domain_arguments = {'autocorr' : 'signal',
                                'zero_crossing' : 'signal',
                                'mean_abs_diff' : 'signal',
                                'distance' : 'signal',
                                'sum_abs_diff' : 'signal',
                                'slope' : 'signal',
                                'abs_energy' : 'signal',
                                'pk_pk_distance' : 'signal',
                                'entropy' : 'signal',
                                'max' : 'signal',
                                'std' : 'signal',
                                'mean' : 'signal'
                                }

#######################################################################################################################
# Spectral features functions
#######################################################################################################################


def spectral_distance(signal, fs):
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

def fundamental_frequency(signal, fs):
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

def max_power_spectrum(signal, fs):
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

def max_frequency(signal, fs):
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

def median_frequency(signal, fs):
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

def spectral_centroid(signal, fs):
    """Barycenter of the spectrum.

    Description and formula in Article:
    The Timbre Toolbox: Extracting audio descriptors from musicalsignals
    Authors Peeters G., Giordano B., Misdariis P., McAdams S.

    Feature computational cost: 2

    Parameters
    ----------
    signal : nd-array
        Signal from which spectral centroid is computed
    fs: int
        Sampling frequency

    Returns
    -------
    float
        Centroid

    """
    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))

def spectral_decrease(signal, fs):
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

def spectral_kurtosis(signal, fs):
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

def spectral_slope(signal, fs):
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

def spectral_variation(signal, fs):
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

def spectral_roll_off(signal, fs):
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

def spectral_roll_on(signal, fs):
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

def human_range_energy(signal, fs):
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

def power_bandwidth(signal, fs):
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

def spectral_entropy(signal, fs):
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

def wavelet_entropy(signal, fs,function=ricker, widths=np.arange(1, 10)):
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

def mfcc(signal, fs, pre_emphasis=0.97, nfft=512, nfilt=40, num_ceps=12, cep_lifter=22):
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
    nd-array
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


''' 
retourne un tuple avec nfreq valeurs, nfreq c'est la granularité du découpage du spectre
Mettre pour nfreq la valeur de la fréquence d'acquisition divisée par 2
'''

def fft_mean_coeff(signal, fs, nfreq=10):
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
        The number of frequencies

    Returns
    -------
    nd-array
        The mean value of each spectrogram frequency

    """
    if nfreq > len(signal) // 2 + 1:
        nfreq = len(signal) // 2 + 1

    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)

    return tuple(fmag_mean)

''' 
return n_coeff
https://tsfel.readthedocs.io/en/latest/_modules/tsfel/feature_extraction/features_utils.html

'''

def lpcc(signal, n_coeff=12):
    """Computes the linear prediction cepstral coefficients.

    Implementation details and description in:
    http://www.practicalcryptography.com/miscellaneous/machine-learning/tutorial-cepstrum-and-lpccs/

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from linear prediction cepstral coefficients are computed
    n_coeff : int
        Number of coefficients

    Returns
    -------
    nd-array
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

def wavelet_abs_mean(signal, function=ricker, widths=np.arange(1, 10)):
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


def wavelet_std(signal, function= ricker, widths=np.arange(1, 10)):
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

def wavelet_energy(signal, function=ricker, widths=np.arange(1, 10)):
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

#######################################################################################################################
# A dict to retrieve easily all features
#######################################################################################################################


dict_frequential_mvt_features = {'spectral_distance' : spectral_distance,
                                    'wavelet_entropy' : wavelet_entropy,
                                    'spectral_entropy' : spectral_entropy,
                                    'power_bandwidth' : power_bandwidth,
                                    'human_range_energy' : human_range_energy,
                                    'spectral_roll_on' : spectral_roll_on,
                                    'spectral_roll_off' : spectral_roll_off,
                                    'spectral_variation' : spectral_variation,
                                    'spectral_slope' : spectral_slope,
                                    'spectral_kurtosis' : spectral_kurtosis,
                                    'spectral_decrease' : spectral_decrease,
                                    'spectral_centroid' : spectral_centroid,
                                    'median_frequency' : median_frequency,
                                    'max_power_spectrum' : max_power_spectrum,
                                    'max_frequency' : max_frequency,
                                    'fundamental_frequency' : fundamental_frequency,
                                    'spectral_distance' : spectral_distance
}

#######################################################################################################################
# For each features, the numbers of spectral features depends on the signal and on the value of parameters
#######################################################################################################################

dict_frequential_mvt_multiple_features = {'mfcc' : mfcc,
                                          'fft_mean_coeff' : fft_mean_coeff,
                                          'lpcc' : lpcc,
                                          'wavelet_abs_mean' : wavelet_abs_mean,
                                          'wavelet_std' : wavelet_std,
                                          'wavelet_energy' : wavelet_energy } 



