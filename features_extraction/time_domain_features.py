"""Time domain features functions"""

import numpy as np

def autocorr(signal: np.ndarray) -> float:
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

def zero_crossing(signal: np.ndarray) -> float:
    """Computes zero crossing of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which zero crossing is computed

    Returns
    -------
    float
        Zero crossing of the signal
    """
    signal = signal - np.mean(signal)
    return len(np.where(np.diff(np.sign(signal)))[0])

def mean_abs_diff(signal: np.ndarray) -> float:
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

def distance(signal: np.ndarray) -> float:
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

def sum_abs_diff(signal: np.ndarray) -> float:
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

def slope(signal: np.ndarray) -> float:
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

def abs_energy(signal: np.ndarray) -> float:
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

def pk_pk_distance(signal: np.ndarray) -> float:
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
    signal = np.round(signal,2) # round because values are continuous
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

    return(np.mean(signal)) # TODO: check if this is correct, never called