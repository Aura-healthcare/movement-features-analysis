# Movement-features-analysis 

A repository to compute the movement features related to epilepsy seizures <br />
The library to execute the code are in the requirements.txt file <br />
Most of the spectral features are extracted from [TSFEL library](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html) 

## Getting started

**Prerequisites**: start conda environment and install dependencies

```bash
conda create --name myenv python=3.8
conda activate myenv
pip install -r requirements.txt
```

Run the library 
```bash
cd features_extraction
python features_computation.py 10 "./../datas/datas_test_feature_extraction.csv" "all"
```

**Movement-features-analysis** is a Python module for computing the features based on the movement of the patient

The developement of this librairy started in /date/ as part of [Aura Healthcare](https://www.aura.healthcare) project.

**Full documentation** : https://github.com/Aura-healthcare/movement-features-analysis

**Website** : https://www.aura.healthcare

**Github** : https://github.com/Aura-healthcare  

## List of features  

###  Temporal features  

| **Name of the feature** | **Description** |
|------------------------|-----------------|
| Autocorr | Computes auto-correlation of the signal |
| Zero_crossing | Compute the number of time the signal cross zero |
| Mean_abs_diff | Computes mean of absolute difference of the signal |
| Distance | Computes signal traveled distance |
| Sum_aps_diff | Computes sum of absolute difference of the signal |
| Slope | Computes the slope of the signal |
| Abs_energy | Compute the absolute energy of the signal |
| Pk_pk_distance | Distance between the max and the min of the signal (peak to peak) |
| Entropy | Computes the entropy of the signal using the Shannon entropy |

###  Frequency features  

CWT = continuous wavelet transform

| **Name of the feature** | **Description** |
|------------------------|-----------------|
| spectral_distance | Computes the signal spectral distance |
| wavelet_entropy | Computes CWT entropy of the signal |
| spectral_entropy | Computes the signal spectral entropy |
| power_bandwidth | Computes power spectrum density bandwidth of the signa |
| human_range_energy | Computes the human range energy ratio |
| spectral_roll_on | Computes the spectral roll-on of the signal |
| spectral_roll_off | Computes the spectral roll-off of the signal |
| spectral_variation | Computes the amount of variation of the spectrum along time |
| spectral_slope | Computes the spectral slope |
| spectral_kurtosis | Computes the spectral_kurtosis |
| spectral_decrease | Represents the amount of decreasing of the spectra amplitude |
| spectral_centroid | Computes the barycenter of the spectrum |
| median_frequency | Computes median frequency of the signal |
| max_power_spectrum | Computes the max of the dft |
| max_frequency | Computes the frequency of the max power |
| fundamental_frequency | Computes the fundamental_frequency (ie first harmonic in music) |
| power in band | Computes the power of the signal in the selected band of frequency |
| relative power in band | Computes the relativ power (eg divide by the Norm L2 of the signal) of the signal in the selected band of frequency |


**Few definitions :**  
https://analyticsindiamag.com/a-tutorial-on-spectral-feature-extraction-for-audio-analytics/

**spectral roll-off :** The spectral roll-off point is the fraction of bins in the power spectrum at which 85% of the power is at lower frequencies 

**spectral Centroid :** A spectral centroid is the location of the centre of mass of the spectrum.

**spectral_kurtosis :** Measures the flatness of a distribution around its mean value. 

**mfcc :** https://en.wikipedia.org/wiki/Cepstrum  
In Fourier analysis, the cepstrum is the result of computing the inverse Fourier transform (IFT) of the logarithm of the estimated signal spectrum. The method is a tool for investigating periodic structures in frequency spectra. The 
The cepstrum can be seen as information about the rate of change in the different spectrum bands. It was originally invented for characterizing the seismic echoes resulting from earthquakes and bomb explosions.

**lpcc :** LPCC is a technique that combines LP and cepstral analysis by taking the inverse Fourier transform of the log magnitude of the LPC spectrum for improved accuracy and robustness of the voice features extracted. 

**wavelet_abs_mean, wavelet_energy and wavelet_std :** The  mean, std and energy applyed to wavelet.


