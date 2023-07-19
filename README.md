# Movement-features-analysis
A repository to compute the movement features related to epilepsy seizures
Test

**Movement-features-analysis** is a Python module for computing the features based on the movement of the patient

The developement of this librairy started in /date/ as part of [Aura Healthcare](https://www.aura.healthcare) project.

**Full documentation** : https://github.com/Aura-healthcare/movement-features-analysis

**Website** : https://www.aura.healthcare

**Github** : https://github.com/Aura-healthcare

<br/>

## List of features
<br/>

###  Temporal features 
<br/>

| **Name of the feature** | **Description** | 
| :--- | :--- | 
| Autocorr | Computes auto-correlation of the signal in order not to have <br/>non-valid data |
| Zero_crossing | - |
| Mean_abs_diff | Computes mean of absolute difference of the signal |
| Distance | Computes signal traveled distance |
| Sum_aps_diff | Computes sum of absolute difference of the signal | 
| Slope | Computes the slope of the signal |
| Abs_energy | Compute the absolute energy of the signal |
| Pk_pk_distance | Distance between the max and the min of the signal (peak to peak) |
| Entropy | Computes the entropy of the signal using the Shannon entropy |
