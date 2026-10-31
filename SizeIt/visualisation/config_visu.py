"""Chemins et réglages de l'outil de visualisation (les chemins viennent de sizeit/config.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # dossier SizeIt/
from sizeit import config as C  # noqa: E402

RESULTATS_DIR = C.RESULTATS_DIR             # prédictions à afficher : tout parquet (patient_id, run_id, elapsed_s, proba)
RAW_ACC_DIR = C.CSV_ACC_RENORM_DIR          # accélération brute renormalisée (étape 2 du pipeline)
DATA_DIR = C.DATA_DIR

RAW_FS = 25                       # fréquence d'échantillonnage de l'accélération brute (Hz)
RAW_COLS = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z', 'norme_1']
HR_COL = 'hrv_mean_hr'
ACT_COL = 'acc_EEG_SD_ACC_norme_1_std'
DEFAULT_PORT = 8050

# features supplémentaires affichables (rangée « Features », échelle log) ; seules celles présentes sont proposées
EXTRA_FEATURES = ['CL_Fourier_2_7', 'CL_Fourier_7_125', 'acc_EEG_SD_ACC_F_custom_0_5', 'acc_EEG_SD_ACC_F_custom_5_8',
                  'acc_EEG_SD_ACC_F_custom_8_12', 'acc_EEG_SD_ACC_norme_1_abs_energy']
DEFAULT_FEATURES = ['acc_EEG_SD_ACC_F_custom_5_8', 'acc_EEG_SD_ACC_F_custom_8_12']
