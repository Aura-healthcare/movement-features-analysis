"""Configuration commune : chemins, colonnes, jeux de features, catégories de crises, règles de comptage.
Tout le code importe ce module : un réglage ne se change qu'ici.

Chemins (variables d'environnement, voir README) :
- SIZEIT_RAW_DIR   : racine BIDS de SeizeIT2 (ds005873), contient sub-xxx/ses-01/{mov,eeg}/... ;
- SIZEIT_HRV_DIR   : dossier des features HRV fournies (un sous-dossier <run>_ecg/fast/features/ par run) ;
- SIZEIT_WORK_DIR  : dossier de travail où le pipeline écrit tout (par défaut SizeIt/donnees).
"""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # dossier SizeIt/
RAW_DIR = os.environ.get('SIZEIT_RAW_DIR', os.path.join(ROOT_DIR, 'donnees', 'brutes', 'ds005873'))
HRV_DIR = os.environ.get('SIZEIT_HRV_DIR', os.path.join(ROOT_DIR, 'donnees', 'brutes', 'feats-hrv'))
WORK_DIR = os.environ.get('SIZEIT_WORK_DIR', os.path.join(ROOT_DIR, 'donnees'))

CSV_ACC_DIR = os.path.join(WORK_DIR, '01_csv_acc')                 # accélération brute convertie depuis les EDF
CSV_ACC_RENORM_DIR = os.path.join(WORK_DIR, '02_csv_acc_renormalise')
FEATURES_ACC_DIR = os.path.join(WORK_DIR, '03_features_acc')
GRILLE_DIR = os.path.join(WORK_DIR, '04_grille')                   # HRV + ACC + annotations, une grille à 1 s par run
DATA_DIR = os.environ.get('SIZEIT_PATIENTS_DIR', os.path.join(WORK_DIR, '05_patients'))   # un parquet par patient
DECOUPAGES_RES = os.path.join(WORK_DIR, '06_decoupages')
RESULTATS_DIR = os.path.join(WORK_DIR, '07_resultats')
MODELES_DIR = os.path.join(WORK_DIR, '08_modeles')

META_COLS = ['elapsed_s', 'hrv_interval_index', 'hrv_interval_start_time', 'hrv_available', 'acc_available',
             'label', 'seizure_type', 'seizure_id', 'timestamp', 'patient_id', 'run_id']

# ---------------------------------------------------------------- jeux de features
def feature_set(columns, name='hrv_norme_custom'):
    """Renvoie la liste des features d'un jeu nommé, dans l'ordre des colonnes.

    - 'hrv_norme_custom' (jeu de référence, 59 features) : HRV + norme de l'accélération + bandes 'custom'
      (indépendant de l'orientation du capteur ; pas de moyennes d'axes, qui codent la posture) ;
    - 'tout' : les 155 features d'origine ;
    - 'hrv_norme_custom_ecarts' : jeu de référence + les 10 écarts au contexte `dev_*` ;
    - 'hrv_norme_custom_cl' : jeu de référence + les puissances spectrales CL_Fourier_2_7 et CL_Fourier_7_125
      (calculées sur l'accélération brute, notebooks/preprocessing/preprocessing_CL_fourier.ipynb).
    Les colonnes ajoutées après coup (dev_*, CL_*) ne sont jamais incluses implicitement.
    """
    base = [c for c in columns if c not in META_COLS and not c.startswith(('dev_', 'CL_'))]
    hrv_norme_custom = [c for c in base if c.startswith('hrv_') or
                        (c.startswith('acc_') and ('norme_1' in c or 'custom' in c))]
    sets = {
        'hrv_norme_custom': hrv_norme_custom,
        'tout': base,
        'hrv_norme_custom_ecarts': hrv_norme_custom + [c for c in columns if c.startswith('dev_')],
        'hrv_norme_custom_cl': hrv_norme_custom + [c for c in columns if c.startswith('CL_Fourier')],
    }
    return sets[name]


def feature_family(c):
    if c.startswith('CL_'):
        return 'CL Fourier'
    if c.startswith('dev_'):
        return 'Écart au contexte'
    if c.startswith('hrv_'):
        return 'HRV'
    if 'norme_1' in c:
        return 'ACC norme'
    if 'custom' in c:
        return 'ACC bandes custom'
    if '_F_' in c:
        return 'ACC bandes 1 Hz'
    return 'ACC axes x/y/z'


# ---------------------------------------------------------------- types de crises
# groupes d'analyse (composante motrice)
SEIZURE_GROUPS = {
    'Hypermotrices et convulsives': ['a_m_hyperkinetic', 'ia_m_hyperkinetic', 'ua_m_hyperkinetic', 'f2b'],
    'Motrices focales discrètes': ['a_m_automatisms', 'ia_m_automatisms', 'ia_m_tonic', 'ua_m_tonic'],
    'Non motrices': ['a_nm', 'ia_nm', 'ua_nm', 'a_nm_behavior', 'ua_nm_behavior', 'ia_nm_behavior', 'sz_uo_nm'],
    'Sémiologie non précisée': ['a_um', 'ua_um', 'ia', 'ia_um', 'ua_m'],
}
# types absents des 32 patients de développement, rencontrés dans la lockbox (08_gel_modele), classés AVANT
# l'évaluation : ua_m_tonic (tonique) ; ia_nm_behavior et sz_uo_nm (début inconnu, non moteur) ; ia_um ; ua_m (moteur
# sans sous-type : on ne peut pas savoir s'il est hypermoteur, donc « non précisée »)
# catégories opérationnelles (ce que le produit promet)
CATEGORIES = {
    'Hypermotrices et convulsives': 'cible principale',
    'Motrices focales discrètes': 'cible exploratoire',
    'Non motrices': 'hors champ',
    'Sémiologie non précisée': 'à revoir (annotation)',
}
TYPE_TO_GROUP = {t: g for g, ts in SEIZURE_GROUPS.items() for t in ts}


def short_type(seizure_type):
    return None if seizure_type is None or seizure_type != seizure_type else str(seizure_type).replace('sz_foc_', '')


# ---------------------------------------------------------------- données
HOUR_FILE_MIN_S, HOUR_FILE_MAX_S = 56 * 60, 62 * 60   # run « fichier d'une heure » recollé au run suivant
SEIZURE_GROUP_GAP_S = 600                              # crises à < 10 min : un seul groupe (jamais séparées)
MIN_CRISES_MODELE = 3              # moins de 3 crises dans les données : pas de modèle personnalisé évaluable
EXCLUDED_FOR_MODELS = []           # exclusions manuelles éventuelles

# ---------------------------------------------------------------- règles de comptage des alarmes
SCORING = {
    'standard': {'pre_s': 30, 'post_s': 60, 'min_alarm_s': 1},     # convention « tout chevauchement »
    'strict': {'pre_s': 0, 'post_s': 0, 'min_alarm_s': 10},        # alarme d'au moins 10 s pendant la crise
}
ALARM_MERGE_S = 60       # deux alarmes séparées de moins de 60 s = une seule
POST_ICTAL_S = 3600      # une fausse alarme dans l'heure qui suit une crise est étiquetée « post-ictale »
