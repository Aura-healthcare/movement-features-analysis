# SizeIt — détection personnalisée des crises d'épilepsie (HRV + accéléromètre)

Code complet pour reproduire, à partir des données brutes de **SeizeIT2**, le détecteur de crises **personnalisé
patient par patient** développé par Aura Healthcare : préparation des données, calcul des features, fusion avec les
features cardiaques, identification des crises, découpage apprentissage / test, modèle, logique d'alarme, évaluation,
modèles de production et visualisation interactive.

> **Statut** : outil de recherche, **non validé cliniquement**. Voir [Résultats et limites](#résultats-et-limites).

---

## 1. Principe en une minute

- Un capteur porté enregistre **l'ECG** (d'où l'on tire la variabilité de la fréquence cardiaque, HRV) et
  **l'accélération** (capteur placé sur le casque EEG, « EEG SD », 25 Hz).
- Chaque seconde, **59 features** sont calculées : 27 HRV, 28 décrivant la norme de l'accélération, 4 bandes
  d'énergie du mouvement (0–5, 5–8, 8–12, 12–25 Hz).
- Pour **chaque patient**, un gradient boosting (50 arbres, profondeur 4) est appris sur ses propres crises.
- La probabilité de crise est **lissée sur 10 s** ; une alarme est émise quand elle dépasse un **seuil global**
  (0,818 pour viser ≤ 2 fausses alarmes / 24 h). Deux alarmes à moins de 60 s sont fusionnées.
- Évaluation honnête : l'enregistrement de chaque patient est coupé en **5 blocs temporels** ; le modèle est appris
  sur 4 blocs et testé sur le 5e, jamais vu (les crises à moins de 10 min d'écart ne sont jamais séparées).

---

## 2. Données nécessaires

| Donnée | Source | Variable d'environnement |
|---|---|---|
| **SeizeIT2** brut (format BIDS) : EDF `*_mov.edf` (accélération) et `*_events.tsv` (crises annotées) | OpenNeuro, jeu **ds005873** : <https://openneuro.org/datasets/ds005873> | `SIZEIT_RAW_DIR` = dossier `ds005873` (contient `sub-001/`, `sub-002/`…) |
| **Features HRV** par run (calculées par le pipeline cardiaque d'Aura à partir des ECG) | à demander à l'équipe Aura Healthcare | `SIZEIT_HRV_DIR` = dossier `feats-hrv` |
| Dossier de travail (créé par le pipeline) | — | `SIZEIT_WORK_DIR` (défaut : `SizeIt/donnees`) |

### Format attendu des features HRV

Un fichier par run :
`<SIZEIT_HRV_DIR>/<run>_ecg/fast/features/feats_<run>_ecg_fast.csv`, où `<run>` vaut par exemple
`sub-001_ses-01_task-szMonitoring_run-01`. Une ligne par seconde, avec les colonnes :

`interval_index, interval_start_time` (en ms depuis le début du run), puis les 27 features HRV
`mean_nni, sdnn, sdsd, nni_50, pnni_50, nni_20, pnni_20, rmssd, median_nni, range_nni, cvsd, cvnni, mean_hr, max_hr,
min_hr, std_hr, lf, hf, vlf, lf_hf_ratio, csi, cvi, Modified_csi, sampen, sd1, sd2, ratio_sd2_sd1`
(noms de la librairie [hrv-analysis](https://github.com/Aura-healthcare/hrv-analysis)), et `timestamp`.

> Les features HRV ne sont **pas** recalculées par ce dépôt : elles proviennent du pipeline cardiaque d'Aura
> (détection des battements « fast », puis HRV seconde par seconde). Tout le reste est calculé ici depuis les EDF.

Seuls les sous-dossiers `mov/` (EDF d'accélération, ≈ 26 Go pour les 125 patients) et `eeg/*_events.tsv`
(annotations) de SeizeIT2 sont utilisés. Volumes mesurés pour les 125 patients : features HRV ≈ 29 Go ; dossier de
travail ≈ 260 Go au maximum, dont ≈ 70 Go pour `01_csv_acc` et ≈ 170 Go pour `02_csv_acc_renormalise`. Ces deux
dossiers peuvent être supprimés une fois l'étape 3 terminée, sauf si vous voulez afficher l'accélération brute dans
la visualisation, qui lit `02_csv_acc_renormalise`.

---

## 3. Installation

```bash
git clone https://github.com/Aura-healthcare/movement-features-analysis.git
cd movement-features-analysis/SizeIt
python -m venv .venv && source .venv/bin/activate      # ou : conda create -n sizeit python=3.13
pip install -r requirements.txt

export SIZEIT_RAW_DIR=/chemin/vers/ds005873
export SIZEIT_HRV_DIR=/chemin/vers/feats-hrv
export SIZEIT_WORK_DIR=/chemin/vers/dossier_de_travail     # facultatif
```

Vérification rapide (règle de comptage des alarmes, 18 cas, quelques secondes) :

```bash
python tests/test_comptage.py
```

---

## 4. Lancer le pipeline

Tout d'un coup, sur les 32 patients utilisés pour le développement :

```bash
bash pipeline/lancer_tout.sh --liste patients_developpement.txt --jobs 16
```

Ou étape par étape. Toutes les étapes acceptent `--patients sub-001 sub-002 …`, `--liste fichier.txt` et
`--jobs N`, et **reprennent là où elles se sont arrêtées** : les fichiers déjà produits ne sont pas recalculés.

| Étape | Script | Ce qu'elle fait | Sortie (dans `SIZEIT_WORK_DIR`) |
|---|---|---|---|
| 1 | `pipeline/01_edf_vers_csv.py` | extrait les canaux d'accélération des EDF `*_mov.edf` | `01_csv_acc/` |
| 2 | `pipeline/02_renormalisation.py` | ramène la norme moyenne de l'accélération de chaque fichier à 1 (≈ 1 g) ; ignore les runs sans accéléromètre EEG SD | `02_csv_acc_renormalise/` |
| 3 | `pipeline/03_features_acc.py` | features de l'accélération : fenêtres de 10 s, une ligne toutes les 5 s (**étape la plus longue**) | `03_features_acc/` |
| 4 | `pipeline/04_fusion_grille.py` | fusionne HRV + accélération + annotations en une grille à 1 s par run (secondes où HRV **et** accélération existent) ; `label` = 1 pendant une crise, `seizure_type` = type ILAE | `04_grille/` |
| 5 | `pipeline/05_preparation_patients.py` | un fichier par patient, runs dans l'ordre, **identifiant de chaque crise** (`seizure_id` = `<patient>_<n>`) | `05_patients/<patient>.parquet` |
| 6 | `pipeline/06_decoupage.py` | découpage apprentissage / test : 5 blocs temporels par patient (patients avec moins de 3 crises écartés) | `06_decoupages/decoupage_blocs_temporels.json` |
| 7 | `pipeline/07_evaluation.py` | modèle personnalisé évalué bloc par bloc, alarmes, rapports PDF + CSV | `07_resultats/` |
| 8 | `pipeline/08_entrainer_modeles.py` | modèle de production de chaque patient (toutes ses données) | `08_modeles/<patient>.joblib` + `.json` |
| 9 | `pipeline/09_surveiller.py` | applique le modèle d'un patient à un nouvel enregistrement → alarmes | CSV d'alarmes + probabilités |

Durée indicative : environ 2 minutes pour 3 patients (≈ 130 h d'enregistrement) sur 20 processus, dont l'essentiel à
l'étape 3.

Exemple pour surveiller un nouvel enregistrement du patient sub-001 (préparé par les étapes 1 à 5) :

```bash
python pipeline/09_surveiller.py --patient sub-001 --donnees nouveau/sub-001.parquet --sortie alarmes_sub-001 --fa 2
```

### Features optionnelles (non utilisées par le modèle v1)

À lancer après l'étape 5 ; elles ajoutent des colonnes aux fichiers de `05_patients/` :

- `pipeline/optionnel/features_cl_fourier.py` : puissance du mouvement dans les bandes 2–7 Hz et 7–12,5 Hz (Welch) ;
- `pipeline/optionnel/features_contexte.py` : écarts au contexte des 30 dernières minutes et 2 dernières heures
  (24 candidats ; les 10 plus liés aux crises sont retenus, la sélection dépend donc des patients traités).

Ces features ont été évaluées sans gain démontré ; le modèle v1 ne les utilise pas (il sélectionne ses 59 colonnes
par leur nom).

---

## 5. Visualiser les crises et les prédictions

Application web locale (Dash / Plotly) : choix du patient, du type de crise, de la crise ; accélération brute,
fréquence cardiaque, agitation, probabilité, probabilité lissée, seuil et alarmes ; vue de tout l'enregistrement et
tableau des crises détectées ou manquées.

```bash
python visualisation/app.py                    # puis ouvrir http://127.0.0.1:8050
python visualisation/app.py --port 8060 --host 0.0.0.0
```

Prérequis : étapes 2 (accélération brute), 5 (données patients) et 7 (`07_resultats/predictions.parquet`). Tout
fichier de prédictions placé sous `07_resultats/` (colonnes `patient_id, run_id, elapsed_s, proba`) apparaît dans le
menu « Modèle ». Depuis un poste distant : `ssh -L 8050:localhost:8050 <serveur>`.

---

## 6. Organisation du code

```
SizeIt/
├── README.md
├── requirements.txt
├── patients_developpement.txt     les 32 patients SeizeIT2 du développement
├── sizeit/                        bibliothèque
│   ├── config.py                  chemins, jeux de features, groupes de crises, règles de comptage
│   ├── preparation.py             EDF -> CSV, renormalisation, fusion HRV + ACC, seizure_id
│   ├── features_acc.py            features de l'accélération (fenêtre 10 s, pas 5 s)
│   ├── features_functions.py      fonctions de calcul des features (temporelles, spectrales, Fourier)
│   ├── donnees.py                 chargement d'un patient dans l'ordre chronologique, segments continus, crises
│   ├── decoupages.py              blocs temporels, sous-échantillonnage
│   ├── modele.py                  recette du modèle v1 (gelée), seuils
│   ├── scoring.py                 lissage, alarmes, comptage des détections et fausses alarmes, IC
│   ├── alarme.py                  outils d'étude de la logique d'alarme (seuils, période réfractaire)
│   └── rapport.py                 rapport type (PDF + CSV)
├── pipeline/                      étapes 01 à 09, lancer_tout.sh, optionnel/
├── visualisation/                 application Dash
└── tests/test_comptage.py
```

### Détails de méthode utiles

- **Ordre des données** : les runs sont triés par **numéro** (run-9 < run-10 < run-100). Les fichiers d'une heure
  consécutifs (56 à 62 min suivis du run de numéro suivant) sont **recollés** en segments continus, sur lesquels
  sont calculés le lissage et les alarmes.
- **Groupes de crises** (`config.SEIZURE_GROUPS`) : *hypermotrices et convulsives* (hyperkinétiques, focales
  évoluant vers tonico-cloniques bilatérales : **cible principale**), *motrices focales discrètes*, *non
  motrices*, *sémiologie non précisée*. Le modèle personnalisé apprend toutes les crises du patient.
- **Comptage** (`scoring.py`) : règle *standard* — une crise est détectée si une alarme chevauche
  [début − 30 s ; fin + 60 s] ; règle *stricte* — alarme d'au moins 10 s pendant la crise. Fausse alarme = alarme
  hors de toute crise ; étiquetée post-ictale si elle survient dans l'heure qui suit une crise.
- **Apprentissage** : toutes les secondes de crise, et juste assez de secondes sans crise pour que les crises
  représentent 2 % de l'échantillon.
- **Seuils gelés** (`modele.SEUILS`) : 0,924 / 0,818 / 0,777 / 0,622 pour viser ≤ 1 / 2 / 3 / 5 fausses alarmes
  par 24 h.

---

## 7. Résultats et limites

Sur les 31 patients de développement (crises de la cible, règle standard, seuil 0,818, blocs temporels) :
**≈ 58 % des crises hypermotrices et convulsives détectées** (69 % des crises hyperkinétiques conscientes, 68 % des
crises focales évoluant vers une crise tonico-clonique bilatérale), **≈ 1,5 fausse alarme / 24 h**, délai médian
d'alerte ≈ 30 s.

**Limite majeure** : évaluée une seule fois, avec un protocole figé à l'avance, sur 31 autres patients de
SeizeIT2 jamais utilisés, la même recette ne détecte que **18 % [8 ; 27]** des crises de la cible (1,8 fausse
alarme / 24 h). Cause identifiée : les crises de ces patients sont **très courtes** (médiane 14 s contre 97 s dans
la cohorte de développement) ; le modèle v1 ne détecte pas les crises de moins de 20 s. Les performances ci-dessus
ne valent donc que pour des crises motrices longues et doivent être confirmées sur de nouvelles données avant tout
usage clinique.

---

## 8. Licence et citation

Code : voir `LICENSE` à la racine du dépôt. Données SeizeIT2 : licence et citation indiquées sur la page OpenNeuro
ds005873.
