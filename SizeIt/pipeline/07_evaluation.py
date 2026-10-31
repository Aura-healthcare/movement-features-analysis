"""Étape 7 : évaluation du modèle personnalisé (recette gelée, sizeit/modele.py).

Pour chaque patient et chaque bloc temporel, le modèle est entraîné sur les 4 autres blocs du patient et prédit le
bloc laissé de côté. Les probabilités sont lissées sur 10 s, comparées au seuil, et les alarmes sont comptées
(règles standard et stricte). Sorties dans 07_resultats/ : predictions.parquet, rapports types (PDF + CSV) pour la
cible principale et pour toutes les crises.
"""
import os
import pandas as pd
from _commun import arguments, en_parallele, log
from sizeit import config as C, donnees as D, decoupages as DC, modele as M, rapport as RP

a = arguments(__doc__, seuil=dict(type=float, default=M.SEUIL_PRINCIPAL, help='seuil d’alarme (défaut : 0,818)'))
os.makedirs(C.RESULTATS_DIR, exist_ok=True)
CFG = DC.load(os.path.join(C.DECOUPAGES_RES, 'decoupage_blocs_temporels.json'))
pids = [p for p in CFG['patients'] if not a.patients or p in a.patients]


def predire(pid):
    df = D.load_patient(pid)
    fo = DC.assign_folds(df, CFG['patients'][pid])
    proba, _ = M.predictions_personnalisees(df, fo)
    return pd.DataFrame({'patient_id': pid, 'run_id': df['run_id'].astype(str).values, 'elapsed_s': df['elapsed_s'].values,
                         'proba': proba})


preds = pd.concat(en_parallele(predire, pids, a.jobs, 'modèles personnalisés'), ignore_index=True)
preds.to_parquet(os.path.join(C.RESULTATS_DIR, 'predictions.parquet'), index=False)
for nom, scope in (('cible', ['Hypermotrices et convulsives']), ('toutes_les_crises', None)):
    r = RP.generer(preds, os.path.join(C.RESULTATS_DIR, f'rapport_{nom}'),
                   meta={'nom': f'Modèle personnalisé v{M.VERSION} — {nom.replace("_", " ")}', 'patients': len(pids)},
                   postprocess=M.POST_TRAITEMENT, seuil=a.seuil, in_scope=scope, n_boot=1000)
    s = r['resume']['standard']
    log.info(f"{nom} : {s['n_detectees']}/{s['n_crises']} crises détectées ({s['sensibilite']:.0%}), "
             f"{s['fa_par_24h']:.2f} fausses alarmes / 24 h -> {C.RESULTATS_DIR}/rapport_{nom}/rapport.pdf")
