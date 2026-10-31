"""Étape 6 : découpage de chaque patient en 5 blocs temporels (les crises à moins de 10 min ne sont jamais séparées)
-> 06_decoupages/decoupage_blocs_temporels.json. Les patients avec moins de C.MIN_CRISES_MODELE crises sont écartés."""
import json
import os
import pandas as pd
from _commun import arguments, log
from sizeit import config as C, donnees as D, decoupages as DC

a = arguments(__doc__)
os.makedirs(C.DECOUPAGES_RES, exist_ok=True)
pids = [p for p in D.list_patients(exclude=C.EXCLUDED_FOR_MODELS) if not a.patients or p in a.patients]
blocs, resume, ecartes = {}, [], []
for pid in pids:
    df = D.load_patient(pid, columns=[])
    sz = D.seizures(df)
    if len(sz) < C.MIN_CRISES_MODELE:
        ecartes.append(pid)
        continue
    groupes = D.seizure_groups(sz)
    fo, extra = DC.blocs_temporels(df, groupes)
    blocs[pid] = DC.describe(df, fo, sz, {**extra, 'groupes': groupes})
    resume.append({'patient_id': pid, 'heures': len(df) / 3600, 'crises': len(sz), 'groupes_de_crises': len(groupes),
                   'blocs_avec_crise': sum(bool(f['crises_test']) for f in blocs[pid]['folds'])})
    log.info(f'{pid} : {len(sz)} crises, {len(df) / 3600:.0f} h')
DC.save(blocs, {'version': 2, 'ordre': 'runs triés par numéro', 'exclus': ecartes, 'ecart_groupe_s': C.SEIZURE_GROUP_GAP_S,
                'decoupage': 'blocs_temporels', 'n_folds': DC.N_FOLDS},
        os.path.join(C.DECOUPAGES_RES, 'decoupage_blocs_temporels.json'))
pd.DataFrame(resume).to_csv(os.path.join(C.DECOUPAGES_RES, 'resume_decoupage.csv'), index=False)
log.info(f'{len(blocs)} patients découpés ; écartés (moins de {C.MIN_CRISES_MODELE} crises) : {ecartes}')
