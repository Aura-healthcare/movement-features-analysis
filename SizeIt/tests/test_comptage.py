"""Tests de la règle de comptage des alarmes (sizeit/scoring.py) : 18 cas.
Lancer : python tests/test_comptage.py   (ou pytest tests/)"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sizeit import scoring as S  # noqa: E402

resultats = []


def check(nom, cond):
    resultats.append({'test': nom, 'ok': bool(cond)})
    print(('OK   ' if cond else 'ÉCHEC') + ' ' + nom)


def crises(*items, groupe='Hypermotrices et convulsives'):
    # items : (start, end) en secondes, segment 0
    return pd.DataFrame([{'seizure_id': f's{i}', 'start': a, 'end': b, 'segment': 0, 'groupe': groupe, 'type': 'x',
                          'duree_s': b - a} for i, (a, b) in enumerate(items)])



def test_comptage():
    N = 3000
    seg = np.zeros(N, int)
    segb = {0: (0, N)}

    # 1. alarme pendant la crise -> détectée, aucune fausse alarme
    p = np.zeros(N); p[1010:1030] = 0.9
    ev = S.alarms(p, seg, 0.5)
    s, a = S.score_patient(ev, crises((1000, 1060)), segb, 'standard')
    check('alarme pendant la crise : détectée', s['detectee'].iat[0] and (a['statut'] == 'fausse alarme').sum() == 0)
    check('latence = début alarme - début crise = 10 s', s['latence_s'].iat[0] == 10)

    # 2. alarme 50 s après la fin : détectée en standard (tolérance 60 s), pas en strict
    p = np.zeros(N); p[1110:1130] = 0.9
    ev = S.alarms(p, seg, 0.5)
    s_std, _ = S.score_patient(ev, crises((1000, 1060)), segb, 'standard')
    s_str, a_str = S.score_patient(ev, crises((1000, 1060)), segb, 'strict')
    check('alarme 50 s après la fin : détectée (standard)', s_std['detectee'].iat[0])
    check('alarme 50 s après la fin : manquée (strict) et comptée fausse alarme',
          (not s_str['detectee'].iat[0]) and (a_str['statut'] == 'fausse alarme').sum() == 1)

    # 3. alarme loin de toute crise -> fausse alarme, post-ictale si < 1 h après une crise
    p = np.zeros(N); p[2000:2005] = 0.9
    ev = S.alarms(p, seg, 0.5)
    s, a = S.score_patient(ev, crises((1000, 1060)), segb, 'standard')
    check('alarme isolée : fausse alarme', (a['statut'] == 'fausse alarme').sum() == 1 and not s['detectee'].iat[0])
    check('fausse alarme < 1 h après une crise : post-ictale', bool(a['post_ictale'].iat[0]))

    # 4. deux pics séparés de 30 s -> une seule alarme (fusion < 60 s) ; séparés de 120 s -> deux alarmes
    p = np.zeros(N); p[500:505] = 0.9; p[535:540] = 0.9; p[700:705] = 0.9
    ev = S.alarms(p, seg, 0.5)
    check('fusion des alarmes séparées de moins de 60 s', len(ev) == 2 and ev[0, 0] == 500 and ev[0, 1] == 540)

    # 5. règle stricte : alarme de 5 s pendant la crise -> non comptée (minimum 10 s), alarme de 12 s -> détectée
    p = np.zeros(N); p[1010:1015] = 0.9
    ev = S.alarms(p, seg, 0.5)
    s, _ = S.score_patient(ev, crises((1000, 1060)), segb, 'strict')
    check('strict : alarme de 5 s ignorée', not s['detectee'].iat[0])
    p = np.zeros(N); p[1010:1022] = 0.9
    s, _ = S.score_patient(S.alarms(p, seg, 0.5), crises((1000, 1060)), segb, 'strict')
    check('strict : alarme de 12 s détectée', s['detectee'].iat[0])

    # 6. crise d'un groupe non ciblé : alarme neutre (ni détection ni fausse alarme)
    p = np.zeros(N); p[1010:1030] = 0.9
    s, a = S.score_patient(S.alarms(p, seg, 0.5), crises((1000, 1060), groupe='Non motrices'), segb, 'standard',
                           in_scope=['Hypermotrices et convulsives'])
    check('crise hors périmètre : alarme neutre', len(s) == 0 and (a['statut'] == 'neutre').sum() == 1)

    # 7. post-traitement k-ième maximum : 5 secondes à 0,9 sur 20 -> score >= 0,5 ; 4 secondes -> score < 0,5
    p = np.zeros(100); p[50:55] = 0.9
    sc = S.postprocess(p, np.zeros(100, int), method='kieme_max', n=20, k=5)
    check('k-ième max (k=5, n=20) : 5 secondes hautes suffisent', sc.max() >= 0.5)
    p = np.zeros(100); p[50:54] = 0.9
    sc = S.postprocess(p, np.zeros(100, int), method='kieme_max', n=20, k=5)
    check('k-ième max (k=5, n=20) : 4 secondes hautes ne suffisent pas', sc.max() < 0.5)
    p = np.zeros(100); p[[40, 44, 48, 52, 56]] = 0.9                  # 5 secondes non consécutives sur 20
    sc = S.postprocess(p, np.zeros(100, int), method='kieme_max', n=20, k=5)
    check('k-ième max : 5 secondes non consécutives dans 20 s suffisent', sc.max() >= 0.5)
    p = np.zeros(100); p[50] = 1.0
    sc = S.postprocess(p, np.zeros(100, int), method='moyenne', n=10)
    check('moyenne 10 s : un pic isolé à 1,0 reste sous 0,5', sc.max() < 0.5)

    # 8. les fenêtres ne traversent pas les segments
    p = np.zeros(40); p[15:20] = 0.9
    sc = S.postprocess(p, np.r_[np.zeros(18, int), np.ones(22, int)], method='kieme_max', n=10, k=3)
    check('fenêtres limitées au segment', sc[18:20].max() < 0.5)

    # 9. Wilson et bootstrap
    lo, hi = S.wilson(0, 1)
    check('Wilson 0/1 : intervalle large [0 ; 0,79]', lo == 0 and 0.75 < hi < 0.85)
    per = pd.DataFrame({'patient_id': list('abcd'), 'n_crises': [10, 2, 3, 1], 'n_detectees': [9, 0, 1, 1],
                        'n_fa': [5, 1, 0, 2], 'heures': [100, 50, 80, 40], 'latences': [[10] * 9, [], [5], [2]]})
    sm = S.summarize(per); ci = S.bootstrap_ci(per, n_boot=500)
    check('sensibilité poolée = 11/16', abs(sm['sensibilite'] - 11 / 16) < 1e-9)
    check('IC bootstrap contient l\'estimation', ci['sensibilite'][0] <= sm['sensibilite'] <= ci['sensibilite'][1])

    print(f"\n{sum(r['ok'] for r in resultats)}/{len(resultats)} tests réussis")
    assert all(r['ok'] for r in resultats), 'au moins un test a échoué'


if __name__ == '__main__':
    test_comptage()
