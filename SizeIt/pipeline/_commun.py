"""Outils communs aux scripts du pipeline : import du package, arguments, exécution parallèle, journal."""
import argparse
import logging
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')          # un modèle par processus : peu de fils chacun

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # dossier SizeIt/

from joblib import Parallel, delayed  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('sizeit')


def arguments(description, **extra):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--patients', nargs='*', default=None,
                   help='patients à traiter (ex. sub-001 sub-002) ; par défaut tous, ou la liste --liste')
    p.add_argument('--liste', default=None, help='fichier texte, un patient par ligne (ex. patients_developpement.txt)')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='processus en parallèle')
    for nom, kw in extra.items():
        p.add_argument(f'--{nom}', **kw)
    a = p.parse_args()
    if a.liste:
        a.patients = [l.strip() for l in open(a.liste) if l.strip() and not l.startswith('#')]
    return a


def en_parallele(fonction, elements, jobs, nom):
    t0 = time.time()
    log.info(f'{nom} : {len(elements)} éléments, {jobs} processus')
    res = Parallel(n_jobs=jobs, verbose=0)(delayed(fonction)(e) for e in elements)
    log.info(f'{nom} : terminé en {(time.time() - t0) / 60:.1f} min')
    return res
