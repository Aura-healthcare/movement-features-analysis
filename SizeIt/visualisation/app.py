"""Visualisation interactive des crises et des prédictions (Dash / Plotly, serveur local).

Lancement :
    python app.py                              # http://127.0.0.1:8050
    python app.py --port 8060 --host 0.0.0.0
Depuis un poste distant : tunnel SSH, ex. `ssh -L 8050:localhost:8050 <serveur>` puis http://localhost:8050

Menus : modèle → patient → groupe de crises → type → crise. Réglages : signal (probabilité ou probabilité CRF),
post-traitement (moyenne sur n s ou k-ième maximum sur n s), seuil, règle de comptage, largeur de fenêtre.
Vues : (1) la crise sélectionnée ± fenêtre : accélération brute 25 Hz, FC et agitation, features choisies (dont les
puissances spectrales CL_Fourier_2_7 / CL_Fourier_7_125, échelle log), probabilité, score, seuil, alarmes ; (2) tout l'enregistrement du patient (cliquer sur une crise pour la sélectionner) ; (3) tableau des crises
du patient et résumé détections / fausses alarmes. Comptage identique aux rapports (sizeit/scoring.py).
"""
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ctx

import config_visu as CV
import sources as SRC

COL = {'crise': 'rgba(227,73,72,0.18)', 'alarme': 'rgba(235,104,52,0.25)', 'fa': 'rgba(235,104,52,0.25)',
       'proba': '#9ec5f4', 'score': '#2a78d6', 'seuil': '#52514e', 'fc': '#e34948', 'agit': '#1baf7a',
       'x': '#2a78d6', 'y': '#eb6834', 'z': '#1baf7a', 'norme': '#0b0b0b', 'det': '#1baf7a', 'manq': '#e34948'}
CAT = SRC.catalogue()
MODELS = list(CAT)
DEFAULT_MODEL = next((m for m in MODELS if m == 'predictions'), MODELS[0])


def statut(v):
    # True / False / None (hors périmètre), quel que soit le type (bool numpy, NaN)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return bool(v)


def lbl(text, child, width='190px'):
    return html.Div([html.Div(text, style={'fontSize': '12px', 'color': '#52514e', 'marginBottom': '2px'}), child],
                    style={'width': width, 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '12px'})


app = Dash(__name__, title='Visualisation des crises SZIT')
app.layout = html.Div(style={'fontFamily': 'sans-serif', 'margin': '12px 18px'}, children=[
    html.H3('Visualisation des crises et des prédictions', style={'margin': '0 0 10px 0'}),
    html.Div([
        lbl('Modèle', dcc.Dropdown(MODELS, DEFAULT_MODEL, id='modele', clearable=False), '520px'),
        lbl('Patient', dcc.Dropdown(id='patient', clearable=False)),
        lbl('Groupe de crises', dcc.Dropdown(id='groupe', placeholder='tous'), '260px'),
        lbl('Type de crise', dcc.Dropdown(id='type', placeholder='tous')),
        lbl('Crise', dcc.Dropdown(id='crise', clearable=False), '360px'),
    ]),
    html.Div(style={'marginTop': '8px'}, children=[
        lbl('Signal', dcc.Dropdown(id='signal', clearable=False), '170px'),
        lbl('Post-traitement', dcc.Dropdown([{'label': 'moyenne sur n s', 'value': 'moyenne'},
                                             {'label': 'k-ième max sur n s', 'value': 'kieme_max'},
                                             {'label': 'brut', 'value': 'brut'}], 'moyenne', id='methode',
                                            clearable=False), '170px'),
        lbl('n (s)', dcc.Input(id='n', type='number', value=10, min=1, max=300, step=1, style={'width': '70px'}), '80px'),
        lbl('k', dcc.Input(id='k', type='number', value=5, min=1, max=300, step=1, style={'width': '60px'}), '70px'),
        lbl('Seuil', dcc.Slider(0.05, 0.95, 0.05, value=0.5, id='seuil', marks={0.1: '0,1', 0.5: '0,5', 0.9: '0,9'},
                                tooltip={'placement': 'bottom'}), '260px'),
        lbl('Règle de comptage', dcc.RadioItems([{'label': ' standard (−30 s / +60 s)', 'value': 'standard'},
                                                 {'label': ' stricte (≥ 10 s pendant la crise)', 'value': 'strict'}],
                                                'standard', id='regle'), '250px'),
        lbl('Fenêtre (± min)', dcc.Slider(1, 30, 1, value=3, id='fenetre', marks={1: '1', 5: '5', 15: '15', 30: '30'},
                                          tooltip={'placement': 'bottom'}), '220px'),
        lbl('Accélération brute', dcc.Checklist([{'label': ' afficher', 'value': 'oui'}], ['oui'], id='brut'), '130px'),
    ]),
    html.Div(style={'marginTop': '8px'}, children=[
        lbl('Features à afficher (rangée « Features », échelle log)',
            dcc.Dropdown(id='features', multi=True, value=CV.DEFAULT_FEATURES), '760px'),
    ]),
    html.Div(id='resume', style={'margin': '10px 0', 'padding': '8px 12px', 'background': '#f2f6fc', 'fontSize': '13px'}),
    dcc.Loading(dcc.Graph(id='fig_crise', style={'height': '900px'})),
    dcc.Loading(dcc.Graph(id='fig_patient', style={'height': '300px'})),
    html.H4('Crises du patient', style={'margin': '8px 0 4px 0'}),
    dash_table.DataTable(id='table', page_size=15, sort_action='native', filter_action='native',
                         style_cell={'fontSize': '12px', 'padding': '3px', 'fontFamily': 'sans-serif'},
                         style_header={'fontWeight': 'bold', 'background': '#f2f6fc'},
                         style_data_conditional=[{'if': {'filter_query': '{détectée} = "non"'}, 'color': COL['manq']}]),
])


# ---------------------------------------------------------------- menus en cascade
@app.callback(Output('patient', 'options'), Output('patient', 'value'), Output('signal', 'options'),
              Output('signal', 'value'), Input('modele', 'value'), State('patient', 'value'))
def maj_patients(model, current):
    pats = SRC.patients_du_modele(model)
    pid = current if current in pats else pats[0]
    cols = [c for c in ('proba', 'crf_proba') if c in SRC.predictions(model, pid).columns]
    return pats, pid, [{'label': {'proba': 'probabilité', 'crf_proba': 'probabilité CRF'}[c], 'value': c} for c in cols], 'proba'


@app.callback(Output('features', 'options'), Input('patient', 'value'))
def maj_features(pid):
    return [{'label': c.replace('acc_EEG_SD_ACC_', 'acc_'), 'value': c} for c in SRC.features_disponibles(pid)] if pid else []


@app.callback(Output('groupe', 'options'), Output('type', 'options'), Input('patient', 'value'), Input('groupe', 'value'))
def maj_groupes(pid, groupe):
    sz = SRC.crises(pid)
    t = sz if not groupe else sz[sz['groupe'] == groupe]
    return sorted(sz['groupe'].dropna().unique()), sorted(t['type'].dropna().unique())


@app.callback(Output('crise', 'options'), Output('crise', 'value'),
              Input('patient', 'value'), Input('groupe', 'value'), Input('type', 'value'),
              Input('fig_patient', 'clickData'), State('crise', 'value'))
def maj_crises(pid, groupe, typ, click, current):
    sz = SRC.crises(pid)
    if groupe:
        sz = sz[sz['groupe'] == groupe]
    if typ:
        sz = sz[sz['type'] == typ]
    opts = [{'label': f"{r.seizure_id} | {r.type} | {int(r.duree_s)} s | run {r.rn}", 'value': r.seizure_id}
            for r in sz.itertuples()]
    ids = [o['value'] for o in opts]
    if click and click['points'][0].get('customdata') and ctx.triggered_id == 'fig_patient':
        clicked = click['points'][0]['customdata']
        clicked = clicked[0] if isinstance(clicked, list) else clicked
        return opts, clicked if clicked in ids else (ids[0] if ids else None)
    return opts, current if current in ids else (ids[0] if ids else None)


# ---------------------------------------------------------------- figures
def _shade(fig, x0, x1, color, rows, label=None):
    for r in rows:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, row=r, col=1, layer='below')


@app.callback(Output('fig_crise', 'figure'), Output('fig_patient', 'figure'), Output('table', 'data'),
              Output('table', 'columns'), Output('resume', 'children'),
              Input('modele', 'value'), Input('patient', 'value'), Input('crise', 'value'), Input('signal', 'value'),
              Input('methode', 'value'), Input('n', 'value'), Input('k', 'value'), Input('seuil', 'value'),
              Input('regle', 'value'), Input('fenetre', 'value'), Input('brut', 'value'), Input('features', 'value'))
def maj_figures(model, pid, sid, signal, methode, n, k, seuil, regle, fenetre, brut, features):
    if not pid or not signal:
        return (no_update,) * 5
    n = int(n or 10); k = int(min(k or 5, n))
    df = SRC.patient(model, pid)
    score, sz_eval, al = SRC.evaluer(model, pid, signal, methode, n, k, seuil, regle)
    sz_all = SRC.crises(pid)
    det = sz_eval.set_index('seizure_id')[['detectee', 'latence_s']] if len(sz_eval) else pd.DataFrame(columns=['detectee', 'latence_s'])
    hours = len(df) / 3600
    n_fa = int((al['statut'] == 'fausse alarme').sum())
    n_ci = len(sz_eval); n_det = int(sz_eval['detectee'].sum()) if n_ci else 0
    grp = SRC.catalogue()[model]['groupe']

    # ---- table des crises du patient
    tab = sz_all.merge(det, left_on='seizure_id', right_index=True, how='left')
    rows = [{'crise': r.seizure_id, 'type': r.type, 'groupe': r.groupe, 'durée (s)': int(r.duree_s), 'run': int(r.rn),
             'début dans le run': f"{int(df.at[int(r.start), 'elapsed_s'] // 3600)}h{int(df.at[int(r.start), 'elapsed_s'] % 3600 // 60):02d}",
             'détectée': {True: 'oui', False: 'non', None: 'hors périmètre'}[statut(r.detectee)],
             'latence (s)': None if pd.isna(r.latence_s) else int(r.latence_s),
             'proba max': round(float(np.nanmax(df[signal].values[int(r.start):int(r.end)])), 3)
             if np.isfinite(df[signal].values[int(r.start):int(r.end)]).any() else None} for r in tab.itertuples()]
    columns = [{'name': c, 'id': c} for c in rows[0]] if rows else []

    # ---- vue patient entier (max du score par minute)
    step = 60
    nb = len(score) // step
    smax = np.nanmax(score[:nb * step].reshape(nb, step), axis=1)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scattergl(x=np.arange(nb) * step / 3600, y=smax, mode='lines', line=dict(color=COL['score'], width=1),
                                 name='score (max par minute)', hoverinfo='skip'))
    fa = al[al['statut'] == 'fausse alarme']
    fig_p.add_trace(go.Scatter(x=fa['start'] / 3600, y=[1.04] * len(fa), mode='markers', marker=dict(symbol='line-ns-open', size=8,
                               color='#eb6834'), name='fausses alarmes', hovertext=[f'fausse alarme {d} s' for d in fa['duree_s']]))
    stat = tab['detectee'].map(lambda v: {True: 'détectée', False: 'manquée', None: 'hors périmètre'}[statut(v)])
    for s_, colr in (('détectée', COL['det']), ('manquée', COL['manq']), ('hors périmètre', '#a8a6a0')):
        t = tab[stat == s_]
        fig_p.add_trace(go.Scatter(x=t['start'] / 3600, y=[1.1] * len(t), mode='markers', name=f'crise {s_}',
                                   marker=dict(size=11, color=colr, symbol='triangle-down', line=dict(width=1, color='white')),
                                   customdata=t['seizure_id'], hovertext=t['seizure_id'] + ' ' + t['type'], hoverinfo='text'))
    if sid in set(tab['seizure_id']):
        x0 = float(tab.loc[tab['seizure_id'] == sid, 'start'].iat[0]) / 3600
        fig_p.add_vline(x=x0, line=dict(color='#0b0b0b', width=1, dash='dot'))
    fig_p.add_hline(y=seuil, line=dict(color=COL['seuil'], width=1, dash='dash'))
    fig_p.update_layout(margin=dict(l=50, r=10, t=30, b=40), yaxis=dict(range=[0, 1.15], title='score'),
                        xaxis=dict(title="temps d'enregistrement cumulé (h) — cliquer sur une crise pour la sélectionner"),
                        legend=dict(orientation='h', y=1.18), title=dict(text=f'{pid} : tout l\'enregistrement', x=0.01, font=dict(size=13)))

    # ---- vue de la crise : une rangée par élément affiché
    features = [f for f in (features or []) if f in df.columns]
    rows_ = (['brut'] if 'oui' in (brut or []) else []) + ['fc', 'agit'] + (['feat'] if features else []) + ['proba']
    H = {'brut': 0.27, 'fc': 0.16, 'agit': 0.16, 'feat': 0.18, 'proba': 0.23}
    T = {'brut': 'Accélération brute (25 Hz)', 'fc': 'Fréquence cardiaque (bpm)',
         'agit': 'Agitation (écart-type de la norme, features 1 s)', 'feat': 'Features (échelle log)',
         'proba': 'Probabilité, score post-traité et alarmes'}
    R = {name: i + 1 for i, name in enumerate(rows_)}
    nrows = len(rows_)
    hs = np.array([H[r] for r in rows_]); hs = (hs / hs.sum()).tolist()
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=hs,
                        subplot_titles=[T[r] for r in rows_])
    resume = []
    if sid and sid in set(sz_all['seizure_id']):
        s = sz_all[sz_all['seizure_id'] == sid].iloc[0]
        a, b = int(s['start']), int(s['end'])
        lo_seg, hi_seg = SRC.bornes_segments(pid)[int(s['segment'])]
        w0, w1 = max(lo_seg, a - fenetre * 60), min(hi_seg, b + fenetre * 60)
        t = np.arange(w0, w1) - a
        if 'brut' in R:
            raw = SRC.accel_brute(df, w0, w1, a)
            for c, colr, name in (('EEG SD ACC X', COL['x'], 'x'), ('EEG SD ACC Y', COL['y'], 'y'),
                                  ('EEG SD ACC Z', COL['z'], 'z'), ('norme_1', COL['norme'], 'norme')):
                if c in raw:
                    fig.add_trace(go.Scattergl(x=raw['t'], y=raw[c], mode='lines', line=dict(color=colr, width=0.8),
                                               name=f'acc {name}', legendgroup='acc'), row=R['brut'], col=1)
        fig.add_trace(go.Scattergl(x=t, y=df[CV.HR_COL].values[w0:w1], mode='lines', line=dict(color=COL['fc'], width=1.2),
                                   name='FC'), row=R['fc'], col=1)
        fig.add_trace(go.Scattergl(x=t, y=df[CV.ACT_COL].values[w0:w1], mode='lines', line=dict(color=COL['agit'], width=1.2),
                                   name='agitation'), row=R['agit'], col=1)
        if 'feat' in R:
            palette = ['#eb6834', '#4a3aa7', '#2a78d6', '#1baf7a', '#eda100', '#e87ba4']
            for i, f in enumerate(features):
                fig.add_trace(go.Scattergl(x=t, y=df[f].values[w0:w1], mode='lines', name=f.replace('acc_EEG_SD_ACC_', 'acc_'),
                                           line=dict(color=palette[i % len(palette)], width=1.6 if f.startswith('CL_') else 1,
                                                     dash='solid' if f.startswith('CL_') else 'dot')), row=R['feat'], col=1)
            fig.update_yaxes(type='log', row=R['feat'], col=1)
        fig.add_trace(go.Scattergl(x=t, y=df[signal].values[w0:w1], mode='lines', line=dict(color=COL['proba'], width=1),
                                   name='probabilité brute'), row=R['proba'], col=1)
        fig.add_trace(go.Scattergl(x=t, y=score[w0:w1], mode='lines', line=dict(color=COL['score'], width=2),
                                   name='score post-traité'), row=R['proba'], col=1)
        if 'crf_state' in df.columns and signal == 'crf_proba':
            fig.add_trace(go.Scattergl(x=t, y=df['crf_state'].values[w0:w1] * 0.08, mode='lines', fill='tozeroy',
                                       line=dict(color='#4a3aa7', width=0), name='état CRF'), row=R['proba'], col=1)
        fig.add_hline(y=seuil, line=dict(color=COL['seuil'], width=1, dash='dash'), row=R['proba'], col=1)
        # crises (toutes celles de la fenêtre) et alarmes
        for r in sz_all[(sz_all['end'] > w0) & (sz_all['start'] < w1)].itertuples():
            _shade(fig, r.start - a, r.end - a, COL['crise'], range(1, nrows + 1))
        for r in al[(al['end'] > w0) & (al['start'] < w1)].itertuples():
            colr = {'détection': 'rgba(27,175,122,0.25)', 'fausse alarme': COL['fa'], 'neutre': 'rgba(168,166,160,0.3)'}[r.statut]
            fig.add_vrect(x0=r.start - a, x1=r.end - a, fillcolor=colr, line_width=0, row=R['proba'], col=1)
        d_ = statut(det['detectee'].get(sid)) if sid in det.index else None
        st = 'hors périmètre du modèle' if d_ is None else (
            f"DÉTECTÉE (latence {det['latence_s'].get(sid):+.0f} s)" if d_ else 'MANQUÉE')
        resume.append(html.Div([html.B(f"{sid} — {s['type']} ({s['groupe']}, {s['categorie']}) — {int(s['duree_s'])} s — run {s['rn']} — "),
                                html.B(st, style={'color': COL['det'] if d_ is True else COL['manq']})]))
        fig.update_xaxes(title_text='secondes (0 = début de la crise annotée) ; rouge = crise, vert = alarme de détection, '
                                    'orange = fausse alarme, gris = alarme sur une crise hors périmètre', row=nrows, col=1)
    fig.update_yaxes(range=[0, 1.02], row=nrows, col=1)
    fig.update_layout(margin=dict(l=60, r=10, t=40, b=50), legend=dict(orientation='h', y=1.06), hovermode='x unified')
    resume.append(html.Div(
        f"Patient {pid} | modèle « {model} »{' (groupe ' + grp + ')' if grp else ''} | signal {signal}, "
        f"{ {'moyenne': f'moyenne sur {n} s', 'kieme_max': f'{k}e max sur {n} s', 'brut': 'brut'}[methode] }, seuil {seuil}, "
        f"règle {regle} : {n_det}/{n_ci} crises détectées, {n_fa} fausses alarmes ({n_fa / hours * 24:.1f} / 24 h, "
        f"{int((al['post_ictale']).sum()) if len(al) else 0} post-ictales), {hours:.0f} h d'enregistrement."))
    return fig, fig_p, rows, columns, resume


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=CV.DEFAULT_PORT)
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    print(f'{len(MODELS)} modèles trouvés. Ouvrir http://{a.host}:{a.port}')
    app.run(host=a.host, port=a.port, debug=a.debug)
