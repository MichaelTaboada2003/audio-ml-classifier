#!/usr/bin/env python3
"""
filtrar_audios.py
=================
Calcula scores acusticos por emocion para cada audio del dataset.
NO reasigna etiquetas — cada audio mantiene su clase original con su score.

Los notebooks luego seleccionan los top-N audios por clase para entrenar,
o filtran por umbral de score si lo necesitan.

Scores calculados (cada uno en [0, 1]):
  - score_enojo:        firma de enojo (energia alta, pitch variable, dinamica rapida)
  - score_tristeza:     firma de tristeza (energia baja, pitch grave, monotonia)
  - score_tranquilidad: firma de calma (energia media-baja estable, pitch suave)
  - score_neutro:       ausencia de firma emocional (audio plano)

Uso (desde la raiz del proyecto):
    python scripts/filtrar_audios.py

Output:
    - outputs/reporte_filtrado_v2.csv       (una fila por audio, todos los scores)
    - outputs/figuras/reporte_filtrado_v2.png
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

warnings.filterwarnings('ignore')

# Rutas relativas a la raiz del proyecto
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, 'data', 'AUDIOS MACHINE LEARNING')
REPORTE_CSV = os.path.join(_ROOT, 'outputs', 'reporte_filtrado_v2.csv')
REPORTE_PNG = os.path.join(_ROOT, 'outputs', 'figuras', 'reporte_filtrado_v2.png')

CLASES = ['Aburrido', 'Enojo', 'Tranquilidad', 'Tristeza']
EXTS   = {'.ogg', '.mp3', '.mp4', '.mpeg', '.wav', '.flac', '.m4a'}
SR     = 22050


# --- Metricas acusticas ---
def calcular_metricas(ruta):
    """
    Extrae metricas acusticas globales. Optimizado: 10s, hop=2048.
    """
    try:
        y, _ = librosa.load(ruta, sr=SR, mono=True, duration=10.0)
        if len(y) < SR * 0.3:
            return None
        hop = 2048

        # Pitch (F0)
        f0 = librosa.yin(y, fmin=80, fmax=600, hop_length=hop)
        f0_v = f0[(f0 > 0) & (f0 < 600) & np.isfinite(f0)]
        if len(f0_v) < 5:
            pitch_mean_st = pitch_std_st = pitch_range_st = 0.0
        else:
            f0_log2 = np.log2(f0_v)
            pitch_mean_st  = float((np.median(f0_log2) - np.log2(110)) * 12)
            pitch_std_st   = float(np.std(f0_log2) * 12)
            pitch_range_st = float((np.percentile(f0_log2, 95) -
                                     np.percentile(f0_log2, 5)) * 12)

        # Energia
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))
        energy_cv = rms_std / rms_mean if rms_mean > 1e-6 else 0.0
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        dynamic_range_db = float(np.percentile(rms_db, 95) -
                                  np.percentile(rms_db, 5))
        energy_level = float(np.mean(rms_db))

        # Tempo / dinamica
        onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
        onset_rate = float(np.mean(onset_env))
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        zcr_mean = float(np.mean(zcr))

        # Silencio
        umb_silencio = 0.1 * np.max(rms) if np.max(rms) > 0 else 0
        frac_silencio = float(np.mean(rms < umb_silencio))

        # Brillo espectral
        centroid = librosa.feature.spectral_centroid(y=y, sr=SR, hop_length=hop)[0]
        centroid_mean = float(np.mean(centroid))

        return {
            'pitch_mean_st':    pitch_mean_st,
            'pitch_std_st':     pitch_std_st,
            'pitch_range_st':   pitch_range_st,
            'energy_cv':        energy_cv,
            'energy_level_db':  energy_level,
            'dynamic_range_db': dynamic_range_db,
            'onset_rate':       onset_rate,
            'zcr_mean':         zcr_mean,
            'frac_silencio':    frac_silencio,
            'centroid_hz':      centroid_mean,
            'duracion_s':       len(y) / SR,
        }
    except Exception as e:
        print('  [ERROR] {}: {}'.format(os.path.basename(ruta), e), flush=True)
        return None


# --- Scores por emocion ---
def _norm(x, lo, hi):
    """Normaliza x del rango [lo, hi] a [0, 1] saturando."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score_enojo(m):
    """
    Firma de enojo: energia alta y dinamica, pitch medio-alto variable,
    onsets frecuentes, brillo espectral alto.
    """
    s_energia    = _norm(m['energy_level_db'],   -40.0, -22.0)
    s_dinamica   = _norm(m['dynamic_range_db'],   25.0,  50.0)
    s_pitch_alt  = _norm(m['pitch_mean_st'],      10.0,  24.0)
    s_pitch_var  = _norm(m['pitch_std_st'],        2.5,   8.0)
    s_onsets     = _norm(m['onset_rate'],          1.0,   4.0)
    s_brillo     = _norm(m['centroid_hz'],      1500.0, 3500.0)
    return (0.25 * s_energia + 0.20 * s_dinamica + 0.15 * s_pitch_alt +
            0.15 * s_pitch_var + 0.15 * s_onsets + 0.10 * s_brillo)


def score_tristeza(m):
    """
    Firma de tristeza: energia baja, pitch bajo y monotono,
    fraccion alta de silencio, ritmo lento, brillo bajo.
    """
    s_energia_baja = _norm(-m['energy_level_db'],    22.0,  40.0)
    s_pitch_bajo   = _norm(-m['pitch_mean_st'],      -5.0,  10.0)
    s_monotonia    = _norm(-m['pitch_std_st'],       -5.0,  -1.5)
    s_silencios    = _norm(m['frac_silencio'],         0.10,  0.45)
    s_lentitud     = _norm(-m['onset_rate'],         -3.0,  -1.0)
    s_brillo_bajo  = _norm(-m['centroid_hz'],     -3000.0, -1200.0)
    return (0.20 * s_energia_baja + 0.15 * s_pitch_bajo + 0.20 * s_monotonia +
            0.15 * s_silencios + 0.15 * s_lentitud + 0.15 * s_brillo_bajo)


def score_tranquilidad(m):
    """
    Firma de calma activa: energia media estable, pitch en rango medio
    sin extremos, pocas variaciones bruscas, brillo medio.
    Diferencia clave con neutro: la tranquilidad es "voz hablada normal",
    no ausencia de senal.
    """
    # Energia en rango medio (no muy alta, no muy baja)
    e = m['energy_level_db']
    s_energia_media = 1.0 - min(abs(e - (-30.0)) / 12.0, 1.0)  # pico en -30 dB

    # Pitch en rango medio (no extremo)
    p = m['pitch_mean_st']
    s_pitch_medio = 1.0 - min(abs(p - 6.0) / 8.0, 1.0)  # pico en 6 st

    # Variabilidad de pitch baja-media (estable pero no plano como tristeza)
    s_pitch_estable = _norm(m['pitch_std_st'], 1.0, 3.5)
    s_pitch_estable = min(s_pitch_estable, 1.0 - max(m['pitch_std_st'] - 4.5, 0) / 3.5)
    s_pitch_estable = max(s_pitch_estable, 0.0)

    # Pocos silencios largos (habla continua)
    s_continuidad = 1.0 - _norm(m['frac_silencio'], 0.10, 0.35)

    # Dinamica moderada
    d = m['dynamic_range_db']
    s_dinamica_mod = 1.0 - min(abs(d - 28.0) / 18.0, 1.0)

    # Onsets moderados
    o = m['onset_rate']
    s_ritmo_mod = 1.0 - min(abs(o - 2.0) / 1.5, 1.0)

    return (0.20 * s_energia_media + 0.15 * s_pitch_medio +
            0.20 * s_pitch_estable + 0.15 * s_continuidad +
            0.15 * s_dinamica_mod + 0.15 * s_ritmo_mod)


def score_neutro(m):
    """
    Score de audio plano/sin firma emocional clara. Util para detectar
    audios de baja calidad o sin expresividad.
    """
    s_pitch_plano   = _norm(-m['pitch_std_st'],     -4.0, -1.5)
    s_dinamica_baja = _norm(-m['dynamic_range_db'], -35.0, -20.0)
    s_energia_med   = 1.0 - abs(_norm(m['energy_level_db'], -40.0, -22.0) - 0.5) * 2
    return 0.4 * s_pitch_plano + 0.4 * s_dinamica_baja + 0.2 * s_energia_med


# --- Analisis del dataset ---
def analizar_dataset(data_dir):
    """Recorre todas las carpetas y calcula scores por audio."""
    todos = [(clase, nombre)
             for clase in CLASES
             if os.path.isdir(os.path.join(data_dir, clase))
             for nombre in sorted(os.listdir(os.path.join(data_dir, clase)))
             if os.path.splitext(nombre)[1].lower() in EXTS]
    total = len(todos)
    print('  Total de audios encontrados: {}'.format(total), flush=True)

    filas = []
    for i, (clase, nombre) in enumerate(todos, 1):
        print('  [{:3d}/{}] {}/{}'.format(i, total, clase, nombre), flush=True)
        ruta = os.path.join(data_dir, clase, nombre)
        m = calcular_metricas(ruta)
        if m is None:
            continue
        m.update({
            'archivo':            nombre,
            'clase_original':     clase,
            'recolector':         nombre[:2],
            'score_enojo':        score_enojo(m),
            'score_tristeza':     score_tristeza(m),
            'score_tranquilidad': score_tranquilidad(m),
            'score_neutro':       score_neutro(m),
        })
        filas.append(m)
    return pd.DataFrame(filas)


# --- Reportes ---
def imprimir_reporte(df):
    print('\n' + '=' * 72)
    print('SCORES ACUSTICOS POR AUDIO')
    print('=' * 72)
    print('\nTotal de audios analizados: {}'.format(len(df)))

    print('\nDistribucion por clase original:')
    print(df['clase_original'].value_counts().to_string())

    print('\nMedia de score por clase original:')
    cols_score = ['score_enojo', 'score_tristeza', 'score_tranquilidad', 'score_neutro']
    print(df.groupby('clase_original')[cols_score].mean().round(3).to_string())

    # Top-5 por cada score
    for score in cols_score:
        print('\nTop 5 por {}:'.format(score))
        top = df.sort_values(score, ascending=False).head(5)
        print(top[['archivo', 'clase_original', score]].round(3).to_string(index=False))

    # --- Resumen de seleccion por ranking ---
    print('\n' + '=' * 72)
    print('RESUMEN DE SELECCION POR RANKING')
    print('=' * 72)
    print('\nAudios disponibles por clase (ordenados por su score correspondiente):')
    print('  (Se muestra cuantos audios de cada clase tienen score > umbral)\n')

    umbrales = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    score_map = {
        'Enojo':        'score_enojo',
        'Tristeza':     'score_tristeza',
        'Tranquilidad': 'score_tranquilidad',
    }

    print('{:<15} {:>8}  '.format('Clase', 'Total') +
          '  '.join('{:>5}'.format('>' + str(u)) for u in umbrales))
    print('-' * 72)
    for clase, score_col in score_map.items():
        sub = df[df['clase_original'] == clase][score_col]
        total = len(sub)
        counts = [int((sub > u).sum()) for u in umbrales]
        print('{:<15} {:>8}  '.format(clase, total) +
              '  '.join('{:>5}'.format(c) for c in counts))

    # --- Sugerencia de configuraciones balanceadas ---
    print('\nConfiguraciones balanceadas sugeridas:')
    for n in [10, 15, 20, 25, 30]:
        disponible = {}
        for clase, score_col in score_map.items():
            sub = df[df['clase_original'] == clase].sort_values(score_col, ascending=False)
            disponible[clase] = len(sub)
        max_n = min(disponible.values())
        if n > max_n:
            print('  Top-{}: no posible (max disponible: {})'.format(n, max_n))
            continue
        # Mostrar el score minimo que entraria en cada clase
        scores_min = {}
        for clase, score_col in score_map.items():
            sub = df[df['clase_original'] == clase].sort_values(score_col, ascending=False)
            scores_min[clase] = sub.iloc[n-1][score_col] if n <= len(sub) else 0
        print('  Top-{}: Enojo>={:.3f}  Tristeza>={:.3f}  Tranquilidad>={:.3f}'.format(
            n, scores_min['Enojo'], scores_min['Tristeza'], scores_min['Tranquilidad']))

    # --- Detalle final: cuantos audios "buenos" por clase ---
    print('\n' + '-' * 72)
    print('AUDIOS UTILES POR CLASE (score de su emocion > 0.40):')
    print('-' * 72)
    for clase, score_col in score_map.items():
        buenos = df[(df['clase_original'] == clase) & (df[score_col] > 0.40)]
        total = len(df[df['clase_original'] == clase])
        print('  {:<15} {}/{} audios utiles'.format(clase, len(buenos), total))
        if len(buenos) > 0:
            top3 = buenos.sort_values(score_col, ascending=False).head(3)
            for _, r in top3.iterrows():
                print('    {} (score={:.3f})'.format(r['archivo'], r[score_col]))
    print()

    # --- Listado completo por clase con PASA / NO PASA ---
    print('=' * 72)
    print('DETALLE POR CLASE — RANKING COMPLETO')
    print('=' * 72)
    for clase, score_col in score_map.items():
        sub = df[df['clase_original'] == clase].sort_values(score_col, ascending=False)
        n_pasan = (sub[score_col] > 0.35).sum()
        print('\n--- {} ({} audios, {} pasan umbral 0.35) ---'.format(
            clase.upper(), len(sub), n_pasan))
        print('{:<4} {:<28} {:>6} {:>8}  {}'.format(
            '#', 'Archivo', 'Recol', 'Score', 'Estado'))
        print('  ' + '-' * 60)
        for rank, (_, r) in enumerate(sub.iterrows(), 1):
            score_val = r[score_col]
            if score_val > 0.45:
                estado = 'EXCELENTE'
            elif score_val > 0.35:
                estado = 'PASA'
            elif score_val > 0.30:
                estado = 'borderline'
            else:
                estado = 'NO PASA'
            print('{:<4} {:<28} {:>6} {:>8.3f}  {}'.format(
                rank, r['archivo'], r['recolector'], score_val, estado))
    print()

    # --- 15 peores por clase con metricas acusticas para diagnostico ---
    print('=' * 72)
    print('DIAGNOSTICO — 15 PEORES POR CLASE (por que no pasaron)')
    print('=' * 72)
    metricas_diag = {
        'Enojo':        ('score_enojo',        ['energy_level_db', 'dynamic_range_db', 'pitch_std_st', 'onset_rate', 'centroid_hz']),
        'Tristeza':     ('score_tristeza',      ['energy_level_db', 'pitch_mean_st',    'pitch_std_st', 'frac_silencio', 'onset_rate']),
        'Tranquilidad': ('score_tranquilidad',  ['energy_level_db', 'pitch_std_st',     'frac_silencio','dynamic_range_db','onset_rate']),
    }
    # Referencia: que valores esperamos para cada emocion
    referencia = {
        'Enojo':        'Esperado: energy_db alto (>-30), dynamic_range alto (>30), pitch_std alto (>4), onsets altos (>2)',
        'Tristeza':     'Esperado: energy_db bajo (<-35), pitch_mean bajo (<5st), pitch_std bajo (<2), frac_silencio alto (>0.2)',
        'Tranquilidad': 'Esperado: energy_db medio (-35 a -25), pitch_std medio (1-4), frac_silencio bajo (<0.15)',
    }
    for clase, (score_col, metricas) in metricas_diag.items():
        sub = df[df['clase_original'] == clase].sort_values(score_col, ascending=True).head(15)
        print('\n--- {} — 15 peores ---'.format(clase.upper()))
        print('  {}'.format(referencia[clase]))
        header = '{:<28} {:>6} {:>7}  ' + '  '.join('{:>12}' for _ in metricas)
        print(header.format('Archivo', 'Recol', 'Score', *metricas))
        print('  ' + '-' * (28 + 6 + 7 + 14 * len(metricas)))
        for _, r in sub.iterrows():
            vals = [r[m] if m in r else float('nan') for m in metricas]
            row = '{:<28} {:>6} {:>7.3f}  ' + '  '.join('{:>12.3f}' for _ in metricas)
            print(row.format(r['archivo'], r['recolector'], r[score_col], *vals))
    print()


def graficar_reporte(df, output_path):
    cols_score = ['score_enojo', 'score_tristeza', 'score_tranquilidad', 'score_neutro']
    colores = {
        'Aburrido':     '#4C72B0',
        'Enojo':        '#DD8452',
        'Tranquilidad': '#55A868',
        'Tristeza':     '#C44E52',
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, score in zip(axes.ravel(), cols_score):
        for clase in CLASES:
            sub = df[df['clase_original'] == clase][score].values
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=15, alpha=0.55, label='{} (n={})'.format(clase, len(sub)),
                    color=colores[clase], edgecolor='white')
        ax.set_xlabel(score)
        ax.set_ylabel('# audios')
        ax.set_title('Distribucion de {}'.format(score))
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Scores acusticos por clase original (sin reasignar)', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print('\nGrafico guardado: {}'.format(output_path))


# --- Main ---
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=DATA_DIR,
                        help='directorio con las carpetas de audios')
    args = parser.parse_args()

    print('Analizando audios en {}...'.format(args.data_dir))
    df = analizar_dataset(args.data_dir)
    if len(df) == 0:
        print('[ERROR] No se proceso ningun audio.')
        sys.exit(1)

    imprimir_reporte(df)

    os.makedirs(os.path.dirname(REPORTE_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(REPORTE_PNG), exist_ok=True)

    df.to_csv(REPORTE_CSV, index=False)
    print('\nReporte CSV guardado: {}'.format(REPORTE_CSV))
    graficar_reporte(df, REPORTE_PNG)


if __name__ == '__main__':
    main()
