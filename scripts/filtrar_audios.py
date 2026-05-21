#!/usr/bin/env python3
"""
filtrar_audios.py
=================
Filtro de audios — orientado a clasificación binaria Enojo vs Tristeza,
con Tranquilidad como bucket para todo lo neutro/ambiguo.

Estrategia: respeta la etiqueta original (Enojo/Tristeza) y descarta
a Tranquilidad los audios sin firma emocional suficiente.
Aburrido siempre va a Tranquilidad (acústicamente indistinguible).

Uso (desde la raíz del proyecto):
    python scripts/filtrar_audios.py                  # dry-run, reporta
    python scripts/filtrar_audios.py --apply --copy   # copia archivos filtrados

Outputs:
    - outputs/reporte_filtrado_v2.csv
    - outputs/figuras/reporte_filtrado_v2.png
    - data/AUDIOS_FILTRADOS_V2/  (con --apply)
"""

import os
import sys
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

warnings.filterwarnings('ignore')

# ─── Configuración ────────────────────────────────────────────────
# Rutas relativas a la raiz del proyecto (ejecutar desde clasificador-audios/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(_ROOT, 'data', 'AUDIOS MACHINE LEARNING')
OUT_DIR     = os.path.join(_ROOT, 'data', 'AUDIOS_FILTRADOS_V2')
REPORTE_CSV = os.path.join(_ROOT, 'outputs', 'reporte_filtrado_v2.csv')
REPORTE_PNG = os.path.join(_ROOT, 'outputs', 'figuras', 'reporte_filtrado_v2.png')
CLASES_ORIGEN = ['Aburrido', 'Enojo', 'Tranquilidad', 'Tristeza']
EXTS = {'.ogg', '.mp3', '.mp4', '.mpeg', '.wav', '.flac', '.m4a'}
SR   = 22050

COLORES = {'Enojo':'#DD8452', 'Tristeza':'#4C72B0',
           'Tranquilidad':'#55A868', 'Descartado':'#999999'}


# ─── Métricas acústicas ───────────────────────────────────────────
def calcular_metricas(ruta):
    """
    Extrae métricas acústicas relevantes para distinguir Enojo, Tristeza y
    estados neutros.

    Optimización: carga solo 10s, SR=16000, hop=2048 → ~1-2s por audio.
    """
    try:
        y, _ = librosa.load(ruta, sr=SR, mono=True, duration=10.0)
        if len(y) < SR * 0.3:
            return None

        hop = 2048  # agresivo pero suficiente para estadísticas globales

        # --- Pitch (F0) con yin (rápido con hop grande) ---
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

        # --- Energía / RMS ---
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))
        energy_cv = rms_std / rms_mean if rms_mean > 1e-6 else 0.0
        rms_db = librosa.amplitude_to_db(rms + 1e-8)
        dynamic_range_db = float(np.percentile(rms_db, 95) -
                                  np.percentile(rms_db, 5))
        energy_level = float(np.mean(rms_db))

        # --- Tempo / dinámica temporal ---
        onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
        onset_rate = float(np.mean(onset_env))
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]
        zcr_mean = float(np.mean(zcr))

        # --- Fracción de silencio ---
        umb_silencio = 0.1 * np.max(rms) if np.max(rms) > 0 else 0
        frac_silencio = float(np.mean(rms < umb_silencio))

        # --- Brillo espectral (centroide) ---
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
        print(f'\n  [ERROR] {os.path.basename(ruta)}: {e}')
        return None


# ─── Scores específicos por emoción ───────────────────────────────
def _norm(x, lo, hi):
    """Mapea x desde [lo, hi] al rango [0, 1] saturando."""
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def score_enojo(m):
    """
    Firma acústica de enojo (literatura: Banse & Scherer 1996, Juslin & Laukka 2003):
      - Energía alta y dinámica
      - Pitch medio-alto con variabilidad
      - Más onsets por segundo (habla "atacada")
      - Brillo espectral alto
    """
    s_energia    = _norm(m['energy_level_db'],   -40.0, -22.0)   # dB
    s_dinamica   = _norm(m['dynamic_range_db'],   25.0,  50.0)
    s_pitch_alt  = _norm(m['pitch_mean_st'],      10.0,  24.0)   # st sobre A2
    s_pitch_var  = _norm(m['pitch_std_st'],        2.5,   8.0)
    s_onsets     = _norm(m['onset_rate'],          1.0,   4.0)
    s_brillo     = _norm(m['centroid_hz'],      1500.0, 3500.0)
    return (0.25*s_energia + 0.20*s_dinamica + 0.15*s_pitch_alt +
            0.15*s_pitch_var + 0.15*s_onsets + 0.10*s_brillo)


def score_tristeza(m):
    """
    Firma acústica de tristeza:
      - Energía baja
      - Pitch bajo y poco variable (monotonía)
      - Más fracción de silencio (pausas)
      - Pocos onsets, ritmo lento
      - Brillo espectral bajo
    """
    s_energia_baja = _norm(-m['energy_level_db'],  22.0,  40.0)
    s_pitch_bajo   = _norm(-m['pitch_mean_st'],    -5.0,  10.0)  # negativo = grave
    s_monotonia    = _norm(-m['pitch_std_st'],     -5.0,  -1.5)  # std baja = monótono
    s_silencios    = _norm(m['frac_silencio'],      0.10,  0.45)
    s_lentitud     = _norm(-m['onset_rate'],       -3.0,  -1.0)
    s_brillo_bajo  = _norm(-m['centroid_hz'],   -3000.0, -1200.0)
    return (0.20*s_energia_baja + 0.15*s_pitch_bajo + 0.20*s_monotonia +
            0.15*s_silencios + 0.15*s_lentitud + 0.15*s_brillo_bajo)


def score_neutro(m):
    """
    Score de "audio neutro / plano": baja expresividad sin firma clara
    de enojo ni tristeza. Útil para mandar a Tranquilidad.
    """
    s_pitch_plano  = _norm(-m['pitch_std_st'],     -4.0, -1.5)
    s_dinamica_baja= _norm(-m['dynamic_range_db'], -35.0, -20.0)
    s_energia_med  = 1.0 - abs(_norm(m['energy_level_db'], -40.0, -22.0) - 0.5)*2
    return 0.4*s_pitch_plano + 0.4*s_dinamica_baja + 0.2*s_energia_med


# ─── Decisión de etiqueta ─────────────────────────────────────────
def decidir_etiqueta(m, clase_original, umbral_emocion=0.35, margen_min=0.08):
    """
    Decide si un audio es útil o debe descartarse a Tranquilidad.

    Estrategia: respeta la etiqueta original (Enojo/Tristeza) y solo descarta
    a Tranquilidad si el audio no muestra firma emocional suficiente.
    Aburrido siempre va a Tranquilidad (clase ambigua que se fusiona con neutros).

    Reglas:
      - Aburrido → siempre Tranquilidad
      - Enojo original: si score_enojo < umbral → Tranquilidad (neutro)
      - Tristeza original: si score_tristeza < umbral → Tranquilidad (neutro)
      - Tranquilidad original: se mantiene si score_enojo y score_tristeza bajos,
        si alguno supera umbral+margen → sospechoso (va a Tranquilidad igual,
        pero se marca como dudoso)

    Devuelve (etiqueta_final, confianza, razon).
    """
    s_e = score_enojo(m)
    s_t = score_tristeza(m)
    s_n = score_neutro(m)

    if clase_original == 'Aburrido':
        return 'Tranquilidad', s_n, 'Aburrido → fusionado con neutros'

    if clase_original == 'Enojo':
        if s_e >= umbral_emocion:
            return 'Enojo', s_e, f'firma de enojo confirmada (s_e={s_e:.2f})'
        else:
            return 'Tranquilidad', s_n, f'enojo sin firma acústica (s_e={s_e:.2f} < {umbral_emocion})'

    if clase_original == 'Tristeza':
        if s_t >= umbral_emocion:
            return 'Tristeza', s_t, f'firma de tristeza confirmada (s_t={s_t:.2f})'
        else:
            return 'Tranquilidad', s_n, f'tristeza sin firma acústica (s_t={s_t:.2f} < {umbral_emocion})'

    # Tranquilidad original
    return 'Tranquilidad', s_n, f'neutro original (s_e={s_e:.2f}, s_t={s_t:.2f})'


# ─── Análisis del dataset ─────────────────────────────────────────
def analizar_dataset(data_dir, umbral, margen):
    filas = []
    # Contar total para progreso
    total = sum(1 for c in CLASES_ORIGEN
                if os.path.isdir(os.path.join(data_dir, c))
                for f in os.listdir(os.path.join(data_dir, c))
                if os.path.splitext(f)[1].lower() in EXTS)
    procesados = 0
    for clase in CLASES_ORIGEN:
        carpeta = os.path.join(data_dir, clase)
        if not os.path.isdir(carpeta):
            print(f'  [WARN] Carpeta no encontrada: {carpeta}', flush=True)
            continue
        for nombre in sorted(os.listdir(carpeta)):
            if os.path.splitext(nombre)[1].lower() not in EXTS:
                continue
            procesados += 1
            print(f'  [{procesados:3d}/{total}] {clase}/{nombre}', flush=True)
            ruta = os.path.join(carpeta, nombre)
            m = calcular_metricas(ruta)
            if m is None:
                continue
            etiqueta, conf, razon = decidir_etiqueta(m, clase, umbral, margen)
            m.update({
                'archivo':        nombre,
                'clase_original': clase,
                'recolector':     nombre[:2],
                'score_enojo':    score_enojo(m),
                'score_tristeza': score_tristeza(m),
                'score_neutro':   score_neutro(m),
                'destino':        etiqueta,
                'confianza':      conf,
                'razon':          razon,
                'cambia':         etiqueta != clase,
            })
            filas.append(m)
    return pd.DataFrame(filas)


# ─── Reportes ─────────────────────────────────────────────────────
def imprimir_reporte(df, binary=False):
    print('\n' + '='*72)
    print('REPORTE DE FILTRADO V2 — Etiquetado por evidencia acústica')
    print('='*72)
    print(f'\nTotal de audios analizados: {len(df)}')

    print('\nMatriz original → destino:')
    tabla = pd.crosstab(df['clase_original'], df['destino'])
    print(tabla.to_string())

    print('\nDistribución final:')
    dist = df['destino'].value_counts()
    for k, v in dist.items():
        print(f'  {k:<14} {v:3d}')

    cambios = df[df['cambia']]
    print(f'\nAudios que cambian de etiqueta: {len(cambios)}/{len(df)} '
          f'({100*len(cambios)/len(df):.1f}%)')

    print('\nReclasificados a Tranquilidad (top 20 por confianza):')
    rec = df[(df['cambia']) & (df['destino']=='Tranquilidad')].sort_values('confianza', ascending=False)
    if len(rec):
        cols = ['archivo','clase_original','score_enojo','score_tristeza','score_neutro','razon']
        print(rec[cols].head(20).round(3).to_string(index=False))
    else:
        print('  (ninguno)')

    print('\nAudios reasignados entre Enojo↔Tristeza (etiquetado dudoso):')
    cruce = df[(df['cambia']) & (df['destino'].isin(['Enojo','Tristeza'])) &
               (df['clase_original'].isin(['Enojo','Tristeza']))]
    if len(cruce):
        cols = ['archivo','clase_original','destino','score_enojo','score_tristeza']
        print(cruce[cols].round(3).to_string(index=False))
    else:
        print('  (ninguno)')


def graficar_reporte(df, output_path='reporte_filtrado_v2.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter score_enojo vs score_tristeza, coloreado por destino
    ax = axes[0]
    for destino, color in [('Enojo','#DD8452'),
                            ('Tristeza','#4C72B0'),
                            ('Tranquilidad','#55A868')]:
        sub = df[df['destino']==destino]
        ax.scatter(sub['score_enojo'], sub['score_tristeza'],
                   c=color, label=f'{destino} (n={len(sub)})',
                   s=60, alpha=0.75, edgecolors='white', linewidths=0.5)
    ax.plot([0,1],[0,1], '--', color='gray', alpha=0.4, label='diagonal')
    ax.set_xlabel('Score Enojo')
    ax.set_ylabel('Score Tristeza')
    ax.set_title('Decisión de etiqueta por evidencia acústica')
    ax.legend(loc='upper right', fontsize=9)
    ax.spines[['top','right']].set_visible(False)

    # Cuántos de cada clase original van a cada destino
    ax = axes[1]
    pivot = pd.crosstab(df['clase_original'], df['destino'])
    pivot = pivot.reindex(columns=['Enojo','Tristeza','Tranquilidad'], fill_value=0)
    pivot.plot(kind='bar', stacked=True, ax=ax,
               color=['#DD8452','#4C72B0','#55A868'])
    ax.set_xlabel('Carpeta original')
    ax.set_ylabel('# audios')
    ax.set_title('Migración de etiquetas')
    ax.legend(title='Destino', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

    fig.suptitle('Filtrado v2 — Evidencia acústica por emoción', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f'\nGráfico guardado: {output_path}')


def aplicar_movimiento(df, src_dir, out_dir, copiar=False, binary=False):
    if os.path.exists(out_dir):
        print(f'\n[WARN] La carpeta {out_dir} ya existe. Borra o renombra antes de --apply.')
        return
    destinos_validos = ['Enojo','Tristeza','Tranquilidad']
    for clase in destinos_validos:
        os.makedirs(os.path.join(out_dir, clase), exist_ok=True)

    op = shutil.copy2 if copiar else shutil.move
    accion = 'Copiando' if copiar else 'Moviendo'
    print(f'\n{accion} archivos a {out_dir}/...')

    n_movidos = 0
    for _, row in df.iterrows():
        src = os.path.join(src_dir, row['clase_original'], row['archivo'])
        dst = os.path.join(out_dir, row['destino'], row['archivo'])
        if not os.path.exists(src):
            continue
        op(src, dst)
        n_movidos += 1
    print(f'  {accion.lower()}: {n_movidos} archivos')


# ─── Main ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true',
                        help='aplicar el filtrado (mover/copiar archivos)')
    parser.add_argument('--copy', action='store_true',
                        help='con --apply, copiar en lugar de mover')
    parser.add_argument('--binary', action='store_true',
                        help='modo binario activo: Enojo vs Tristeza (Tranquilidad=descartes)')
    parser.add_argument('--umbral', type=float, default=0.35,
                        help='umbral mínimo del score emocional para conservar la etiqueta (default: 0.35)')
    parser.add_argument('--margen', type=float, default=0.08,
                        help='margen mínimo entre score_enojo y score_tristeza (default: 0.08)')
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--out-dir',  default=OUT_DIR)
    args = parser.parse_args()

    print(f'Analizando audios en {args.data_dir}...')
    print(f'  umbral_emocion = {args.umbral}, margen_min = {args.margen}')
    df = analizar_dataset(args.data_dir, args.umbral, args.margen)
    if len(df) == 0:
        print('[ERROR] No se procesó ningún audio.')
        sys.exit(1)

    imprimir_reporte(df, binary=args.binary)

    df.to_csv(REPORTE_CSV, index=False)
    print('\nReporte CSV guardado: ' + REPORTE_CSV)
    graficar_reporte(df, output_path=REPORTE_PNG)

    if args.apply:
        aplicar_movimiento(df, args.data_dir, args.out_dir,
                           copiar=args.copy, binary=args.binary)
        print(f'\n✓ Estructura nueva en {args.out_dir}/')
    else:
        print('\nModo dry-run. Para aplicar:')
        print('  python3 filtrar_audios_v2.py --apply --copy')


if __name__ == '__main__':
    main()
