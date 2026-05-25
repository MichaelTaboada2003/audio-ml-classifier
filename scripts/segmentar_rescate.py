#!/usr/bin/env python3
"""
segmentar_rescate.py
====================
Extrae, de cada audio (Enojo / Feliz / Tranquilidad / Tristeza), el SEGMENTO de
10 s con el score mas alto para su propia clase, en vez de usar siempre los
primeros 10 s. Sirve para construir un dataset localizado (cada audio recortado
a su tramo mas emotivo). Por defecto procesa las 4 clases; usa --clase para una.

Motivacion: el pipeline actual scorea cada audio sobre los primeros 10 s
(MAX_DURATION), pero la emocion suele estar mas adelante en el clip. Aqui
pasamos una ventana deslizante de 10 s sobre el audio COMPLETO, scoreamos cada
ventana con el modelo emocional audeering (mismo mapeo AVD->clase que la
auditoria) y nos quedamos con la ventana de mayor score para la clase. Ese
mejor segmento de 10 s se recorta a un .wav listo para entrenar.

Estados por audio (informativos, van al CSV):
  - OK          el clip de los primeros 10 s ya suena a su clase.
  - RESCATADO   los primeros 10 s NO suenan a la clase, pero otra ventana de 10 s si.
  - SIN_RESCATE ninguna ventana de 10 s suena a la clase -> probablemente mal etiquetado.

Por defecto se recorta el mejor segmento de 10 s de TODOS los audios. El estado
solo informa cuanto aporta segmentar respecto a usar los primeros 10 s.

Salida:
  - outputs/mejores_segmentos.csv      (un row por audio; reporte)
  - data/procesado/<Clase>/*.wav       (mejor segmento de 10 s; dataset de entrenamiento, no toca originales)

Uso:
    python scripts/segmentar_rescate.py                      # Enojo + Feliz, recorta el mejor 10s de cada uno
    python scripts/segmentar_rescate.py --clase Enojo        # solo una clase
    python scripts/segmentar_rescate.py --win 8 --hop 2      # ventana 8s, paso 2s
    python scripts/segmentar_rescate.py --solo-rescatables   # recorta solo los RESCATADO
    python scripts/segmentar_rescate.py --solo-reporte       # no escribe recortes
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from emotion_encoder import load_encoder, extract_avd, MODEL_ID  # noqa: E402
# Fuente unica del mapeo AVD->clase (no duplicar): vive en la auditoria.
from auditar_etiquetas_ia import EMOTION_POINTS, score_clase  # noqa: E402

DATA_DIR = ROOT / "data" / "AUDIOS MACHINE LEARNING"
OUT_CSV = ROOT / "outputs" / "mejores_segmentos.csv"
OUT_DIR = ROOT / "data" / "procesado"   # dataset de entrenamiento (lo lee entrenar_procesado.py)
SR = 16000
MAX_DURATION = 10.0   # baseline: lo que ve el pipeline actual (primeros 10 s)
MIN_SEG = 1.0         # un segmento mas corto que esto no se scorea (el encoder necesita senal)
EXTS = {".ogg", ".mp3", ".mp4", ".mpeg", ".wav", ".flac", ".m4a"}
# Clases con punto teorico en EMOTION_POINTS (borderline queda fuera: no tiene clase propia).
CLASES = ["Enojo", "Feliz", "Tranquilidad", "Tristeza"]


def ventanas(n: int, win: int, hop: int) -> list[tuple[int, int]]:
    """Genera (inicio, fin) en samples. Garantiza ventanas de largo `win`
    y que la ultima cubra el final del audio."""
    if n <= win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] != n - win:
        starts.append(n - win)
    return [(s, s + win) for s in starts]


def scores_segmento(seg, proc, model, device) -> tuple[dict, str]:
    """Devuelve (scores_por_clase, clase_argmax) de un segmento de audio."""
    arousal, dominance, valence = extract_avd(seg, SR, proc, model, device)
    avd = (arousal, valence, dominance)
    scores = {c: score_clase(avd, c) for c in EMOTION_POINTS}
    return scores, max(scores, key=scores.get)


def guardar_wav(path: Path, y, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(path), y, sr)
    except Exception:
        from scipy.io import wavfile
        wavfile.write(str(path), sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clase", choices=CLASES, default=None,
                        help="Escanear solo esta clase (default: las 4 clases).")
    parser.add_argument("--win", type=float, default=10.0,
                        help="Largo de ventana en segundos (default 10.0 = el que usa el pipeline).")
    parser.add_argument("--hop", type=float, default=5.0,
                        help="Paso entre ventanas en segundos (default 5.0).")
    parser.add_argument("--rescate-min", type=float, default=0.45, dest="rescate_min",
                        help="Score minimo de la clase propia para marcar una ventana como confirmatoria (default 0.45).")
    parser.add_argument("--solo-rescatables", action="store_true", dest="solo_rescatables",
                        help="Recortar solo los audios RESCATADO (default: recorta el mejor segmento de todos).")
    parser.add_argument("--solo-reporte", action="store_true",
                        help="No escribir recortes de audio, solo el CSV + terminal.")
    args = parser.parse_args()

    clases = [args.clase] if args.clase else CLASES
    win = int(args.win * SR)
    hop = int(args.hop * SR)

    print(f"Cargando {MODEL_ID} ...")
    t0 = time.time()
    proc, model, device = load_encoder(prefer_mps=True)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s sobre {device}")
    print(f"  Ventana={args.win}s  paso={args.hop}s  umbral rescate={args.rescate_min}")

    rows = []
    for clase in clases:
        carpeta = DATA_DIR / clase
        if not carpeta.is_dir():
            print(f"  [SKIP] no existe carpeta {carpeta}")
            continue
        archivos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTS)
        print(f"\n>>> {clase}: {len(archivos)} audios")
        for ruta in archivos:
            try:
                y, _ = librosa.load(str(ruta), sr=SR, mono=True)  # audio completo
            except Exception as e:
                print(f"    ERROR {ruta.name}: {e}")
                continue
            dur = len(y) / SR

            # Baseline: lo que ve el pipeline actual (primeros 10 s).
            base = y[: int(MAX_DURATION * SR)]
            sc_full, suena_full = scores_segmento(base, proc, model, device)
            score_full = sc_full[clase]

            # Barrido de segmentos sobre el audio completo.
            segs = ventanas(len(y), win, hop)
            best_overall = None   # (score_propio, ini_s, fin_s, suena_seg)  -> mejor score propio (cualquiera)
            best_conf = None      # idem pero solo donde argmax == clase y score >= umbral
            for s, e in segs:
                seg = y[s:e]
                if len(seg) < SR * MIN_SEG:
                    continue
                sc, suena = scores_segmento(seg, proc, model, device)
                sp = sc[clase]
                cand = (sp, s / SR, e / SR, suena)
                if best_overall is None or sp > best_overall[0]:
                    best_overall = cand
                if suena == clase and sp >= args.rescate_min:
                    if best_conf is None or sp > best_conf[0]:
                        best_conf = cand

            # Estado.
            if suena_full == clase:
                estado = "OK"
            elif best_conf is not None:
                estado = "RESCATADO"
            else:
                estado = "SIN_RESCATE"

            # Segmento de referencia = la ventana de mayor score para la clase.
            seg_score, seg_ini, seg_fin, suena_seg = best_overall

            # Recorte: el mejor segmento de todos por defecto; solo los RESCATADO si --solo-rescatables.
            recorte_path = ""
            do_cut = (not args.solo_reporte) and (not args.solo_rescatables or estado == "RESCATADO")
            if do_cut:
                s_idx, e_idx = int(seg_ini * SR), int(seg_fin * SR)
                out = OUT_DIR / clase / f"{ruta.stem}_top{seg_ini:.0f}-{seg_fin:.0f}s.wav"
                guardar_wav(out, y[s_idx:e_idx], SR)
                recorte_path = str(out.relative_to(ROOT))

            rows.append({
                "archivo": ruta.name,
                "clase": clase,
                "dur_s": round(dur, 1),
                "n_seg": len(segs),
                "score_full": round(score_full, 3),
                "suena_full": suena_full,
                "best_seg_score": round(seg_score, 3),
                "seg_ini_s": round(seg_ini, 1),
                "seg_fin_s": round(seg_fin, 1),
                "suena_seg": suena_seg,
                "mejora": round(seg_score - score_full, 3),
                "estado": estado,
                "recorte": recorte_path,
            })
            print(f"    {ruta.name:<22} {estado:<11} full={score_full:.2f}({suena_full[:4]}) "
                  f"best={seg_score:.2f}({suena_seg[:4]}) @[{seg_ini:.0f}-{seg_fin:.0f}s]")

    if not rows:
        print("\nNo se proceso ningun audio.")
        return

    # CSV ordenado: primero por clase, dentro por mejora descendente (lo mas rescatable arriba).
    rows.sort(key=lambda r: (r["clase"], -r["mejora"]))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Resumen.
    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for clase in clases:
        sub = [r for r in rows if r["clase"] == clase]
        if not sub:
            continue
        ok = sum(1 for r in sub if r["estado"] == "OK")
        resc = sum(1 for r in sub if r["estado"] == "RESCATADO")
        sin = sum(1 for r in sub if r["estado"] == "SIN_RESCATE")
        print(f"  {clase:<8} total={len(sub):3d}  OK={ok:3d}  RESCATADO={resc:3d}  SIN_RESCATE={sin:3d}")
    print(f"\nCSV: {OUT_CSV.relative_to(ROOT)}")
    n_cortes = sum(1 for r in rows if r["recorte"])
    if n_cortes:
        print(f"Recortes escritos: {n_cortes} en {OUT_DIR.relative_to(ROOT)}/<Clase>/")
    elif not args.solo_reporte:
        print("No hubo audios RESCATADO que recortar.")


if __name__ == "__main__":
    main()
