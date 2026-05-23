#!/usr/bin/env python3
"""
auditar_etiquetas_ia.py
=======================
Audita las etiquetas humanas del dataset usando un modelo de IA pre-entrenado para
Speech Emotion Recognition (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim).

A diferencia del filtro acustico basado en 7 metricas globales (energia, pitch, brillo,
etc.), este modelo captura prosodia fina (entonacion, micro-tension) y es robusto
cross-language. Lo usamos como "oraculo de etiquetas" — no para clasificar, sino
para detectar audios cuya etiqueta humana no concuerda con lo que el audio realmente
transmite emocionalmente.

Salida (3 dimensiones continuas en [0, 1]):
  - arousal:   activacion / energia emocional  (alto = enojo o feliz, bajo = triste o calmo)
  - valence:   positivo / negativo             (alto = feliz o calmo, bajo = enojo o triste)
  - dominance: control / sumision              (alto = enojo, bajo = triste / sumiso)

Mapeo a las 4 clases del proyecto (kernel gaussiano centrado en cada emocion):
  - Enojo:        (A, V, D) cerca de (0.80, 0.20, 0.75)
  - Feliz:        cerca de (0.75, 0.85, 0.60)
  - Tranquilidad: cerca de (0.30, 0.60, 0.50)
  - Tristeza:     cerca de (0.20, 0.25, 0.30)

Para cada audio se calcula score por clase. Si la clase con mayor score no coincide
con la etiqueta humana, el audio es candidato a estar mal etiquetado.

Uso:
    python scripts/auditar_etiquetas_ia.py

Output:
    outputs/audios_sospechosos_ia.csv  (un row por audio, ordenado por delta descendente)
"""

import os
import sys
import csv
import time
import warnings
from collections import Counter

import numpy as np
import librosa
import torch

warnings.filterwarnings("ignore")

# Modelo emocional compartido — definicion en raiz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emotion_encoder import load_encoder, extract_avd, MODEL_ID


# --- Puntos teoricos en (arousal, valence, dominance) ---
EMOTION_POINTS = {
    "Enojo":        {"center": (0.80, 0.20, 0.75), "sigma": (0.20, 0.25, 0.20)},
    "Feliz":        {"center": (0.75, 0.85, 0.60), "sigma": (0.20, 0.20, 0.25)},
    "Tranquilidad": {"center": (0.30, 0.60, 0.50), "sigma": (0.20, 0.25, 0.25)},
    "Tristeza":     {"center": (0.20, 0.25, 0.30), "sigma": (0.20, 0.20, 0.25)},
}


def score_clase(avd, clase):
    """Score gaussiano: 1.0 si avd coincide con centro, decae con la distancia."""
    a, v, d = avd
    ca, cv, cd = EMOTION_POINTS[clase]["center"]
    sa, sv, sd = EMOTION_POINTS[clase]["sigma"]
    z2 = ((a - ca) / sa) ** 2 + ((v - cv) / sv) ** 2 + ((d - cd) / sd) ** 2
    return float(np.exp(-z2 / 2))


def main():
    # --- Config ---
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = os.path.join(_ROOT, "data", "AUDIOS MACHINE LEARNING")
    OUT_CSV = os.path.join(_ROOT, "outputs", "audios_sospechosos_ia.csv")
    SR = 16000
    MAX_DUR = 10.0
    EXTS = {".ogg", ".mp3", ".mp4", ".mpeg", ".wav", ".flac", ".m4a"}

    print(f"Cargando {MODEL_ID} (descarga ~1.2 GB en primera ejecucion)...")
    t0 = time.time()
    processor, model, device = load_encoder(prefer_mps=True)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s sobre {device}")

    # --- Recolectar audios ---
    audios = []
    for clase in sorted(os.listdir(DATA)):
        carpeta = os.path.join(DATA, clase)
        if not os.path.isdir(carpeta):
            continue
        for archivo in sorted(os.listdir(carpeta)):
            if os.path.splitext(archivo)[1].lower() in EXTS:
                audios.append((clase, archivo))
    print(f"  Total audios: {len(audios)}")

    # --- Inferencia ---
    rows = []
    t_start = time.time()
    for i, (clase, archivo) in enumerate(audios, 1):
        ruta = os.path.join(DATA, clase, archivo)
        try:
            y, _ = librosa.load(ruta, sr=SR, mono=True, duration=MAX_DUR)
            if len(y) < SR * 1.0:
                print(f"  [{i:3d}/{len(audios)}] SKIP {clase}/{archivo} (demasiado corto)")
                continue
            arousal, dominance, valence = extract_avd(y, SR, processor, model, device)

            scores = {c: score_clase((arousal, valence, dominance), c) for c in EMOTION_POINTS}
            score_propio = scores.get(clase, None)
            otras = {k: v for k, v in scores.items() if k != clase}
            suena_a = max(otras, key=otras.get) if otras else None
            score_otra = otras[suena_a] if suena_a else None
            delta = (score_otra - score_propio) if (score_propio is not None and score_otra is not None) else None

            rows.append({
                "archivo": archivo,
                "clase_etiqueta": clase,
                "recolector": archivo[:2],
                "arousal": round(arousal, 3),
                "valence": round(valence, 3),
                "dominance": round(dominance, 3),
                "score_Enojo": round(scores["Enojo"], 3),
                "score_Feliz": round(scores["Feliz"], 3),
                "score_Tranquilidad": round(scores["Tranquilidad"], 3),
                "score_Tristeza": round(scores["Tristeza"], 3),
                "score_propio": round(score_propio, 3) if score_propio is not None else "",
                "suena_a": suena_a or "",
                "score_suena_a": round(score_otra, 3) if score_otra is not None else "",
                "delta": round(delta, 3) if delta is not None else "",
            })

            if i % 20 == 0:
                rate = i / (time.time() - t_start)
                eta = (len(audios) - i) / rate
                print(f"  [{i:3d}/{len(audios)}] {clase}/{archivo}  A={arousal:.2f} V={valence:.2f} D={dominance:.2f}  ETA {eta:.0f}s")
        except Exception as e:
            print(f"  [{i:3d}/{len(audios)}] ERROR {clase}/{archivo}: {e}")

    print(f"\nProcesados: {len(rows)}/{len(audios)} en {time.time()-t_start:.1f}s")

    # --- Guardar CSV ordenado por delta descendente ---
    def sort_key(r):
        d = r["delta"]
        if d == "" or d is None:
            return (1, 0)
        return (0, -d)

    rows_sorted = sorted(rows, key=sort_key)
    cols = list(rows_sorted[0].keys())
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_sorted)
    print(f"CSV guardado: {OUT_CSV}")

    # --- Resumen ---
    print("\n" + "=" * 82)
    print("RESUMEN AUDITORIA IA — clase etiquetada vs lo que el modelo emocional detecta")
    print("=" * 82)
    for c in ["Enojo", "Tristeza", "Feliz", "Tranquilidad", "Aburrido"]:
        sub = [r for r in rows if r["clase_etiqueta"] == c]
        if not sub:
            continue
        if c == "Aburrido":
            # Aburrido: ver a que sueno mas
            suena = Counter(r["suena_a"] for r in sub if r["suena_a"])
            print(f"  {c:14s} total={len(sub):3d}  (sin score teorico, suenan a: {dict(suena)})")
            continue
        deltas = [r["delta"] for r in sub if r["delta"] != ""]
        sosp_fuerte = sum(1 for d in deltas if d > 0.15)
        sosp_leve = sum(1 for d in deltas if 0 < d <= 0.15)
        ok = sum(1 for d in deltas if d <= 0)
        print(f"  {c:14s} total={len(sub):3d}  OK={ok:3d}  borderline={sosp_leve:3d}  SOSPECHOSO={sosp_fuerte:3d}")
        if sosp_fuerte > 0:
            ttop = Counter(r["suena_a"] for r in sub if r["delta"] != "" and r["delta"] > 0.15)
            print(f"                 suenan a: {dict(ttop)}")


if __name__ == "__main__":
    main()
