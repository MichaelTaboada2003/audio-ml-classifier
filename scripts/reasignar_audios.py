#!/usr/bin/env python3
"""
reasignar_audios.py
====================
Re-etiqueta audios sospechosos detectados por scripts/auditar_etiquetas_ia.py
moviendolos fisicamente entre carpetas segun la prediccion del modelo emocional
audeering. Guarda un manifest reversible en outputs/reasignacion_log.json.

Uso:
    python scripts/reasignar_audios.py                  # dry-run (default)
    python scripts/reasignar_audios.py --apply          # ejecutar movimientos
    python scripts/reasignar_audios.py --apply --umbral 0.40   # umbral distinto
    python scripts/reasignar_audios.py --revert         # deshacer ultimo apply

Notas:
- Solo mueve audios con delta > umbral en outputs/audios_sospechosos_ia.csv.
- El manifest guarda origen, destino, delta y scores A/V/D por audio.
- Tras revertir, el manifest se archiva con timestamp.
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_ROOT, "data", "AUDIOS MACHINE LEARNING")
CSV_IN = os.path.join(_ROOT, "outputs", "audios_sospechosos_ia.csv")
LOG_PATH = os.path.join(_ROOT, "outputs", "reasignacion_log.json")
UMBRAL_DEFAULT = 0.25


def cargar_movimientos(umbral):
    if not os.path.exists(CSV_IN):
        sys.exit(f"No existe {CSV_IN}. Corre primero scripts/auditar_etiquetas_ia.py")
    movs = []
    with open(CSV_IN, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["delta"] == "":
                continue
            d = float(r["delta"])
            if d <= umbral:
                continue
            movs.append({
                "archivo": r["archivo"],
                "origen": r["clase_etiqueta"],
                "destino": r["suena_a"],
                "delta": round(d, 3),
                "arousal": float(r["arousal"]),
                "valence": float(r["valence"]),
                "dominance": float(r["dominance"]),
                "score_origen": float(r["score_propio"]),
                "score_destino": float(r["score_suena_a"]),
            })
    movs.sort(key=lambda r: -r["delta"])
    return movs


def aplicar(movs):
    realizados, errores = [], []
    for mv in movs:
        src = os.path.join(DATA, mv["origen"], mv["archivo"])
        dst = os.path.join(DATA, mv["destino"], mv["archivo"])
        if not os.path.exists(src):
            errores.append((mv["archivo"], f"origen no existe en {mv['origen']}/"))
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            errores.append((mv["archivo"], f"destino {mv['destino']}/ ya tiene un archivo igual"))
            continue
        shutil.move(src, dst)
        realizados.append(mv)
    return realizados, errores


def guardar_log(realizados, umbral):
    payload = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "umbral_delta": umbral,
        "total_movidos": len(realizados),
        "movimientos": realizados,
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Log guardado: {LOG_PATH}")


def revertir():
    if not os.path.exists(LOG_PATH):
        sys.exit(f"No hay manifest en {LOG_PATH}")
    with open(LOG_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    print(f"Manifest del {payload['fecha']}, {payload['total_movidos']} archivos a revertir")
    revertidos, errores = [], []
    for mv in payload["movimientos"]:
        # Para revertir, origen y destino se invierten
        src = os.path.join(DATA, mv["destino"], mv["archivo"])
        dst = os.path.join(DATA, mv["origen"], mv["archivo"])
        if not os.path.exists(src):
            errores.append((mv["archivo"], "no esta en destino — ya revertido?"))
            continue
        if os.path.exists(dst):
            errores.append((mv["archivo"], "origen ya tiene un archivo igual"))
            continue
        shutil.move(src, dst)
        revertidos.append(mv["archivo"])
    print(f"  Revertidos: {len(revertidos)}/{payload['total_movidos']}")
    if errores:
        print(f"  Errores ({len(errores)}):")
        for f, m in errores[:10]:
            print(f"    {f}: {m}")
    if revertidos:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        archived = LOG_PATH.replace(".json", f"_revertido_{ts}.json")
        os.rename(LOG_PATH, archived)
        print(f"  Manifest archivado: {archived}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Aplicar movimientos (default: dry-run)")
    ap.add_argument("--revert", action="store_true", help="Revertir el ultimo apply usando el manifest")
    ap.add_argument("--umbral", type=float, default=UMBRAL_DEFAULT, help=f"Umbral de delta (default {UMBRAL_DEFAULT})")
    args = ap.parse_args()

    if args.revert:
        revertir()
        return

    movs = cargar_movimientos(args.umbral)
    print(f"Umbral: delta > {args.umbral}")
    print(f"Audios a mover: {len(movs)}\n")

    pairs = Counter((m["origen"], m["destino"]) for m in movs)
    print("Resumen por direccion (origen -> destino):")
    for (o, d), n in pairs.most_common():
        print(f"  {o:14s} -> {d:14s}  {n} audios")

    if not args.apply:
        print("\n--- DRY RUN. Pasa --apply para ejecutar de verdad. ---\n")
        print("Detalle (top 30):")
        for mv in movs[:30]:
            print(f"  {mv['archivo']:30s}  {mv['origen']:14s} -> {mv['destino']:14s}  Δ={mv['delta']:.3f}")
        if len(movs) > 30:
            print(f"  ... y {len(movs)-30} mas")
        return

    print("\nAplicando movimientos...")
    realizados, errores = aplicar(movs)
    print(f"  Movidos: {len(realizados)}/{len(movs)}")
    if errores:
        print(f"  Errores ({len(errores)}):")
        for f, m in errores[:10]:
            print(f"    {f}: {m}")
    guardar_log(realizados, args.umbral)

    # Snapshot final por clase
    print("\nConteo final por carpeta:")
    EXTS = {".ogg", ".mp3", ".mp4", ".mpeg", ".wav", ".flac", ".m4a"}
    for c in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, c)
        if not os.path.isdir(p):
            continue
        n = sum(1 for f in os.listdir(p) if os.path.splitext(f)[1].lower() in EXTS)
        print(f"  {c:14s} {n}")


if __name__ == "__main__":
    main()
