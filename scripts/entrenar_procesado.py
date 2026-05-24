#!/usr/bin/env python3
"""
entrenar_procesado.py
=====================
Entrena los clasificadores sobre el dataset de segmentos localizados
(data/procesado) y reporta DOS metricas lado a lado para las clases en
config.ACTIVE_CLASSES:

  1. LOOCV sobre SEGMENTOS    — entrena y testea con recortes. Optimista
     (los segmentos se eligieron con audeering; medir sobre ellos infla).
  2. LOOCV HONESTO            — entrena con los segmentos de los OTROS audios
     y testea sobre el AUDIO CRUDO (primeros 10 s) del que se dejo fuera. El
     test nunca se segmenta -> es el numero que reflejaria produccion.

La diferencia entre ambas columnas es la "prima de cherry-picking": si la
honesta acompana a la de segmentos, la localizacion ayudo de verdad; si solo
sube la de segmentos, fue inflacion. Ver [[project_localizacion_segmentos]].

NO serializa modelos ni toca el pipeline canonico (exportar_modelos.py) — es
solo evaluacion. Reutiliza MODEL_SPECS y el encoder de exportar_modelos.

Uso:
    python scripts/entrenar_procesado.py                 # usa config.ACTIVE_CLASSES
    python scripts/entrenar_procesado.py --rebuild       # fuerza recalcular embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from config import ACTIVE_CLASSES
from emotion_encoder import load_encoder
from exportar_modelos import MODEL_SPECS, extraer_embedding

PROC_DIR = ROOT / "data" / "procesado"
ORIG_DIR = ROOT / "data" / "AUDIOS MACHINE LEARNING"
CACHE = ROOT / "outputs" / "proc_embeddings.npz"
MODEL_DIR = ROOT / "outputs" / "modelos"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
V2_CACHE = ROOT / "outputs" / "embeddings_v2.npz"   # cache que lee app.py (TRAINING_FILES)
EXTS = {".ogg", ".mp3", ".mp4", ".mpeg", ".wav", ".flac", ".m4a"}


def stem_de(nombre_seg: str) -> str:
    """'ED_01_FEL_top5-15s.wav' -> 'ED_01_FEL' (nombre del audio original)."""
    return nombre_seg.rsplit("_top", 1)[0]


def original_de(stem: str, clase: str) -> Path | None:
    for p in (ORIG_DIR / clase).glob(stem + ".*"):
        if p.suffix.lower() in EXTS:
            return p
    return None


def construir_cache(rebuild: bool = False):
    """Embeddings de TODOS los segmentos de data/procesado y de su audio crudo.

    Devuelve (Xseg, Xraw, y, files). Cachea en outputs/proc_embeddings.npz y
    reutiliza si el conjunto de segmentos en disco no cambio.
    """
    clases_disco = sorted(d.name for d in PROC_DIR.iterdir() if d.is_dir())
    segs_disco = sorted(
        f"{clase}/{p.name}"
        for clase in clases_disco
        for p in (PROC_DIR / clase).glob("*.wav")
    )

    if CACHE.exists() and not rebuild:
        c = np.load(CACHE, allow_pickle=True)
        if set(c["files"].tolist()) == set(segs_disco):
            print(f"Usando cache: {CACHE.relative_to(ROOT)}  ({len(c['files'])} audios)")
            return c["Xseg"], c["Xraw"], c["y"], c["files"]
        print("Cache no coincide con disco. Recalculando...")

    print(f"Cargando encoder...")
    t0 = time.time()
    proc, model, device = load_encoder(prefer_mps=True)
    print(f"  Encoder listo en {time.time() - t0:.1f}s sobre {device}")

    Xseg, Xraw, y, files = [], [], [], []
    t_start = time.time()
    for i, rel in enumerate(segs_disco, 1):
        clase, nombre = rel.split("/", 1)
        seg_path = PROC_DIR / clase / nombre
        orig_path = original_de(stem_de(nombre), clase)
        if orig_path is None:
            print(f"  [{i:3d}/{len(segs_disco)}] SKIP {rel}: sin original")
            continue
        emb_seg = extraer_embedding(str(seg_path), proc, model, device)
        emb_raw = extraer_embedding(str(orig_path), proc, model, device)
        if emb_seg is None or emb_raw is None:
            print(f"  [{i:3d}/{len(segs_disco)}] SKIP {rel}: audio corto")
            continue
        Xseg.append(emb_seg)
        Xraw.append(emb_raw)
        y.append(clase)
        files.append(rel)
        if i % 25 == 0:
            print(f"  [{i:3d}/{len(segs_disco)}] {rel}")

    Xseg = np.vstack(Xseg); Xraw = np.vstack(Xraw)
    y = np.array(y); files = np.array(files)
    np.savez(CACHE, Xseg=Xseg, Xraw=Xraw, y=y, files=files)
    print(f"Cache guardado: {CACHE.relative_to(ROOT)}  ({len(y)} audios, {time.time()-t_start:.0f}s)")
    return Xseg, Xraw, y, files


def loocv(builder, X_train_src, X_test_src, y, clases):
    """LOOCV: en cada fold entrena con X_train_src[otros] y testea con X_test_src[i].

    Si X_train_src is X_test_src -> LOOCV clasico. Si difieren (segmento vs crudo)
    -> evaluacion honesta sin leakage (el audio de test nunca esta en train)."""
    loo = LeaveOneOut()
    y_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in loo.split(X_train_src):
        i = test_idx[0]
        pipe = builder()
        pipe.fit(X_train_src[train_idx], y[train_idx])
        y_pred[i] = pipe.predict(X_test_src[i].reshape(1, -1))[0]
    acc = accuracy_score(y, y_pred)
    bal = balanced_accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, labels=clases, average=None, zero_division=0)
    return acc, bal, dict(zip(clases, rec))


def leave_one_collector_out(builder, Xseg, Xraw, y, recs, collectors, clases):
    """Entrena con los segmentos de TODOS los recolectores menos uno y testea sobre
    el AUDIO CRUDO del recolector dejado fuera. Es el test de generalizacion mas duro:
    el recolector de test aporta condiciones de grabacion/hablantes nunca vistos en
    training."""
    y_pred = np.empty(len(y), dtype=object)
    for c in collectors:
        test = recs == c
        if not test.any():
            continue
        pipe = builder()
        pipe.fit(Xseg[~test], y[~test])
        y_pred[test] = pipe.predict(Xraw[test])
    bal = balanced_accuracy_score(y, y_pred)
    rec = recall_score(y, y_pred, labels=clases, average=None, zero_division=0)
    return bal, dict(zip(clases, rec))


def guardar_artefactos(Xseg, y, files_rel, clases, resultados):
    """Serializa los 6 modelos (entrenados con TODOS los segmentos de las clases
    activas), escribe model_metrics.json con la metrica HONESTA como headline, y
    actualiza embeddings_v2.npz para que la app y el explorador holdout funcionen."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    modelos = {}
    for key, spec in MODEL_SPECS.items():
        pipe = spec["builder"]()
        pipe.fit(Xseg, y)
        joblib.dump(pipe, MODEL_DIR / f"{key}.joblib")
        r = resultados[key]
        modelos[key] = {
            "label": spec["label"],
            "description": spec["description"],
            "accuracy": round(r["acc_h"], 4),               # honesta (test = audio crudo)
            "balanced_accuracy": round(r["bal_h"], 4),       # honesta = headline
            "balanced_accuracy_loocv_seg": round(r["bal_s"], 4),  # LOOCV sobre segmentos (optimista)
            "recall_por_clase": {c: round(v, 4) for c, v in r["rec_h"].items()},
        }
    best = max(modelos, key=lambda k: modelos[k]["balanced_accuracy"])
    counts = {c: int(np.sum(y == c)) for c in clases}
    metrics = {
        "active_classes": list(clases),
        "dataset_source": "data/procesado (segmentos localizados de 10 s)",
        "metric": "balanced_accuracy = holdout honesto (train=segmento, test=audio crudo 10 s, LOOCV)",
        "class_counts": counts,
        "samples_per_class": int(min(counts.values())),
        "dataset_size": int(len(y)),
        "chance_accuracy": round(1.0 / len(clases), 4),
        "models": modelos,
        "best_model": best,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Cache que lee app.py: nombres de SEGMENTO (los originales no coinciden -> el
    # explorador muestra todos los audios crudos como holdout testeable).
    files_base = np.array([rel.split("/", 1)[1] for rel in files_rel])
    rec = np.array([fb[:2] for fb in files_base])
    np.savez(V2_CACHE, X=Xseg, y=y, rec=rec, files=files_base)

    print(f"\nGuardado: {len(MODEL_SPECS)} modelos en {MODEL_DIR.relative_to(ROOT)}/")
    print(f"          {METRICS_PATH.relative_to(ROOT)}  (best={best}, bal_acc honesta={modelos[best]['balanced_accuracy']})")
    print(f"          {V2_CACHE.relative_to(ROOT)}  ({len(y)} embeddings de segmento)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="Recalcular embeddings.")
    parser.add_argument("--clases", nargs="+", default=None,
                        help="Clases a evaluar (default: config.ACTIVE_CLASSES).")
    parser.add_argument("--guardar", action="store_true",
                        help="Serializar .joblib + model_metrics.json + embeddings_v2.npz (modelo activo de la app).")
    parser.add_argument("--loco", action="store_true",
                        help="Leave-one-collector-out: test de generalizacion a recolectores no vistos.")
    args = parser.parse_args()

    Xseg, Xraw, y, files = construir_cache(rebuild=args.rebuild)

    # Filtrar a las clases activas.
    clases = list(args.clases) if args.clases else list(ACTIVE_CLASSES)
    mask = np.isin(y, clases)
    Xseg, Xraw, y, files = Xseg[mask], Xraw[mask], y[mask], files[mask]
    counts = {c: int(np.sum(y == c)) for c in clases}

    print()
    print("=" * 92)
    print(f"EXPERIMENTO: {' vs '.join(clases)}  (data/procesado)")
    print(f"  Segmentos por clase: {counts}   chance bal_acc = {1/len(clases):.3f}")
    print("=" * 92)

    rec_cols = "".join(f"{('rec_'+c)[:9]:>10}" for c in clases)
    print(f"{'':<14}│{'LOOCV sobre SEGMENTOS':^{10+10*len(clases)}}│{'LOOCV HONESTO (test=crudo)':^{10+10*len(clases)}}")
    print(f"{'Modelo':<14}│{'bal_acc':>10}{rec_cols}│{'bal_acc':>10}{rec_cols}")
    print("─" * 92)

    resultados = {}
    for key, spec in MODEL_SPECS.items():
        acc_s, bal_s, rec_s = loocv(spec["builder"], Xseg, Xseg, y, clases)
        acc_h, bal_h, rec_h = loocv(spec["builder"], Xseg, Xraw, y, clases)
        resultados[key] = {"acc_h": acc_h, "bal_s": bal_s, "rec_s": rec_s,
                           "bal_h": bal_h, "rec_h": rec_h}
        fila_s = f"{bal_s:>10.3f}" + "".join(f"{rec_s[c]:>10.3f}" for c in clases)
        fila_h = f"{bal_h:>10.3f}" + "".join(f"{rec_h[c]:>10.3f}" for c in clases)
        print(f"{spec['label']:<14}│{fila_s}│{fila_h}")

    print("─" * 92)
    print("Lectura: si la columna HONESTA acompana a la de SEGMENTOS, la localizacion")
    print("ayudo. Si solo sube la de SEGMENTOS, fue inflacion (la prima de cherry-picking).")

    if args.loco:
        recs = np.array([rel.split("/", 1)[1][:2] for rel in files])
        collectors = sorted(set(recs.tolist()))
        print()
        print("=" * 92)
        print("LEAVE-ONE-COLLECTOR-OUT  (train = segmentos de otros recolectores, test = audio CRUDO del que falta)")
        print("Audios de test por recolector:")
        for c in collectors:
            d = {cl: int(((recs == c) & (y == cl)).sum()) for cl in clases}
            print(f"  {c}: {d}")
        print("─" * 92)
        rec_cols = "".join(f"{('rec_'+c)[:9]:>10}" for c in clases)
        print(f"{'Modelo':<14}│{'bal_acc':>10}{rec_cols}")
        print("─" * 92)
        for key, spec in MODEL_SPECS.items():
            bal, rec = leave_one_collector_out(spec["builder"], Xseg, Xraw, y, recs, collectors, clases)
            print(f"{spec['label']:<14}│{bal:>10.3f}" + "".join(f"{rec[c]:>10.3f}" for c in clases))
        print("─" * 92)

    if args.guardar:
        guardar_artefactos(Xseg, y, files, clases, resultados)


if __name__ == "__main__":
    main()
