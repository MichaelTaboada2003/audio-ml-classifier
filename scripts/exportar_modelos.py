import csv
import json
import os
import sys
import time
from collections import Counter

import joblib
import librosa
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import Wav2Vec2Model, Wav2Vec2Processor

# Importa configuración centralizada desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_PER_CLASS, ACTIVE_CLASSES


DATA_DIR = os.path.join("data", "AUDIOS MACHINE LEARNING")
REPORTE = os.path.join("outputs", "reporte_filtrado_v2.csv")
CACHE = os.path.join("outputs", "embeddings_v2.npz")
MODEL_DIR = os.path.join("outputs", "modelos")
METRICS_PATH = os.path.join("outputs", "model_metrics.json")
PREDS_CSV = os.path.join("outputs", "predicciones_loocv.csv")
PREDS_JSON = os.path.join("outputs", "predicciones_loocv.json")

SCORE_COLUMNS = {
    "Enojo": "score_enojo",
    "Tristeza": "score_tristeza",
    "Feliz": "score_feliz",
}
TARGET_SR = 16000
MIN_AUDIO_DURATION = 4.0   # se descartan audios más cortos que esto
MAX_DURATION = 10.0        # ventana de N s a procesar desde el inicio del audio
# Nota: probamos warm-up de 1-2s (offset al inicio) y dilución del audio completo;
# ambos empeoran el bal_acc. 0-10s es el sweet spot para este dataset.
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SPECS = {
    "svm_lineal": {
        "label": "SVM lineal",
        "description": "Excelente separabilidad lineal en el espacio de embeddings.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", SVC(kernel="linear", C=1, probability=True, class_weight="balanced", random_state=42)),
        ]),
    },
    "logreg": {
        "label": "Reg. Logística",
        "description": "Clasificador probabilístico de entropía máxima.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=2000, C=1, class_weight="balanced", random_state=42)),
        ]),
    },
    "svm_rbf": {
        "label": "SVM RBF",
        "description": "Captura fronteras no lineales entre emociones.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        ]),
    },
    "rf": {
        "label": "Random Forest",
        "description": "Ensemble robusto basado en árboles de decisión.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
        ]),
    },
    "knn_k5": {
        "label": "KNN (k=5)",
        "description": "Votación por 5 vecinos más cercanos.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", KNeighborsClassifier(n_neighbors=5)),
        ]),
    },
    "knn_k3": {
        "label": "KNN (k=3)",
        "description": "Votación por 3 vecinos más cercanos.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", KNeighborsClassifier(n_neighbors=3)),
        ]),
    },
}


def seleccionar_audios():
    if not os.path.exists(REPORTE):
        raise FileNotFoundError(f"No se encontró el reporte en {REPORTE}")

    df = np.genfromtxt(REPORTE, delimiter=",", names=True, dtype=None, encoding="utf-8")
    seleccion = {}
    for clase in ACTIVE_CLASSES:
        score_col = SCORE_COLUMNS[clase]
        filas = [row for row in df if row["clase_original"] == clase]
        filas.sort(key=lambda row: row[score_col], reverse=True)
        audios = []
        for row in filas:
            ruta = os.path.join(DATA_DIR, clase, row["archivo"])
            if os.path.exists(ruta):
                audios.append(row["archivo"])
            if len(audios) >= N_PER_CLASS:
                break
        seleccion[clase] = audios
        print(f"{clase:<9} ({len(audios)}): {audios}")
    return seleccion


def cargar_wav2vec2():
    print("Cargando facebook/wav2vec2-base...")
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(DEVICE)
    model.eval()
    print(f"  Modelo listo en {time.time() - t0:.1f}s sobre {DEVICE}")
    return processor, model


def extraer_embedding(ruta, processor, model):
    """Carga los primeros MAX_DURATION s del audio y devuelve un embedding de 768d.

    Sweet spot empírico para este dataset: 10s desde el inicio. Probamos warm-up
    de 1-2s y dilución del audio completo; ambos empeoran el bal_acc.
    """
    y, _ = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=MAX_DURATION)
    if len(y) < TARGET_SR * MIN_AUDIO_DURATION:
        return None

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(inputs.input_values.to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)


def construir_cache(force_rebuild=False):
    """Construye/lee el cache: un embedding por audio (audio completo, sin truncar).

    Tolera que alguna clase tenga MENOS de N_PER_CLASS audios disponibles
    (típicamente Feliz con 15). Se usan TODOS los disponibles hasta N_PER_CLASS.
    """
    if os.path.exists(CACHE) and not force_rebuild:
        c = np.load(CACHE, allow_pickle=True)
        needed = {"X", "y", "rec", "files"}
        if needed.issubset(set(c.files)):
            y = c["y"]
            counts = {clase: int(np.sum(y == clase)) for clase in ACTIVE_CLASSES}
            # Recomputamos selección actual para saber qué deberíamos tener
            seleccion_actual = seleccionar_audios()
            expected_counts = {clase: len(seleccion_actual[clase]) for clase in ACTIVE_CLASSES}
            if "audio_id" in c.files or "rms" in c.files:
                print("Caché en formato chunk antiguo detectado. Recalculando con audio completo...")
            elif counts == expected_counts:
                print(f"Usando caché existente: {CACHE}")
                print(f"  Audios por clase: {counts}")
                return c["X"], y, c["rec"], c["files"]
            else:
                print(f"Caché no coincide con selección actual ({counts} vs {expected_counts}). Recalculando...")
        else:
            print("Caché legacy detectado. Recalculando...")

    seleccion = seleccionar_audios()
    processor, model = cargar_wav2vec2()
    plan = [(archivo, clase) for clase in ACTIVE_CLASSES for archivo in seleccion[clase]]

    X_list, y_list, rec_list, files_list = [], [], [], []
    for idx, (archivo, clase) in enumerate(plan, 1):
        ruta = os.path.join(DATA_DIR, clase, archivo)
        emb = extraer_embedding(ruta, processor, model)
        if emb is None:
            print(f"  [{idx:02d}/{len(plan)}] {clase}/{archivo} → SKIP (audio demasiado corto)")
            continue
        print(f"  [{idx:02d}/{len(plan)}] {clase}/{archivo}")
        X_list.append(emb)
        y_list.append(clase)
        rec_list.append(archivo[:2])
        files_list.append(archivo)

    X = np.vstack(X_list)
    y = np.array(y_list)
    rec = np.array(rec_list)
    files = np.array(files_list)

    np.savez(CACHE, X=X, y=y, rec=rec, files=files)
    final_counts = {clase: int(np.sum(y == clase)) for clase in ACTIVE_CLASSES}
    print(f"Caché guardada en {CACHE}  |  {len(y)} embeddings totales")
    print(f"  Por clase: {final_counts}")
    return X, y, rec, files


def evaluar_modelo(builder, X, y):
    """LOOCV estándar: deja un audio fuera, entrena con los demás, predice el faltante."""
    loo = LeaveOneOut()
    n = len(y)
    classes_order = sorted(set(y.tolist()))
    y_pred = np.empty(n, dtype=object)
    proba_matrix = np.zeros((n, len(classes_order)), dtype=float)

    for train_idx, test_idx in loo.split(X):
        i = test_idx[0]
        pipe = builder()
        pipe.fit(X[train_idx], y[train_idx])
        y_pred[i] = pipe.predict(X[test_idx])[0]
        proba = pipe.predict_proba(X[test_idx])[0]
        model_classes = list(pipe.named_steps["c"].classes_)
        for c_idx, c in enumerate(classes_order):
            if c in model_classes:
                proba_matrix[i, c_idx] = proba[model_classes.index(c)]

    acc = float(np.mean(y == y_pred))
    bal_acc = float(balanced_accuracy_score(y, y_pred))
    return acc, bal_acc, y_pred, proba_matrix, classes_order


def exportar_todos(force_rebuild=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, rec, files = construir_cache(force_rebuild=force_rebuild)
    class_counts = {clase: int(np.sum(y == clase)) for clase in sorted(set(y.tolist()))}
    print(f"Entrenando con {len(y)} audios (1 embedding por audio, audio completo).")
    print(f"Distribución de clases: {class_counts}")

    metrics = {
        "active_classes": ACTIVE_CLASSES,
        "samples_per_class": N_PER_CLASS,
        "dataset_size": int(len(y)),
        "chance_accuracy": round(1.0 / len(ACTIVE_CLASSES), 4),
        "models": {},
    }

    best_key = None
    best_bal_acc = -1.0
    preds_by_model = {}

    for key, spec in MODEL_SPECS.items():
        builder = spec["builder"]
        acc, bal_acc, y_pred, proba, classes_order = evaluar_modelo(builder, X, y)

        pipe = builder()
        pipe.fit(X, y)
        path = os.path.join(MODEL_DIR, f"{key}.joblib")
        joblib.dump(pipe, path)

        metrics["models"][key] = {
            "label": spec["label"],
            "description": spec["description"],
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }
        preds_by_model[key] = {
            "label": spec["label"],
            "y_pred": y_pred,
            "proba": proba,
            "classes_order": classes_order,
        }
        print(f"  {spec['label']:<18} acc={acc:.4f} bal_acc={bal_acc:.4f} -> {path}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_key = key

    metrics["best_model"] = best_key
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Métricas guardadas en {METRICS_PATH}")
    print(f"Mejor modelo: {MODEL_SPECS[best_key]['label']} ({best_bal_acc:.4f} bal_acc)")

    exportar_predicciones(y, rec, files, preds_by_model)


def exportar_predicciones(y, rec, files, preds_by_model):
    """Exporta predicciones LOOCV por audio (CSV + JSON) e imprime fallos."""
    n = len(y)
    model_keys = list(preds_by_model.keys())

    # --- CSV: una fila por audio ---
    with open(PREDS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["archivo", "recolector", "clase_real"]
        for k in model_keys:
            header.append(f"pred_{k}")
        header += ["n_modelos_correctos", "n_modelos_totales", "modelos_fallidos"]
        writer.writerow(header)

        for i in range(n):
            preds = [preds_by_model[k]["y_pred"][i] for k in model_keys]
            n_correct = sum(1 for p in preds if p == y[i])
            failed = [k for k, p in zip(model_keys, preds) if p != y[i]]
            row = [str(files[i]), str(rec[i]), str(y[i]), *preds,
                   n_correct, len(model_keys), ";".join(failed)]
            writer.writerow(row)
    print(f"\nCSV de predicciones por audio: {PREDS_CSV}")

    # --- JSON detallado ---
    payload = {
        "classes_active": ACTIVE_CLASSES,
        "n_audios": n,
        "models": {},
        "audio_summary": [],
    }
    for k in model_keys:
        info = preds_by_model[k]
        classes_order = info["classes_order"]
        predictions = []
        for i in range(n):
            predictions.append({
                "archivo": str(files[i]),
                "recolector": str(rec[i]),
                "clase_real": str(y[i]),
                "prediccion": str(info["y_pred"][i]),
                "correcto": bool(info["y_pred"][i] == y[i]),
                "probabilidades": {c: round(float(info["proba"][i, idx]), 4)
                                   for idx, c in enumerate(classes_order)},
            })
        payload["models"][k] = {
            "label": info["label"],
            "classes_order": classes_order,
            "predictions": predictions,
        }

    for i in range(n):
        preds = [preds_by_model[k]["y_pred"][i] for k in model_keys]
        n_correct = sum(1 for p in preds if p == y[i])
        failed = [k for k, p in zip(model_keys, preds) if p != y[i]]
        wrong_preds = [p for p in preds if p != y[i]]
        confundido_con = Counter(wrong_preds).most_common(1)[0][0] if wrong_preds else None
        payload["audio_summary"].append({
            "archivo": str(files[i]),
            "recolector": str(rec[i]),
            "clase_real": str(y[i]),
            "n_modelos_correctos": n_correct,
            "n_modelos_totales": len(model_keys),
            "modelos_fallidos": failed,
            "confundido_con": str(confundido_con) if confundido_con else None,
        })

    with open(PREDS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"JSON detallado de predicciones: {PREDS_JSON}")

    # --- Consola: ranking de audios problemáticos ---
    print("\n" + "=" * 78)
    print("AUDIOS PROBLEMÁTICOS (ranking por # de modelos que fallan)")
    print("=" * 78)
    ranking = sorted(payload["audio_summary"],
                     key=lambda s: (s["n_modelos_totales"] - s["n_modelos_correctos"]),
                     reverse=True)
    print(f"{'Archivo':<28} {'Clase':<10} {'Fallos':<8} {'Confundido con':<14} Modelos")
    print("-" * 100)
    for s in ranking:
        n_fail = s["n_modelos_totales"] - s["n_modelos_correctos"]
        if n_fail == 0:
            continue
        print(f"{s['archivo']:<28} {s['clase_real']:<10} {n_fail}/{s['n_modelos_totales']:<6} "
              f"{(s['confundido_con'] or '-'):<14} {', '.join(s['modelos_fallidos'])}")

    print("\nRESUMEN POR CLASE (audios que fallaron en al menos 1 modelo):")
    by_class = Counter()
    for s in payload["audio_summary"]:
        if s["n_modelos_totales"] - s["n_modelos_correctos"] > 0:
            by_class[s["clase_real"]] += 1
    for clase in ACTIVE_CLASSES:
        total = sum(1 for s in payload["audio_summary"] if s["clase_real"] == clase)
        print(f"  {clase:<10} {by_class[clase]}/{total} audios con al menos 1 fallo")


if __name__ == "__main__":
    exportar_todos(force_rebuild=True)
