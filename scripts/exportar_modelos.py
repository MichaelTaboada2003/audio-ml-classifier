import json
import os
import sys
import time

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

SCORE_COLUMNS = {
    "Enojo": "score_enojo",
    "Tristeza": "score_tristeza",
    "Feliz": "score_feliz",
}
TARGET_SR = 16000
DURATION = 10.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

MODEL_SPECS = {
    "svm_lineal": {
        "label": "SVM lineal",
        "description": "Excelente separabilidad lineal en el espacio de embeddings.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", SVC(kernel="linear", C=1, probability=True, random_state=42)),
        ]),
    },
    "logreg": {
        "label": "Reg. Logística",
        "description": "Clasificador probabilístico de entropía máxima.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", LogisticRegression(max_iter=2000, C=1, random_state=42)),
        ]),
    },
    "svm_rbf": {
        "label": "SVM RBF",
        "description": "Captura fronteras no lineales entre emociones.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ]),
    },
    "rf": {
        "label": "Random Forest",
        "description": "Ensemble robusto basado en árboles de decisión.",
        "builder": lambda: Pipeline([
            ("s", StandardScaler()),
            ("c", RandomForestClassifier(n_estimators=200, random_state=42)),
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
    y, _ = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=DURATION)
    if len(y) < TARGET_SR * 0.3:
        return None

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(inputs.input_values.to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)


def construir_cache(force_rebuild=False):
    expected = N_PER_CLASS * len(ACTIVE_CLASSES)

    if os.path.exists(CACHE) and not force_rebuild:
        c = np.load(CACHE, allow_pickle=True)
        y = c["y"]
        counts = {clase: int(np.sum(y == clase)) for clase in ACTIVE_CLASSES}
        if len(y) == expected and all(counts.get(clase, 0) == N_PER_CLASS for clase in ACTIVE_CLASSES):
            print(f"Usando caché existente: {CACHE}")
            print(f"Clases en caché: {counts}")
            return c["X"], y, c["rec"]
        print("La caché no coincide con el experimento multiclase. Recalculando...")

    seleccion = seleccionar_audios()
    processor, model = cargar_wav2vec2()
    registros = []
    plan = [(archivo, clase) for clase in ACTIVE_CLASSES for archivo in seleccion[clase]]

    for idx, (archivo, clase) in enumerate(plan, 1):
        ruta = os.path.join(DATA_DIR, clase, archivo)
        print(f"  [{idx:02d}/{len(plan)}] {clase}/{archivo}")
        emb = extraer_embedding(ruta, processor, model)
        if emb is not None:
            registros.append({"emb": emb, "etiqueta": clase, "rec": archivo[:2]})

    if len(registros) != expected:
        raise RuntimeError(f"Se esperaban {expected} embeddings y se obtuvieron {len(registros)}")

    X = np.vstack([r["emb"] for r in registros])
    y = np.array([r["etiqueta"] for r in registros])
    rec = np.array([r["rec"] for r in registros])
    np.savez(CACHE, X=X, y=y, rec=rec)
    print(f"Caché multiclase guardada en {CACHE}")
    return X, y, rec


def evaluar_modelo(builder, X, y):
    loo = LeaveOneOut()
    y_pred = []
    y_true = []
    for train_idx, test_idx in loo.split(X):
        pipe = builder()
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])[0]
        y_pred.append(pred)
        y_true.append(y[test_idx][0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float(np.mean(y_true == y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    return acc, bal_acc


def exportar_todos(force_rebuild=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, _ = construir_cache(force_rebuild=force_rebuild)
    class_counts = {clase: int(np.sum(y == clase)) for clase in sorted(set(y.tolist()))}
    print(f"Entrenando y exportando modelos usando {len(y)} muestras...")
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

    for key, spec in MODEL_SPECS.items():
        builder = spec["builder"]
        acc, bal_acc = evaluar_modelo(builder, X, y)
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
        print(
            f"  {spec['label']:<18} acc={acc:.4f} bal_acc={bal_acc:.4f} -> {path}"
        )

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_key = key

    metrics["best_model"] = best_key
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Métricas guardadas en {METRICS_PATH}")
    print(f"Mejor modelo: {MODEL_SPECS[best_key]['label']} ({best_bal_acc:.4f} bal_acc)")


if __name__ == "__main__":
    exportar_todos(force_rebuild=True)
