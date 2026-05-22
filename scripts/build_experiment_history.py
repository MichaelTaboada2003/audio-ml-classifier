import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


OUTPUT_PATH = os.path.join("outputs", "experiment_history.json")

MODEL_SPECS = {
    "knn_k3": {
        "label": "KNN (k=3)",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=3))]),
    },
    "knn_k5": {
        "label": "KNN (k=5)",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=5))]),
    },
    "svm_lineal": {
        "label": "SVM lineal",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="linear", C=1, probability=True, random_state=42))]),
    },
    "svm_rbf": {
        "label": "SVM RBF",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))]),
    },
    "logreg": {
        "label": "Reg. Logística",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1, random_state=42))]),
    },
    "rf": {
        "label": "Random Forest",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, random_state=42))]),
    },
}

EXPERIMENTS = [
    {
        "id": "2clases_10x10",
        "name": "2 emociones",
        "subtitle": "Enojo vs Tristeza · 10x10",
        "status": "archivado",
        "classes": ["Enojo", "Tristeza"],
        "cache_path": os.path.join("outputs", "embeddings_wav2vec2.npz"),
        "dataset_size": 20,
        "samples_per_class": 10,
        "cards": [
            {
                "key": "comparison",
                "title": "Comparativa de modelos",
                "description": "Comparación entre clasificadores para el experimento binario de 2 emociones.",
                "image": "/outputs/figuras/resultados_2clases.png",
                "observation": "En el escenario binario la separación es más simple y varios modelos se mantienen competitivos.",
            },
            {
                "key": "performance",
                "title": "Desempeño y matriz de confusión",
                "description": "Resumen del rendimiento del mejor modelo sobre Enojo y Tristeza.",
                "image": "/outputs/figuras/resultados_balanceado_10v10.png",
                "observation": "Este experimento sirve como línea base histórica antes de agregar la emoción Feliz.",
            },
            {
                "key": "separability",
                "title": "Separabilidad en 2D",
                "description": "Proyección de embeddings para el escenario binario.",
                "image": "/outputs/figuras/separabilidad_v2.png",
                "observation": "La frontera binaria suele quedar mejor definida que en el caso multiclase.",
            },
            {
                "key": "filtering",
                "title": "Distribución de calidad acústica",
                "description": "Reporte de filtrado usado para escoger las muestras más expresivas.",
                "image": "/outputs/figuras/reporte_filtrado_v2.png",
                "observation": "El filtrado ayuda a aislar audios con una firma emocional más clara.",
            },
        ],
    },
    {
        "id": "3clases_10x10x10",
        "name": "3 emociones",
        "subtitle": "Enojo, Tristeza y Feliz · 10x10x10",
        "status": "actual",
        "classes": ["Enojo", "Tristeza", "Feliz"],
        "cache_path": os.path.join("outputs", "embeddings_v2.npz"),
        "dataset_size": 30,
        "samples_per_class": 10,
        "cards": [
            {
                "key": "comparison",
                "title": "Comparativa de modelos",
                "description": "Comparación de rendimiento entre features manuales y wav2vec2 en el escenario multiclase.",
                "image": "/outputs/figuras/comparativa_features_vs_wav2vec2.png",
                "observation": "Agregar Feliz vuelve más difícil la separación y reduce el margen respecto al caso binario.",
            },
            {
                "key": "performance",
                "title": "Desempeño y matriz de confusión",
                "description": "Matriz de confusión y balanced accuracy del mejor modelo en el experimento de 3 emociones.",
                "image": "/outputs/figuras/resultados_3clases_10x10x10.png",
                "observation": "La mayor confusión aparece entre Enojo y Feliz por su activación acústica similar.",
            },
            {
                "key": "separability",
                "title": "Separabilidad en 2D",
                "description": "Proyección de embeddings wav2vec2 para las 3 emociones activas.",
                "image": "/outputs/figuras/separabilidad_v2.png",
                "observation": "Tristeza tiende a separarse mejor; Feliz y Enojo comparten una región más cercana.",
            },
            {
                "key": "filtering",
                "title": "Distribución de calidad acústica",
                "description": "Distribución de scores que guía el filtrado del dataset completo.",
                "image": "/outputs/figuras/reporte_filtrado_v2.png",
                "observation": "Esta vista ayuda a ver qué clases siguen limitadas por calidad de actuación o solapamiento acústico.",
            },
        ],
    },
]


def evaluar_cache(cache_path):
    c = np.load(cache_path, allow_pickle=True)
    X = c["X"]
    y = c["y"]
    loo = LeaveOneOut()
    metrics = {}
    best_model = None
    best_bal_acc = -1.0

    for key, spec in MODEL_SPECS.items():
        preds = []
        true = []
        for train_idx, test_idx in loo.split(X):
            model = spec["builder"]()
            model.fit(X[train_idx], y[train_idx])
            preds.append(model.predict(X[test_idx])[0])
            true.append(y[test_idx][0])

        preds = np.array(preds)
        true = np.array(true)
        acc = float(np.mean(preds == true))
        bal_acc = float(balanced_accuracy_score(true, preds))
        metrics[key] = {
            "label": spec["label"],
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
        }
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_model = key

    return {
        "metrics": metrics,
        "best_model": best_model,
        "best_balanced_accuracy": round(best_bal_acc, 4),
        "chance_accuracy": round(1.0 / len(np.unique(y)), 4),
        "class_counts": {clase: int(np.sum(y == clase)) for clase in sorted(np.unique(y).tolist())},
    }


def build_history():
    payload = {"default_experiment_id": "3clases_10x10x10", "experiments": []}
    for experiment in EXPERIMENTS:
        if not os.path.exists(experiment["cache_path"]):
            print(f"[WARN] No existe caché para {experiment['id']}: {experiment['cache_path']}")
            continue
        evaluation = evaluar_cache(experiment["cache_path"])
        merged = dict(experiment)
        merged.pop("cache_path", None)
        merged.update(evaluation)
        payload["experiments"].append(merged)
        print(f"{experiment['id']}: mejor={evaluation['best_model']} bal_acc={evaluation['best_balanced_accuracy']:.4f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Historial guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    build_history()
