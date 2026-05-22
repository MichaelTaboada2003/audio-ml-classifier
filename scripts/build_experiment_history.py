import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Importa configuración centralizada desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import N_PER_CLASS, ACTIVE_CLASSES


OUTPUT_PATH = os.path.join("outputs", "experiment_history.json")

# ID y subtítulo derivados de los conteos REALES del cache (Feliz puede tener menos)
def _derive_3cl_id_and_subtitle():
    cache_path = os.path.join("outputs", "embeddings_v2.npz")
    if not os.path.exists(cache_path):
        return f"3clases_{N_PER_CLASS}cap", f"Enojo / Tristeza / Feliz · cap a {N_PER_CLASS} por clase"
    c = np.load(cache_path, allow_pickle=True)
    y = c["y"]
    counts = [int(np.sum(y == clase)) for clase in ACTIVE_CLASSES]
    parts = "x".join(str(c) for c in counts)
    label_parts = ", ".join(f"{clase} {n}" for clase, n in zip(ACTIVE_CLASSES, counts))
    return f"3clases_{parts}", f"Enojo / Tristeza / Feliz · {label_parts}"


CURRENT_3CL_ID, CURRENT_3CL_SUBTITLE = _derive_3cl_id_and_subtitle()

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
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="linear", C=1, probability=True, class_weight="balanced", random_state=42))]),
    },
    "svm_rbf": {
        "label": "SVM RBF",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42))]),
    },
    "logreg": {
        "label": "Reg. Logística",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1, class_weight="balanced", random_state=42))]),
    },
    "rf": {
        "label": "Random Forest",
        "builder": lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))]),
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
        "id": CURRENT_3CL_ID,
        "name": "3 emociones",
        "subtitle": CURRENT_3CL_SUBTITLE,
        "status": "actual",
        "classes": ACTIVE_CLASSES,
        "cache_path": os.path.join("outputs", "embeddings_v2.npz"),
        "samples_per_class": N_PER_CLASS,  # cap deseado; cache real puede tener menos en alguna clase
        "cards": [
            {
                "key": "performance",
                "title": "Desempeño y matriz de confusión (LOOCV)",
                "description": "Matriz de confusión y comparativa LOOCV del mejor modelo. Entrenado con cap N=20 por clase (Feliz limitado a 15 por disponibilidad) y class_weight='balanced'.",
                "image": "/outputs/figuras/resultados_v2.png",
                "observation": "La mayor confusión persiste entre Enojo↔Feliz por su activación acústica similar. Tristeza se separa mejor. Esta cifra LOOCV (~67% LogReg) mide generalización a audios nuevos del mismo recolector — ver tarjeta de holdout para generalización a audios de menor expresividad.",
            },
            {
                "key": "holdout",
                "title": "Generalización a holdout (NUEVA)",
                "description": "Recall por modelo y clase sobre audios que el modelo desplegado NUNCA vio en training (excluidos del cache de embeddings).",
                "image": "/outputs/figuras/holdout_por_modelo.png",
                "observation": "Antes de expandir el training a N=20, SVM lineal acertaba solo 9% de los Enojo holdout. Con N=20 sube a 59%. La mejora confirma que el training previo (N=14, solo audios EXCELENTE) era demasiado homogéneo en expresividad. Feliz holdout=0 porque los 15 audios disponibles ya están todos en training.",
            },
            {
                "key": "separability",
                "title": "Separabilidad en 2D",
                "description": "Proyección PCA y t-SNE de los embeddings wav2vec2 actualmente en cache.",
                "image": "/outputs/figuras/separabilidad_v2.png",
                "observation": "Tristeza tiende a separarse mejor del resto. Feliz y Enojo comparten una región más cercana, consistente con la confusión observada en la matriz de confusión y con la sobreposición de scores acústicos (score_enojo a menudo > score_feliz en los audios Feliz del dataset).",
            },
            {
                "key": "filtering",
                "title": "Distribución de calidad acústica",
                "description": "Distribución de scores por clase calculados sobre los primeros 10s de cada audio. Guía la selección del training set.",
                "image": "/outputs/figuras/reporte_filtrado_v2.png",
                "observation": "Visualiza por qué Feliz es el cuello de botella (solo 15 audios totales) y muestra el rango de scores que el modelo ve en training vs el que aparece en producción.",
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
        "dataset_size": int(len(y)),
    }


def build_history():
    payload = {"default_experiment_id": CURRENT_3CL_ID, "experiments": []}
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
