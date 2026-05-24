"""Construye outputs/experiment_history.json para el switcher de la pestaña Analisis.

Reescrito para el paradigma de segmentos localizados (data/procesado): cada escenario
(2 clases archivado, 3 clases actual) se evalua con LOO HONESTO — entrenar con los
segmentos de los otros audios y testear sobre el AUDIO CRUDO del que se deja fuera.
Las cifras reportadas son de generalizacion real, no la prima de cherry-picking de
medir sobre los mismos segmentos curados. Ver [[project_localizacion_segmentos]].

Lee outputs/proc_embeddings.npz (Xseg + Xraw + y) generado por entrenar_procesado.py.
Las figuras referidas en las tarjetas las produce scripts/regenerar_figuras.py.
"""

import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_PATH = os.path.join("outputs", "experiment_history.json")
PROC_CACHE = os.path.join("outputs", "proc_embeddings.npz")

MODEL_SPECS = {
    "knn_k3": ("KNN (k=3)", lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=3))])),
    "knn_k5": ("KNN (k=5)", lambda: Pipeline([("s", StandardScaler()), ("c", KNeighborsClassifier(n_neighbors=5))])),
    "svm_lineal": ("SVM lineal", lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="linear", C=1, probability=True, class_weight="balanced", random_state=42))])),
    "svm_rbf": ("SVM RBF", lambda: Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42))])),
    "logreg": ("Reg. Logística", lambda: Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, C=1, class_weight="balanced", random_state=42))])),
    "rf": ("Random Forest", lambda: Pipeline([("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))])),
}

# Definicion de escenarios. cards[].image apunta a figuras que produce regenerar_figuras.py.
SCENARIOS = [
    {
        "id": "2clases",
        "name": "2 emociones",
        "subtitle": "Enojo / Tranquilidad · baseline binario",
        "status": "archivado",
        "classes": ["Enojo", "Tranquilidad"],
        "cards": [
            {
                "key": "performance",
                "title": "Desempeño honesto (test = audio crudo)",
                "image": "/outputs/figuras/resultados_2clases.png",
                "description": "Matriz de confusión y comparativa de modelos bajo LOO honesto: se entrena con los segmentos localizados y se predice sobre el audio crudo del audio dejado fuera.",
                "observation": "Baseline binario Enojo vs Tranquilidad: SVM lineal alcanza ~92% balanced accuracy sobre audio crudo no visto. Es la línea base antes de subir a 3 emociones.",
            },
            {
                "key": "separability",
                "title": "Separabilidad en 2D",
                "image": "/outputs/figuras/separabilidad_2clases.png",
                "description": "Proyección PCA y t-SNE de los embeddings emocionales (audeering, 1024d) de los segmentos.",
                "observation": "Enojo y Tranquilidad ocupan regiones bien separadas en el espacio emocional: la frontera binaria es la más limpia.",
            },
        ],
    },
    {
        "id": "3clases",
        "name": "3 emociones",
        "subtitle": "Enojo / Feliz / Tristeza · actual",
        "status": "actual",
        "classes": ["Enojo", "Feliz", "Tristeza"],
        "cards": [
            {
                "key": "performance",
                "title": "Desempeño honesto (test = audio crudo)",
                "image": "/outputs/figuras/resultados_3clases.png",
                "description": "Matriz de confusión y comparativa de modelos bajo LOO honesto (entrena=segmento localizado, test=audio crudo). Modelos con class_weight='balanced'.",
                "observation": "SVM lineal/RBF llegan a ~97% balanced accuracy sobre audio crudo no visto. Feliz y Tristeza se clasifican casi perfecto; Enojo es la clase más difícil por tener solo 13 audios y solaparse acústicamente con Feliz.",
            },
            {
                "key": "separability",
                "title": "Separabilidad en 2D",
                "image": "/outputs/figuras/separabilidad_3clases.png",
                "description": "Proyección PCA y t-SNE de los embeddings emocionales (audeering, 1024d) de los segmentos localizados de las 3 clases.",
                "observation": "Tristeza forma un cúmulo nítido; Enojo y Feliz comparten una región de alta activación, consistente con la confusión residual en esa frontera.",
            },
        ],
    },
]


def evaluar_honesto(Xseg, Xraw, y, clases):
    """LOO honesto por modelo: train=segmentos de otros, test=audio crudo del que falta."""
    metrics = {}
    best_model, best_bal = None, -1.0
    for key, (label, builder) in MODEL_SPECS.items():
        loo = LeaveOneOut()
        y_pred = np.empty(len(y), dtype=object)
        for train_idx, test_idx in loo.split(Xseg):
            pipe = builder()
            pipe.fit(Xseg[train_idx], y[train_idx])
            y_pred[test_idx[0]] = pipe.predict(Xraw[test_idx])[0]
        acc = float(np.mean(y_pred == y))
        bal = float(balanced_accuracy_score(y, y_pred))
        rec = recall_score(y, y_pred, labels=clases, average=None, zero_division=0)
        metrics[key] = {
            "label": label,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal, 4),
            "recall_por_clase": {c: round(float(r), 4) for c, r in zip(clases, rec)},
        }
        if bal > best_bal:
            best_bal, best_model = bal, key
    return metrics, best_model, round(best_bal, 4)


def build_history():
    if not os.path.exists(PROC_CACHE):
        raise FileNotFoundError(f"No existe {PROC_CACHE}. Corre scripts/entrenar_procesado.py primero.")
    cache = np.load(PROC_CACHE, allow_pickle=True)
    Xseg, Xraw, y_all = cache["Xseg"], cache["Xraw"], cache["y"]
    disponibles = set(y_all.tolist())

    payload = {"default_experiment_id": "3clases", "experiments": []}
    for scn in SCENARIOS:
        clases = sorted(scn["classes"])
        if not set(clases).issubset(disponibles):
            print(f"[{scn['id']}] OMITIDO: faltan clases {set(clases) - disponibles}")
            continue
        mask = np.isin(y_all, clases)
        Xs, Xr, y = Xseg[mask], Xraw[mask], y_all[mask]
        metrics, best_model, best_bal = evaluar_honesto(Xs, Xr, y, clases)
        entry = {
            "id": scn["id"],
            "name": scn["name"],
            "subtitle": scn["subtitle"],
            "status": scn["status"],
            "classes": clases,
            "class_counts": {c: int(np.sum(y == c)) for c in clases},
            "dataset_size": int(len(y)),
            "chance_accuracy": round(1.0 / len(clases), 4),
            "metric": "balanced_accuracy = LOO honesto (train=segmento, test=audio crudo 10 s)",
            "metrics": metrics,
            "best_model": best_model,
            "best_balanced_accuracy": best_bal,
            "cards": scn["cards"],
        }
        payload["experiments"].append(entry)
        print(f"{scn['id']}: mejor={best_model} bal_acc_honesto={best_bal:.4f} | {entry['class_counts']}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Historial guardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    build_history()
