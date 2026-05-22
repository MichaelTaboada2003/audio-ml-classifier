"""Genera datos cuantitativos para el modulo de Toma de Decisiones.

NO MODIFICA el proyecto de ML. Solo CONSUME los embeddings cacheados y
exporta un archivo JSON con matrices de confusion, probabilidades LOO
y curvas ROC para que el dashboard pueda hacer simulacion y sensibilidad
con datos reales (no inventados).
"""

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CACHE_3CL = os.path.join("outputs", "embeddings_v2.npz")
CACHE_2CL = os.path.join("outputs", "embeddings_wav2vec2.npz")
OUT = os.path.join("outputs", "decisions_data.json")

TARGET_CLASS = "Enojo"  # clase positiva para la decision de "flag/escalar"

MODEL_SPECS = {
    "svm_lineal": ("SVM lineal", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", SVC(kernel="linear", C=1, probability=True, random_state=42)),
    ])),
    "logreg": ("Reg. Logistica", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=2000, C=1, random_state=42)),
    ])),
    "svm_rbf": ("SVM RBF", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])),
    "rf": ("Random Forest", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])),
    "knn_k5": ("KNN (k=5)", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", KNeighborsClassifier(n_neighbors=5)),
    ])),
    "knn_k3": ("KNN (k=3)", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", KNeighborsClassifier(n_neighbors=3)),
    ])),
}


def loocv_scenario(X, y, classes_order):
    """Ejecuta LOOCV para cada modelo y devuelve probabilidades + metricas."""
    n = len(y)
    results = {}
    for key, (label, builder) in MODEL_SPECS.items():
        loo = LeaveOneOut()
        prob_matrix = np.zeros((n, len(classes_order)), dtype=float)
        preds = np.empty(n, dtype=object)

        for train_idx, test_idx in loo.split(X):
            pipe = builder()
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[test_idx])[0]
            model_classes = pipe.named_steps["c"].classes_
            # alinea columnas al orden global
            for c_idx, c in enumerate(classes_order):
                if c in model_classes:
                    prob_matrix[test_idx[0], c_idx] = proba[list(model_classes).index(c)]
            preds[test_idx[0]] = pipe.predict(X[test_idx])[0]

        y_pred = np.array(preds.tolist())
        acc = float(np.mean(y_pred == y))
        bal_acc = float(balanced_accuracy_score(y, y_pred))
        cm = confusion_matrix(y, y_pred, labels=classes_order).tolist()
        prec, rec, f1, support = precision_recall_fscore_support(
            y, y_pred, labels=classes_order, zero_division=0
        )

        per_class = {}
        for i, c in enumerate(classes_order):
            per_class[c] = {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }

        # Curva ROC binaria contra TARGET_CLASS
        roc_data = None
        if TARGET_CLASS in classes_order:
            target_col = classes_order.index(TARGET_CLASS)
            y_bin = (y == TARGET_CLASS).astype(int)
            scores = prob_matrix[:, target_col]
            fpr, tpr, thr = roc_curve(y_bin, scores)
            # Sub-muestreo a maximo 60 puntos para JSON ligero
            if len(fpr) > 60:
                idx = np.linspace(0, len(fpr) - 1, 60).astype(int)
                fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
            # Sanitiza inf
            thr = np.where(np.isinf(thr), 1.0, thr)
            roc_data = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thr.tolist(),
            }

        results[key] = {
            "label": label,
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "confusion_matrix": cm,
            "per_class": per_class,
            "roc_target_class": TARGET_CLASS if roc_data else None,
            "roc": roc_data,
            # Probabilidades LOO en orden de classes_order (para recomputo de umbral en cliente)
            "loo_probs": prob_matrix.round(4).tolist(),
            "loo_true": y.tolist(),
        }
    return results


def main():
    if not os.path.exists(CACHE_3CL):
        raise FileNotFoundError(f"No existe {CACHE_3CL}. Corre exportar_modelos.py primero.")

    scenarios = {}

    # Escenario 3 clases (multiclase con clase positiva = Enojo)
    print(f"[3 clases] Cargando {CACHE_3CL}...")
    c = np.load(CACHE_3CL, allow_pickle=True)
    X3, y3 = c["X"], c["y"]
    classes_3 = sorted(set(y3.tolist()))
    print(f"  Clases: {classes_3} | n={len(y3)}")
    scenarios["3clases"] = {
        "id": "3clases",
        "name": "3 emociones",
        "subtitle": "Enojo / Tristeza / Feliz",
        "classes": classes_3,
        "n_total": int(len(y3)),
        "class_counts": {c_: int(np.sum(y3 == c_)) for c_ in classes_3},
        "target_class": TARGET_CLASS,
        "models": loocv_scenario(X3, y3, classes_3),
    }

    # Escenario 2 clases
    if os.path.exists(CACHE_2CL):
        print(f"[2 clases] Cargando {CACHE_2CL}...")
        c2 = np.load(CACHE_2CL, allow_pickle=True)
        X2, y2 = c2["X"], c2["y"]
        classes_2 = sorted(set(y2.tolist()))
        print(f"  Clases: {classes_2} | n={len(y2)}")
        scenarios["2clases"] = {
            "id": "2clases",
            "name": "2 emociones",
            "subtitle": "Enojo / Tristeza",
            "classes": classes_2,
            "n_total": int(len(y2)),
            "class_counts": {c_: int(np.sum(y2 == c_)) for c_ in classes_2},
            "target_class": TARGET_CLASS,
            "models": loocv_scenario(X2, y2, classes_2),
        }
    else:
        print(f"[2 clases] Cache binaria no encontrada ({CACHE_2CL}), se omite.")

    # Defaults de negocio (call center: detectar Enojo para escalar a supervisor)
    business = {
        "context": "Call center que evalua detector de Enojo para escalar llamadas a supervisor senior.",
        "stakeholder": "Coordinador de Operaciones",
        "decision": "Desplegar el clasificador en produccion: si, no, y con que configuracion (escenario + modelo + umbral).",
        "actions": {
            "TP": "Escalar correctamente la llamada al supervisor (salva la relacion con el cliente).",
            "FP": "Escalar una llamada que no era enojo (supervisor distraido innecesariamente).",
            "FN": "No escalar una llamada enojada (riesgo de churn / queja).",
            "TN": "No escalar correctamente (operacion normal, sin costo).",
        },
        "defaults": {
            "volume_per_month": 10000,
            "prevalence_enojo": 0.18,
            "cost_FN": 80.0,
            "cost_FP": 4.0,
            "value_TP": 25.0,
            "cost_per_inference": 0.02,
            "fixed_monthly_cost": 1200.0,
            "threshold": 0.5,
        },
        "ranges": {
            "volume_per_month": [1000, 50000],
            "prevalence_enojo": [0.05, 0.40],
            "cost_FN": [20, 250],
            "cost_FP": [1, 20],
            "value_TP": [5, 80],
            "cost_per_inference": [0.0, 0.10],
            "fixed_monthly_cost": [0, 5000],
            "threshold": [0.05, 0.95],
        },
    }

    payload = {
        "version": 1,
        "target_class": TARGET_CLASS,
        "scenarios": scenarios,
        "business": business,
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nDatos de decision guardados en {OUT}")
    print(f"Tamano: {os.path.getsize(OUT) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
