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


PROC_CACHE = os.path.join("outputs", "proc_embeddings.npz")  # Xseg + Xraw + y + files
OUT = os.path.join("outputs", "decisions_data.json")

TARGET_CLASS = "Enojo"  # clase positiva para la decision de "flag/escalar"

# Escenarios a publicar en el simulador. El front (toggle de Decisiones) usa estas keys.
# "actual" = escenario por defecto; "archivado" = se conserva con sus metricas.
SCENARIOS_DEF = {
    "3clases": {"name": "3 emociones", "subtitle": "Enojo / Feliz / Tristeza",
                "classes": ["Enojo", "Feliz", "Tristeza"], "status": "actual"},
    "2clases": {"name": "2 emociones", "subtitle": "Enojo / Tranquilidad",
                "classes": ["Enojo", "Tranquilidad"], "status": "archivado"},
}

MODEL_SPECS = {
    "svm_lineal": ("SVM lineal", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", SVC(kernel="linear", C=1, probability=True, class_weight="balanced", random_state=42)),
    ])),
    "logreg": ("Reg. Logistica", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=2000, C=1, class_weight="balanced", random_state=42)),
    ])),
    "svm_rbf": ("SVM RBF", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced", random_state=42)),
    ])),
    "rf": ("Random Forest", lambda: Pipeline([
        ("s", StandardScaler()),
        ("c", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),
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


def loocv_scenario(Xseg, Xraw, y, classes_order):
    """LOO HONESTO: en cada fold entrena con los SEGMENTOS de los otros audios y
    predice sobre el AUDIO CRUDO (primeros 10 s) del que se deja fuera. Las probs y
    metricas reflejan generalizacion real (no la prima de cherry-picking de medir
    sobre los mismos segmentos curados). Ver [[project_localizacion_segmentos]]."""
    n = len(y)
    results = {}
    for key, (label, builder) in MODEL_SPECS.items():
        loo = LeaveOneOut()
        prob_matrix = np.zeros((n, len(classes_order)), dtype=float)
        preds = np.empty(n, dtype=object)

        for train_idx, test_idx in loo.split(Xseg):
            pipe = builder()
            pipe.fit(Xseg[train_idx], y[train_idx])
            Xt = Xraw[test_idx]                      # test = audio crudo
            proba = pipe.predict_proba(Xt)[0]
            model_classes = pipe.named_steps["c"].classes_
            # alinea columnas al orden global
            for c_idx, c in enumerate(classes_order):
                if c in model_classes:
                    prob_matrix[test_idx[0], c_idx] = proba[list(model_classes).index(c)]
            preds[test_idx[0]] = pipe.predict(Xt)[0]

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
    if not os.path.exists(PROC_CACHE):
        raise FileNotFoundError(
            f"No existe {PROC_CACHE}. Corre scripts/entrenar_procesado.py primero "
            f"(construye el cache de embeddings segmento+crudo)."
        )

    cache = np.load(PROC_CACHE, allow_pickle=True)
    Xseg, Xraw, y_all = cache["Xseg"], cache["Xraw"], cache["y"]
    disponibles = set(y_all.tolist())

    scenarios = {}
    for sid, sdef in SCENARIOS_DEF.items():
        clases = sdef["classes"]
        if not set(clases).issubset(disponibles):
            print(f"[{sid}] OMITIDO: faltan clases {set(clases) - disponibles} en {PROC_CACHE}")
            continue
        mask = np.isin(y_all, clases)
        Xs, Xr, y = Xseg[mask], Xraw[mask], y_all[mask]
        classes_order = sorted(clases)
        print(f"[{sid}] {sdef['subtitle']} | n={len(y)} | clases={classes_order}")
        scenarios[sid] = {
            "id": sid,
            "name": sdef["name"],
            "subtitle": sdef["subtitle"],
            "status": sdef["status"],
            "classes": classes_order,
            "n_total": int(len(y)),
            "class_counts": {c_: int(np.sum(y == c_)) for c_ in classes_order},
            "target_class": TARGET_CLASS,
            "metric": "LOO honesto (train=segmento, test=audio crudo 10 s)",
            "models": loocv_scenario(Xs, Xr, y, classes_order),
        }

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
