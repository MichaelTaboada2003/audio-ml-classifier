"""Regenera las figuras del dashboard (pestaña Análisis) para el paradigma de
segmentos localizados, con métricas HONESTAS.

Para cada escenario (2 clases archivado, 3 clases actual) produce:
  - outputs/figuras/resultados_<sid>.png    (CM del mejor modelo + barchart, LOO honesto)
  - outputs/figuras/separabilidad_<sid>.png  (PCA + t-SNE de los embeddings de segmento)

LOO honesto = entrenar con los segmentos de los otros audios y predecir sobre el
AUDIO CRUDO del audio dejado fuera. Lee outputs/proc_embeddings.npz (Xseg+Xraw+y).
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_experiment_history import MODEL_SPECS  # fuente unica de los 6 modelos

PROC_CACHE = os.path.join("outputs", "proc_embeddings.npz")
FIG_DIR = os.path.join("outputs", "figuras")
CLASS_COLOR = {"Enojo": "#DD8452", "Tristeza": "#C44E52", "Feliz": "#E377C2", "Tranquilidad": "#55A868"}
MODEL_ORDER = ["svm_lineal", "logreg", "svm_rbf", "rf", "knn_k5", "knn_k3"]

SCENARIOS = {
    "2clases": ["Enojo", "Tranquilidad"],
    "3clases": ["Enojo", "Feliz", "Tristeza"],
}


def honest_loo(builder, Xseg, Xraw, y):
    """y_pred bajo LOO honesto (train=segmentos de otros, test=audio crudo del que falta)."""
    y_pred = np.empty(len(y), dtype=object)
    for tr, te in LeaveOneOut().split(Xseg):
        pipe = builder()
        pipe.fit(Xseg[tr], y[tr])
        y_pred[te[0]] = pipe.predict(Xraw[te])[0]
    return y_pred


def figura_resultados(sid, Xseg, Xraw, y, clases):
    classes_order = sorted(clases)
    results = {}
    for key, (label, builder) in MODEL_SPECS.items():
        yp = honest_loo(builder, Xseg, Xraw, y)
        results[key] = (label, float(balanced_accuracy_score(y, yp)), yp)
    best = max(results, key=lambda k: results[k][1])
    chance = 1.0 / len(classes_order)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Matriz de confusión del mejor modelo ---
    ax = axes[0]
    cm = confusion_matrix(y, results[best][2], labels=classes_order)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes_order)))
    ax.set_yticks(range(len(classes_order)))
    ax.set_xticklabels(classes_order)
    ax.set_yticklabels(classes_order)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"{results[best][0]} — CM (LOO honesto, test = audio crudo)")
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=13)

    # --- Barchart comparativa ---
    ax = axes[1]
    keys = [k for k in MODEL_ORDER if k in results]
    labels = [results[k][0] for k in keys]
    bals = [results[k][1] for k in keys]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, bals, color=["#1f77b4" if k == best else "#9ec5dd" for k in keys])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(chance, color="red", linestyle="--", label=f"chance ({chance:.3f})")
    ax.set_xlabel("Balanced Accuracy (LOO honesto)")
    ax.set_xlim(0, 1.0)
    for yp_, v in zip(y_pos, bals):
        ax.text(v + 0.01, yp_, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"Comparativa honesta — {len(classes_order)} clases")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"resultados_{sid}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}  (best={best} bal_acc={results[best][1]:.3f})")


def figura_separabilidad(sid, Xseg, y, clases):
    X_sc = StandardScaler().fit_transform(Xseg)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_sc)
    perp = max(2, min(30, (len(y) - 1) // 3))
    X_tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca").fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proj, titulo in [(axes[0], X_pca, "PCA 2D"), (axes[1], X_tsne, f"t-SNE 2D (perplexity={perp})")]:
        for clase in sorted(clases):
            mask = y == clase
            ax.scatter(proj[mask, 0], proj[mask, 1], c=CLASS_COLOR.get(clase, "#777"),
                       label=f"{clase} (n={int(mask.sum())})", s=60, alpha=0.8, edgecolor="white")
        ax.set_title(titulo)
        ax.legend(title="Clase", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    n_per = ", ".join(f"{c}={int(np.sum(y == c))}" for c in sorted(clases))
    fig.suptitle(f"Separabilidad — embeddings emocionales audeering 1024d ({n_per})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"separabilidad_{sid}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    if not os.path.exists(PROC_CACHE):
        raise FileNotFoundError(f"No existe {PROC_CACHE}. Corre scripts/entrenar_procesado.py primero.")
    os.makedirs(FIG_DIR, exist_ok=True)
    cache = np.load(PROC_CACHE, allow_pickle=True)
    Xseg, Xraw, y_all = cache["Xseg"], cache["Xraw"], cache["y"]
    disponibles = set(y_all.tolist())

    print("Regenerando figuras (LOO honesto)…")
    for sid, clases in SCENARIOS.items():
        if not set(clases).issubset(disponibles):
            print(f"[{sid}] OMITIDO: faltan clases {set(clases) - disponibles}")
            continue
        mask = np.isin(y_all, clases)
        Xs, Xr, y = Xseg[mask], Xraw[mask], y_all[mask]
        print(f"[{sid}] {clases}  n={len(y)}")
        figura_resultados(sid, Xs, Xr, y, clases)
        figura_separabilidad(sid, Xs, y, clases)
    print("Listo.")


if __name__ == "__main__":
    main()
