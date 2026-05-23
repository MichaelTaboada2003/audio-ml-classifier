"""Regenera las figuras del dashboard usando el cache y modelos ACTUALES.

Produce:
  - outputs/figuras/resultados_v2.png   (CM del mejor modelo + barchart LOOCV)
  - outputs/figuras/separabilidad_v2.png (PCA + t-SNE de los embeddings activos)
  - outputs/figuras/holdout_por_modelo.png (recall holdout por modelo y clase)

Usa los .joblib serializados, el cache de embeddings, y reporte_filtrado_v2.csv
para reflejar exactamente lo que el backend está sirviendo.
"""

import json
import os
import sys
import warnings
from collections import Counter

import joblib
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ACTIVE_CLASSES

CACHE = os.path.join("outputs", "embeddings_v2.npz")
PREDS_JSON = os.path.join("outputs", "predicciones_loocv.json")
METRICS = os.path.join("outputs", "model_metrics.json")
REPORTE = os.path.join("outputs", "reporte_filtrado_v2.csv")
FIG_DIR = os.path.join("outputs", "figuras")
DATA_DIR = os.path.join("data", "AUDIOS MACHINE LEARNING")
MODEL_DIR = os.path.join("outputs", "modelos")
TARGET_SR = 16000
MAX_DURATION = 10.0

CLASS_COLOR = {"Enojo": "#DD8452", "Tristeza": "#C44E52", "Feliz": "#E377C2", "Tranquilidad": "#55A868"}
MODEL_ORDER = ["svm_lineal", "logreg", "svm_rbf", "rf", "knn_k5", "knn_k3"]


def figura_resultados():
    """CM del mejor modelo + barchart LOOCV de todos los modelos."""
    with open(METRICS) as f:
        metrics = json.load(f)
    with open(PREDS_JSON) as f:
        preds = json.load(f)

    best_key = metrics["best_model"]
    best_info = preds["models"][best_key]
    classes = best_info["classes_order"]

    y_true = [p["clase_real"] for p in best_info["predictions"]]
    y_pred = [p["prediccion"] for p in best_info["predictions"]]
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Matriz de confusión ---
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"{best_info['label']} — LOOCV (cap N={metrics['samples_per_class']} por clase, balanced)")
    thr = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=13)

    # --- Barchart comparativa ---
    ax = axes[1]
    labels, bals, accs = [], [], []
    for key in MODEL_ORDER:
        m = metrics["models"].get(key)
        if m:
            labels.append(m["label"])
            bals.append(m["balanced_accuracy"])
            accs.append(m["accuracy"])
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, bals, color=["#1f77b4" if k == best_key else "#9ec5dd" for k in MODEL_ORDER if metrics["models"].get(k)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(metrics["chance_accuracy"], color="red", linestyle="--", label=f"chance ({metrics['chance_accuracy']:.3f})")
    ax.set_xlabel("Balanced Accuracy (LOOCV)")
    ax.set_xlim(0, 1.0)
    for bar, v in zip(bars, bals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"Comparativa LOOCV — {len(ACTIVE_CLASSES)} clases · {sum(metrics['models'][k]['accuracy'] is not None for k in metrics['models'])} modelos")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "resultados_v2.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def figura_separabilidad():
    """PCA + t-SNE de los embeddings actuales."""
    c = np.load(CACHE, allow_pickle=True)
    X = c["X"]
    y = c["y"]
    X_sc = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_sc)
    X_tsne = TSNE(n_components=2, perplexity=5, random_state=42, init="pca").fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, proj, titulo in [(axes[0], X_pca, "PCA 2D"), (axes[1], X_tsne, "t-SNE 2D (perplexity=5)")]:
        for clase in ACTIVE_CLASSES:
            mask = y == clase
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=CLASS_COLOR.get(clase, "#777"),
                       label=f"{clase} (n={int(mask.sum())})",
                       s=60, alpha=0.8, edgecolor="white")
        ax.set_title(titulo)
        ax.legend(title="Clase", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    n_per = ", ".join(f"{c}={int(np.sum(y == c))}" for c in ACTIVE_CLASSES)
    fig.suptitle(f"Separabilidad — embeddings wav2vec2 ({n_per})", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "separabilidad_v2.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def figura_holdout():
    """Bar chart recall por modelo y clase sobre el set HOLDOUT (audios no-training)."""
    c = np.load(CACHE, allow_pickle=True)
    training = set(c["files"].tolist())
    rep = pd.read_csv(REPORTE)

    # Embeddings holdout por clase
    print("  Computando embeddings holdout para la figura...")
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    wav = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    wav.eval()

    def emb(ruta):
        y, _ = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=MAX_DURATION)
        inp = proc(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            return wav(inp.input_values).last_hidden_state.mean(dim=1).squeeze().numpy().astype(np.float32)

    holdout_X = {}
    for clase in ACTIVE_CLASSES:
        score_col = {"Enojo": "score_enojo", "Tristeza": "score_tristeza",
                     "Feliz": "score_feliz", "Tranquilidad": "score_tranquilidad"}[clase]
        hold = rep[(rep["clase_original"] == clase) & (~rep["archivo"].isin(training))].sort_values(score_col, ascending=False)
        embs = []
        for _, row in hold.iterrows():
            ruta = os.path.join(DATA_DIR, clase, row["archivo"])
            if os.path.exists(ruta):
                embs.append(emb(ruta))
        holdout_X[clase] = np.vstack(embs) if embs else np.empty((0, 768))

    # Recall por modelo y clase
    fig, ax = plt.subplots(figsize=(11, 5))
    n_classes = len(ACTIVE_CLASSES)
    x = np.arange(len(MODEL_ORDER))
    width = 0.25

    with open(METRICS) as f:
        metrics = json.load(f)

    for i, clase in enumerate(ACTIVE_CLASSES):
        recalls = []
        for k in MODEL_ORDER:
            path = os.path.join(MODEL_DIR, f"{k}.joblib")
            if not os.path.exists(path) or len(holdout_X[clase]) == 0:
                recalls.append(0)
                continue
            clf = joblib.load(path)
            preds = clf.predict(holdout_X[clase])
            recalls.append(float(np.mean(preds == clase)))
        bars = ax.bar(x + (i - 1) * width, recalls, width,
                      color=CLASS_COLOR.get(clase), label=f"{clase} (n={len(holdout_X[clase])})")
        for b, v in zip(bars, recalls):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([metrics["models"][k]["label"] for k in MODEL_ORDER], rotation=15, ha="right")
    ax.set_ylabel("Recall sobre holdout (audios NO usados en training)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Generalización a holdout — recall por modelo y clase")
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "holdout_por_modelo.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Regenerando figuras…")
    figura_resultados()
    figura_separabilidad()
    figura_holdout()
    print("Listo.")
