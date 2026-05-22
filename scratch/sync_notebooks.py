import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def lines(text: str):
    return [line for line in text.splitlines(keepends=True)]


def load_notebook(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(path: Path, nb):
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")


def set_cell_source(nb, index: int, text: str):
    nb["cells"][index]["source"] = lines(text)


def clear_code_outputs(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def sync_nb1():
    path = NOTEBOOKS / "01_clasificador_v1_features.ipynb"
    nb = load_notebook(path)

    set_cell_source(
        nb,
        0,
        """# Clasificador de Voz — Features acústicos manuales
## Enojo, Tristeza y Feliz · Dataset filtrado (10 por clase)

**Objetivo:** comparar la separabilidad emocional con features acústicos clásicos.  
**Dataset:** 30 audios filtrados por evidencia acústica (10 por clase).  
**Evaluación:** Leave-One-Out CV (LOOCV).

| Sección | Contenido |
|---|---|
| 1–3 | Setup, extracción de features, carga del dataset |
| 4 | EDA: distribución, boxplots, espectrogramas |
| 5 | Modelado LOOCV |
| 6 | Diagnóstico: ¿recolector o emoción? |
| 7 | Separabilidad PCA / t-SNE |
| 8 | Conclusiones |
""",
    )

    set_cell_source(
        nb,
        1,
        """import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = os.path.join('..', 'data', 'AUDIOS_FILTRADOS_V2')
REPORTE  = os.path.join('..', 'outputs', 'reporte_filtrado_v2.csv')
CLASES   = ['Enojo', 'Tristeza', 'Feliz']
COLORES  = {'Enojo': '#DD8452', 'Tristeza': '#4C72B0', 'Feliz': '#E377C2'}
SCORE_MAP = {'Enojo': 'score_enojo', 'Tristeza': 'score_tristeza', 'Feliz': 'score_feliz'}
EXTS     = {'.ogg','.mp3','.mp4','.mpeg','.wav','.flac','.m4a'}
SR       = 22050
N_MFCC   = 13
HOP      = 1024
N_PER_CLASS = 10

print('Configuración lista.')
print(f'Dataset: {DATA_DIR}')
print(f'Clases activas: {CLASES}')
""",
    )

    set_cell_source(
        nb,
        5,
        """df_rep = pd.read_csv(REPORTE)

registros = []
seleccion_por_clase = {}

for clase in CLASES:
    score_col = SCORE_MAP[clase]
    carpeta = os.path.join(DATA_DIR, clase)
    seleccion = (
        df_rep[
            (df_rep['clase_original'] == clase)
            & df_rep['archivo'].apply(lambda x: os.path.exists(os.path.join(carpeta, x)))
        ]
        .sort_values(score_col, ascending=False)
        .head(N_PER_CLASS)['archivo']
        .tolist()
    )
    seleccion_por_clase[clase] = seleccion
    print(f'{clase:<9} ({len(seleccion)}): {seleccion}')

for clase in CLASES:
    for nombre in seleccion_por_clase[clase]:
        ruta = os.path.join(DATA_DIR, clase, nombre)
        if not os.path.exists(ruta):
            print(f'  [WARN] No encontrado: {ruta}')
            continue
        vec = extraer_features(ruta)
        if vec is not None:
            registros.append({'archivo': nombre, 'clase': clase,
                               'recolector': nombre[:2], 'features': vec})
            print(f'  OK  {clase}/{nombre}', flush=True)

X   = np.vstack([r['features']  for r in registros])
y   = np.array([r['clase']      for r in registros])
rec = np.array([r['recolector'] for r in registros])
le  = LabelEncoder()
y_enc = le.fit_transform(y)
X_sc  = StandardScaler().fit_transform(X)

print(f'Dataset: {X.shape[0]} muestras x {X.shape[1]} features')
print(f'Clases: {dict(zip(*np.unique(y, return_counts=True)))}')
print(f'Recolectores: {dict(zip(*np.unique(rec, return_counts=True)))}')
""",
    )

    set_cell_source(
        nb,
        11,
        """loo = LeaveOneOut()
modelos = {
    'Baseline':   Pipeline([('s', StandardScaler()), ('c', DummyClassifier(strategy='most_frequent'))]),
    'KNN (k=3)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=3))]),
    'KNN (k=5)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=5))]),
    'SVM lineal': Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear', C=1, class_weight='balanced', random_state=42))]),
    'SVM RBF':    Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42))]),
    'LogReg':     Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1, class_weight='balanced', random_state=42))]),
    'RF':         Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))]),
}

chance = 1.0 / len(CLASES)
print('LOOCV — Features manuales (' + str(X.shape[1]) + ' dims)')
print(f'Chance baseline: {chance:.3f} ({len(CLASES)} clases balanceadas)')
print('{:<15} {:>8} {:>8} {:>10}  {}'.format('Modelo','Acc','BalAcc','Errores','vs chance'))
print('-'*58)

res_manual = {}
for nombre, pipe in modelos.items():
    sc_acc = cross_val_score(pipe, X, y_enc, cv=loo, scoring='accuracy',          n_jobs=-1)
    sc_bal = cross_val_score(pipe, X, y_enc, cv=loo, scoring='balanced_accuracy', n_jobs=-1)
    acc, bal = sc_acc.mean(), sc_bal.mean()
    err   = int(round((1 - acc) * len(y)))
    delta = bal - chance
    res_manual[nombre] = bal
    marca = ' ▲' if delta > 0.10 else (' ▼' if delta < -0.02 else '')
    print('{:<15} {:>8.4f} {:>8.4f} {:>6}/{:}  {:+.4f}{}'.format(nombre, acc, bal, err, len(y), delta, marca))
""",
    )

    set_cell_source(
        nb,
        14,
        """pipe_diag = Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear', C=1, random_state=42))])
y_rec_enc = LabelEncoder().fit_transform(rec)

acc_emo = cross_val_score(pipe_diag, X, y_enc,     cv=loo, scoring='accuracy', n_jobs=-1).mean()
acc_rec = cross_val_score(pipe_diag, X, y_rec_enc, cv=loo, scoring='accuracy', n_jobs=-1).mean()

n_clases_emo = len(np.unique(y_enc))
n_clases_rec = len(np.unique(y_rec_enc))
chance_emo = 1.0 / n_clases_emo
chance_rec = 1.0 / n_clases_rec

fig, ax = plt.subplots(figsize=(7, 4))
etiquetas = [f'Clasificar EMOCION\\n({n_clases_emo} clases, chance {chance_emo*100:.1f}%)',
             f'Clasificar RECOLECTOR\\n({n_clases_rec} clases, chance {chance_rec*100:.1f}%)']
bars = ax.bar(etiquetas, [acc_emo, acc_rec],
              color=['#DD8452', '#4C72B0'], alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(bars, [acc_emo, acc_rec]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            str(round(v,3)), ha='center', fontsize=13, fontweight='bold')
ax.axhline(chance_emo, xmin=0.05, xmax=0.45, color='gray', linestyle='--', linewidth=1.2, label=f'Chance Emo ({chance_emo*100:.1f}%)')
ax.axhline(chance_rec, xmin=0.55, xmax=0.95, color='black', linestyle=':', linewidth=1.2, label=f'Chance Rec ({chance_rec*100:.1f}%)')
ax.set_ylim(0, 1.1)
ax.set_title('SVM lineal — que aprenden los features manuales?', fontsize=12)
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.show()

print(f'Clasificar EMOCION    (chance {chance_emo*100:.1f}%): ' + str(round(acc_emo,4)))
print(f'Clasificar RECOLECTOR (chance {chance_rec*100:.1f}%): ' + str(round(acc_rec,4)))
if acc_rec > acc_emo + 0.1:
    print('Los features capturan mas al recolector que la emocion.')
else:
    print('Los features capturan emocion, no solo identidad del recolector.')
""",
    )

    clear_code_outputs(nb)
    save_notebook(path, nb)


def sync_nb2():
    path = NOTEBOOKS / "02_clasificador_v1_embeddings.ipynb"
    nb = load_notebook(path)

    set_cell_source(
        nb,
        0,
        """# Clasificador de Voz — Embeddings wav2vec2
## Enojo, Tristeza y Feliz · Dataset filtrado (10 por clase)

Usa **facebook/wav2vec2-base** como extractor de embeddings de 768 dimensiones.

| Parámetro | Valor |
|---|---|
| Modelo | `facebook/wav2vec2-base` |
| Sample rate | 16 kHz |
| Embedding | 768 dims (mean-pooling temporal) |
| Dataset | 10 audios por clase |
| Evaluación | LOOCV |
""",
    )

    set_cell_source(
        nb,
        1,
        """import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR  = os.path.join('..', 'data', 'AUDIOS_FILTRADOS_V2')
REPORTE   = os.path.join('..', 'outputs', 'reporte_filtrado_v2.csv')
CLASES    = ['Enojo', 'Tristeza', 'Feliz']
COLORES   = {'Enojo': '#DD8452', 'Tristeza': '#4C72B0', 'Feliz': '#E377C2'}
SCORE_MAP = {'Enojo': 'score_enojo', 'Tristeza': 'score_tristeza', 'Feliz': 'score_feliz'}
TARGET_SR = 16000
CACHE     = os.path.join('..', 'outputs', 'embeddings_wav2vec2.npz')
N_PER_CLASS = 10

DEVICE = 'mps'  if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
print(f'PyTorch: {torch.__version__}')
print(f'Clases activas: {CLASES}')
""",
    )

    set_cell_source(
        nb,
        5,
        """def get_embedding(ruta):
    try:
        y, _ = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=10.0)
        if len(y) < TARGET_SR * 0.3: return None
        inputs = processor(y, sampling_rate=TARGET_SR, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = model(inputs.input_values.to(DEVICE))
        return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f'  [ERROR] {os.path.basename(ruta)}: {e}')
        return None

df_rep = pd.read_csv(REPORTE)
seleccion_por_clase = {}
for clase in CLASES:
    score_col = SCORE_MAP[clase]
    carpeta = os.path.join(DATA_DIR, clase)
    seleccion = (
        df_rep[
            (df_rep['clase_original'] == clase)
            & df_rep['archivo'].apply(lambda x: os.path.exists(os.path.join(carpeta, x)))
        ]
        .sort_values(score_col, ascending=False)
        .head(N_PER_CLASS)['archivo']
        .tolist()
    )
    seleccion_por_clase[clase] = seleccion
    print(f'{clase:<9} ({len(seleccion)}): {seleccion}')

if os.path.exists(CACHE):
    print(f'Cargando embeddings desde caché: {CACHE}')
    c = np.load(CACHE, allow_pickle=True)
    X, y, rec = c['X'], c['y'], c['rec']
    esperadas = N_PER_CLASS * len(CLASES)
    if len(y) != esperadas:
        print('  Caché tiene tamaño distinto, recalculando...')
        os.remove(CACHE)

if not os.path.exists(CACHE):
    print('Extrayendo embeddings...')
    regs = []
    plan = [(archivo, clase) for clase in CLASES for archivo in seleccion_por_clase[clase]]
    for i, (archivo, clase) in enumerate(plan, 1):
        ruta = os.path.join(DATA_DIR, clase, archivo)
        print(f'  [{i:2d}/{len(plan)}] {clase}/{archivo}', flush=True)
        emb = get_embedding(ruta)
        if emb is not None:
            regs.append({'etiqueta': clase, 'rec': archivo[:2], 'emb': emb})
    X   = np.vstack([r['emb']     for r in regs])
    y   = np.array([r['etiqueta'] for r in regs])
    rec = np.array([r['rec']      for r in regs])
    np.savez(CACHE, X=X, y=y, rec=rec)
    print(f'Embeddings guardados en {CACHE}')

le    = LabelEncoder()
y_enc = le.fit_transform(y)
y_rec = LabelEncoder().fit_transform(rec)
print('Dataset: ' + str(dict(zip(*np.unique(y, return_counts=True)))))
""",
    )

    set_cell_source(
        nb,
        11,
        """modelos = {
    'Baseline':   Pipeline([('s', StandardScaler()), ('c', DummyClassifier(strategy='most_frequent'))]),
    'KNN (k=3)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=3))]),
    'KNN (k=5)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=5))]),
    'SVM lineal': Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear', C=1, random_state=42))]),
    'SVM RBF':    Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))]),
    'LogReg':     Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1, random_state=42))]),
    'RF':         Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(n_estimators=200, random_state=42))]),
}

chance = 1.0 / len(CLASES)
print('LOOCV — wav2vec2 embeddings (' + str(X.shape[1]) + ' dims)')
print(f'Chance baseline: {chance:.3f} ({len(CLASES)} clases balanceadas)')
print('{:<15} {:>8} {:>8} {:>10}  {}'.format('Modelo','Acc','BalAcc','Errores','vs chance'))
print('-'*58)

res_emb = {}
for nombre, pipe in modelos.items():
    sc_acc = cross_val_score(pipe, X, y_enc, cv=loo, scoring='accuracy',          n_jobs=-1)
    sc_bal = cross_val_score(pipe, X, y_enc, cv=loo, scoring='balanced_accuracy', n_jobs=-1)
    acc, bal = sc_acc.mean(), sc_bal.mean()
    err   = int(round((1 - acc) * len(y)))
    delta = bal - chance
    res_emb[nombre] = bal
    marca = ' ▲' if delta > 0.10 else (' ▼' if delta < -0.02 else '')
    print('{:<15} {:>8.4f} {:>8.4f} {:>6}/{:}  {:+.4f}{}'.format(nombre, acc, bal, err, len(y), delta, marca))
""",
    )

    set_cell_source(
        nb,
        12,
        """mejor  = max({k: v for k, v in res_emb.items() if k != 'Baseline'}, key=lambda k: res_emb[k])
pipe_m = modelos[mejor]
y_pred = np.zeros(len(y_enc), dtype=int)
for tr, te in loo.split(X):
    pipe_m.fit(X[tr], y_enc[tr])
    y_pred[te] = pipe_m.predict(X[te])

print('Mejor modelo: ' + mejor + '  (BalAcc=' + str(round(res_emb[mejor],4)) + ')')
print(classification_report(y_enc, y_pred, target_names=le.classes_, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
cm = confusion_matrix(y_enc, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=axes[0], cmap='Blues', colorbar=False)
titulo_cm = mejor + ' — LOOCV (' + ', '.join(CLASES) + ', wav2vec2)'
axes[0].set_title(titulo_cm, fontsize=11)

nombres_g = list(res_emb.keys())
vals_g    = [res_emb[n] for n in nombres_g]
best_val  = max(v for k, v in res_emb.items() if k != 'Baseline')
colores_g = ['#999' if n == 'Baseline' else '#2196F3' if res_emb[n] == best_val else '#90CAF9' for n in nombres_g]
axes[1].barh(nombres_g, vals_g, color=colores_g)
axes[1].axvline(1.0 / len(CLASES), color='red', linestyle='--', linewidth=1, label='chance')
axes[1].set_xlabel('Balanced Accuracy')
axes[1].set_title('Comparativa — wav2vec2 embeddings')
axes[1].legend(fontsize=9)
axes[1].spines[['top','right']].set_visible(False)
for i, v in enumerate(vals_g):
    axes[1].text(v + 0.005, i, str(round(v,3)), va='center', fontsize=9)
plt.tight_layout()
plt.show()
""",
    )

    clear_code_outputs(nb)
    save_notebook(path, nb)


def sync_nb3():
    path = NOTEBOOKS / "03_clasificador_v2_balanceado.ipynb"
    nb = load_notebook(path)

    set_cell_source(
        nb,
        0,
        """# Clasificador de Emociones en Voz — v2
## Enojo, Tristeza y Feliz · wav2vec2 embeddings · Dataset balanceado 10x10x10

**Resultado:** notebook de comparación final para ejecutar un experimento multiclase consistente.

| Parametro | Valor |
|---|---|
| Modelo de embeddings | `facebook/wav2vec2-base` (768 dims) |
| Dataset | 10 audios por clase seleccionados por score |
| Evaluacion | Leave-One-Out Cross-Validation |
| Clases | Enojo, Tristeza, Feliz |

### Pipeline

1. Filtrado acustico (`scripts/filtrar_audios.py`) descarta audios sin firma emocional
2. Seleccion de los mejores audios por clase segun su score correspondiente
3. Extraccion de embeddings wav2vec2 (mean-pooling temporal)
4. Clasificacion con LOOCV
""",
    )

    set_cell_source(
        nb,
        1,
        """import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              classification_report, balanced_accuracy_score)
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR  = os.path.join('..', 'data', 'AUDIOS MACHINE LEARNING')
REPORTE   = os.path.join('..', 'outputs', 'reporte_filtrado_v2.csv')
CACHE     = os.path.join('..', 'outputs', 'embeddings_v2.npz')
CLASES    = ['Enojo', 'Tristeza', 'Feliz']
COLORES   = {'Enojo': '#DD8452', 'Tristeza': '#4C72B0', 'Feliz': '#E377C2'}
SCORE_MAP = {'Enojo': 'score_enojo', 'Tristeza': 'score_tristeza', 'Feliz': 'score_feliz'}
TARGET_SR = 16000
DURATION  = 10.0
N_PER_CLASS = 10

DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)
print('Clases activas:', CLASES)
""",
    )

    set_cell_source(
        nb,
        3,
        """df_rep = pd.read_csv(REPORTE)

seleccion_por_clase = {}
for clase in CLASES:
    score_col = SCORE_MAP[clase]
    seleccion = (
        df_rep[df_rep['clase_original'] == clase]
        .sort_values(score_col, ascending=False)
        .head(N_PER_CLASS)['archivo']
        .tolist()
    )
    seleccion_por_clase[clase] = seleccion
    print('{} ({}):\\n  {}'.format(clase, len(seleccion), seleccion))

print('\\nTotal: {} audios balanceados'.format(sum(len(v) for v in seleccion_por_clase.values())))
""",
    )

    set_cell_source(
        nb,
        5,
        """def cargar_modelo():
    print('Cargando facebook/wav2vec2-base...')
    t0 = time.time()
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base')
    model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(DEVICE)
    model.eval()
    print('  Listo en {:.1f}s ({:.1f}M params)'.format(
        time.time() - t0, sum(p.numel() for p in model.parameters()) / 1e6))
    return processor, model


def extraer_embedding(ruta, processor, model):
    y, _ = librosa.load(ruta, sr=TARGET_SR, mono=True, duration=DURATION)
    if len(y) < TARGET_SR * 0.3:
        return None
    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors='pt', padding=True)
    with torch.no_grad():
        out = model(inputs.input_values.to(DEVICE))
    return out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().astype(np.float32)


cache_cargada = False
esperadas = N_PER_CLASS * len(CLASES)
if os.path.exists(CACHE):
    c = np.load(CACHE, allow_pickle=True)
    if len(c['y']) == esperadas:
        print('Cargando embeddings desde cache:', CACHE)
        X, y_labels, recolectores = c['X'], c['y'], c['rec']
        cache_cargada = True
    else:
        print('La cache tiene {} muestras, pero se esperan {}. Recalculando...'.format(len(c['y']), esperadas))

if not cache_cargada:
    processor, model = cargar_modelo()
    registros = []
    plan = [(archivo, clase) for clase in CLASES for archivo in seleccion_por_clase[clase]]
    for i, (archivo, clase) in enumerate(plan, 1):
        ruta = os.path.join(DATA_DIR, clase, archivo)
        print('  [{:2d}/{}] {}/{}'.format(i, len(plan), clase, archivo), flush=True)
        emb = extraer_embedding(ruta, processor, model)
        if emb is not None:
            registros.append({'etiqueta': clase, 'rec': archivo[:2], 'emb': emb})

    X = np.vstack([r['emb'] for r in registros])
    y_labels = np.array([r['etiqueta'] for r in registros])
    recolectores = np.array([r['rec'] for r in registros])
    np.savez(CACHE, X=X, y=y_labels, rec=recolectores)
    print('Embeddings guardados en', CACHE)

le = LabelEncoder()
y_enc = le.fit_transform(y_labels)
print('\\nDataset: {} muestras x {} dims'.format(X.shape[0], X.shape[1]))
print('Clases:', dict(zip(*np.unique(y_labels, return_counts=True))))
""",
    )

    set_cell_source(
        nb,
        9,
        """loo = LeaveOneOut()
modelos = {
    'Baseline':   Pipeline([('s', StandardScaler()), ('c', DummyClassifier(strategy='most_frequent'))]),
    'KNN (k=3)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=3))]),
    'KNN (k=5)':  Pipeline([('s', StandardScaler()), ('c', KNeighborsClassifier(n_neighbors=5))]),
    'SVM lineal': Pipeline([('s', StandardScaler()), ('c', SVC(kernel='linear', C=1, random_state=42))]),
    'SVM RBF':    Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', C=10, gamma='scale', random_state=42))]),
    'LogReg':     Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1, random_state=42))]),
    'RF':         Pipeline([('s', StandardScaler()), ('c', RandomForestClassifier(n_estimators=200, random_state=42))]),
}

chance = 1.0 / len(CLASES)
print('LOOCV — wav2vec2 embeddings ({} dims)'.format(X.shape[1]))
print('Chance: {:.3f} ({} clases balanceadas)\\n'.format(chance, len(CLASES)))
print('{:<15} {:>8} {:>8} {:>8}  {}'.format('Modelo', 'Acc', 'BalAcc', 'Errores', 'vs chance'))
print('-' * 58)

resultados = {}
for nombre, pipe in modelos.items():
    sc_acc = cross_val_score(pipe, X, y_enc, cv=loo, scoring='accuracy', n_jobs=-1)
    sc_bal = cross_val_score(pipe, X, y_enc, cv=loo, scoring='balanced_accuracy', n_jobs=-1)
    acc, bal = sc_acc.mean(), sc_bal.mean()
    err = int(round((1 - acc) * len(y_enc)))
    delta = bal - chance
    resultados[nombre] = bal
    marca = ' ***' if delta > 0.30 else (' ++' if delta > 0.10 else '')
    print('{:<15} {:>8.4f} {:>8.4f} {:>4}/{:<4} {:+.4f}{}'.format(
        nombre, acc, bal, err, len(y_enc), delta, marca))
""",
    )

    set_cell_source(
        nb,
        11,
        """mejor = max({k: v for k, v in resultados.items() if k != 'Baseline'},
            key=lambda k: resultados[k])
pipe_m = modelos[mejor]

y_pred = np.zeros(len(y_enc), dtype=int)
for tr, te in loo.split(X):
    pipe_m.fit(X[tr], y_enc[tr])
    y_pred[te] = pipe_m.predict(X[te])

print('Mejor modelo: {} (BalAcc={:.4f})'.format(mejor, resultados[mejor]))
print()
print(classification_report(y_enc, y_pred, target_names=le.classes_, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

cm = confusion_matrix(y_enc, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(
    ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title(mejor + ' — LOOCV\\n(' + ', '.join(CLASES) + ', wav2vec2)', fontsize=11)

nombres = list(resultados.keys())
vals = [resultados[n] for n in nombres]
best_val = max(v for k, v in resultados.items() if k != 'Baseline')
colores = ['#999' if n == 'Baseline'
           else '#2196F3' if resultados[n] == best_val
           else '#90CAF9' for n in nombres]
axes[1].barh(nombres, vals, color=colores)
axes[1].axvline(1.0 / len(CLASES), color='red', linestyle='--', linewidth=1, label='chance')
axes[1].set_xlabel('Balanced Accuracy')
axes[1].set_title('Comparativa de modelos')
axes[1].legend(fontsize=9)
axes[1].spines[['top', 'right']].set_visible(False)
for i, v in enumerate(vals):
    axes[1].text(v + 0.005, i, '{:.3f}'.format(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join('..', 'outputs', 'figuras', 'resultados_v2.png'),
            dpi=120, bbox_inches='tight')
plt.show()
""",
    )

    set_cell_source(
        nb,
        13,
        """# --- Extraer features manuales sobre los mismos audios ---
import librosa

SR_MAN, HOP = 22050, 1024

def extraer_features_manuales(ruta):
    y, _ = librosa.load(ruta, sr=SR_MAN, mono=True, duration=15.0)
    if len(y) < SR_MAN * 0.1: return None
    f = []
    f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'), hop_length=HOP)
    f0v = f0[f0 > 0] if (f0 > 0).any() else np.array([0.0])
    f += [np.mean(f0v), np.std(f0v), np.percentile(f0v, 25), np.percentile(f0v, 75)]
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    f += [np.mean(rms), np.std(rms), np.mean(rms < 0.1 * np.max(rms))]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]
    f += [np.mean(zcr), np.std(zcr)]
    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_rolloff,
               librosa.feature.spectral_bandwidth]:
        s = fn(y=y, sr=SR_MAN, hop_length=HOP)[0]
        f += [np.mean(s), np.std(s)]
    sct = librosa.feature.spectral_contrast(y=y, sr=SR_MAN, hop_length=HOP)
    f += list(np.mean(sct, 1)) + list(np.std(sct, 1))
    ch = librosa.feature.chroma_stft(y=y, sr=SR_MAN, hop_length=HOP)
    f += list(np.mean(ch, 1)) + list(np.std(ch, 1))
    mfcc = librosa.feature.mfcc(y=y, sr=SR_MAN, n_mfcc=13, hop_length=HOP)
    for p in [10, 25, 50, 75, 90]:
        f += list(np.percentile(mfcc, p, axis=1))
    d1 = librosa.feature.delta(mfcc)
    f += list(np.mean(d1, 1)) + list(np.std(d1, 1))
    return np.array(f, dtype=np.float32)

plan = [(archivo, clase) for clase in CLASES for archivo in seleccion_por_clase[clase]]
regs_man = []
for archivo, clase in plan:
    ruta = os.path.join(DATA_DIR, clase, archivo)
    v = extraer_features_manuales(ruta)
    if v is not None:
        regs_man.append({'etiqueta': clase, 'features': v})

X_man = np.vstack([r['features'] for r in regs_man])
y_man = np.array([r['etiqueta'] for r in regs_man])
y_man_enc = LabelEncoder().fit_transform(y_man)

print('Features manuales: {} muestras x {} dims'.format(X_man.shape[0], X_man.shape[1]))
print('wav2vec2 embeddings: {} muestras x {} dims'.format(X.shape[0], X.shape[1]))
""",
    )

    set_cell_source(
        nb,
        15,
        """# --- Grafico comparativo ---
nombres_g = [n for n in modelos if n != 'Baseline']
vals_man = [res_man[n] for n in nombres_g]
vals_emb = [res_emb[n] for n in nombres_g]

x = np.arange(len(nombres_g))
ancho = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - ancho/2, vals_man, ancho,
               label='Features propios (144 dims)', color='#DD8452', alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + ancho/2, vals_emb, ancho,
               label='wav2vec2 embeddings (768 dims)', color='#4C72B0', alpha=0.85, edgecolor='white')

for bar, v in zip(bars1, vals_man):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            '{:.2f}'.format(v), ha='center', fontsize=9, fontweight='bold')
for bar, v in zip(bars2, vals_emb):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
            '{:.2f}'.format(v), ha='center', fontsize=9, fontweight='bold')

ax.axhline(1.0 / len(CLASES), color='red', linestyle='--', linewidth=1.2, label='Chance (33.3%)')
ax.set_xticks(x)
ax.set_xticklabels(nombres_g, rotation=10, ha='right')
ax.set_ylabel('Balanced Accuracy (LOOCV)')
ax.set_ylim(0, 1.05)
ax.set_title('Features propios vs wav2vec2 — ' + ' / '.join(CLASES) + ' ({})'.format('x'.join([str(N_PER_CLASS)] * len(CLASES))), fontsize=13)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join('..', 'outputs', 'figuras', 'comparativa_features_vs_wav2vec2.png'),
            dpi=120, bbox_inches='tight')
plt.show()
""",
    )

    clear_code_outputs(nb)
    save_notebook(path, nb)


if __name__ == "__main__":
    sync_nb1()
    sync_nb2()
    sync_nb3()
