# Clasificador de Emociones en Voz
## Enojo vs Tristeza — wav2vec2 embeddings

Clasificador binario de emociones a partir de audio usando embeddings
de `facebook/wav2vec2-base`. Alcanza **90% accuracy** con Leave-One-Out CV
sobre un dataset balanceado de 20 audios (10 por clase).

---

## Resultado principal

| Metrica | Valor |
|---|---|
| Mejor modelo | Random Forest |
| Accuracy | 0.900 |
| Balanced Accuracy | 0.900 |
| Recall Enojo | 1.00 |
| Recall Tristeza | 0.80 |

---

## Estructura del proyecto

```
clasificador-audios/
├── notebooks/
│   ├── 01_clasificador_v1_features.ipynb      # Iteracion 1: features manuales
│   ├── 02_clasificador_v1_embeddings.ipynb    # Iteracion 2: wav2vec2 (4 clases)
│   └── 03_clasificador_v2_balanceado.ipynb    # Version final: Enojo vs Tristeza
├── scripts/
│   └── filtrar_audios.py                      # Filtrado acustico del dataset
├── outputs/
│   ├── reporte_filtrado_v2.csv                # Detalle del filtrado por audio
│   └── figuras/                               # Graficos generados
├── data/                                      # (no versionado)
│   ├── AUDIOS MACHINE LEARNING/               # Dataset original (84 audios)
│   └── AUDIOS_FILTRADOS_V2/                   # Dataset filtrado (Enojo/Tristeza/Tranquilidad)
├── .gitignore
└── README.md
```

---

## Pipeline

```
1. Filtrado acustico
   - Calcula scores de enojo/tristeza/neutro por audio
   - Descarta audios sin firma emocional clara → Tranquilidad
   - Fusiona clase "Aburrido" con Tranquilidad (acusticamente indistinguibles)

2. Seleccion balanceada
   - Top 10 Enojo por score_enojo
   - Top 10 Tristeza por score_tristeza

3. Extraccion de embeddings
   - wav2vec2-base (768 dims, mean-pooling temporal)
   - 10 segundos por audio, 16kHz

4. Clasificacion
   - LOOCV con RF, SVM, LogReg, KNN
   - RF: 90% accuracy
```

---

## Uso rapido

```bash
# Activar entorno
source ../.venv/bin/activate

# 1. Filtrar audios (dry-run primero)
python scripts/filtrar_audios.py

# 2. Aplicar filtrado (copia, preserva originales)
python scripts/filtrar_audios.py --apply --copy

# 3. Ejecutar notebook principal
jupyter notebook notebooks/03_clasificador_v2_balanceado.ipynb
```

---

## Hallazgos del proceso

### Por que solo 2 clases?

El dataset original tenia 4 clases (Aburrido, Enojo, Tranquilidad, Tristeza).
Tres de ellas (Aburrido, Tranquilidad, Tristeza) son acusticamente muy similares
— baja energia, pitch bajo, poca variabilidad. Solo Enojo tiene una firma
acustica marcada.

Con 3 clases (10x10x10) el mejor modelo llega a 70%. Con 2 clases (Enojo vs
Tristeza) sube a 90%.

### Por que wav2vec2?

Los features manuales (MFCCs, pitch, energia, spectral) no superaban el azar
con 4 clases. wav2vec2 fue preentrenado en 960h de voz humana y captura
patrones prosodicos que los features clasicos no detectan.

### Por que filtrar?

Muchos audios etiquetados como "Enojo" o "Tristeza" son acusticamente neutros
(la persona no expreso la emocion con suficiente intensidad). El filtrado
descarta estos audios para que el clasificador aprenda de ejemplos claros.

---

## Dependencias

```
librosa>=0.10
torch>=2.0
transformers>=4.30
scikit-learn>=1.3
numpy
pandas
matplotlib
```

---

## Contexto academico

Proyecto del curso Intro-Machine-Learning (pregrado).
Dataset recolectado por 2 personas (MT y VZ), 21 hablantes distintos,
~26 segundos por audio.
