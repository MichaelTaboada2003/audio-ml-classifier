# Clasificador de Emociones en Voz
## Reconocimiento acustico de emociones con features clasicos y wav2vec2

Proyecto de clasificacion de emociones a partir de audio de voz.
Evalua dos enfoques: features acusticos manuales (144 dims) y embeddings
de `facebook/wav2vec2-base` (768 dims), ambos con Leave-One-Out CV.

---

## Resultado principal

Con el dataset filtrado y balanceado (top-13 por clase, Enojo vs Tristeza - 26 audios en total):

| Enfoque | Mejor modelo | Accuracy | Balanced Acc |
|---|---|---|---|
| Features manuales (144 dims) | SVM lineal / Random Forest | **0.85** | **0.8462** |
| wav2vec2 embeddings (768 dims) | SVM lineal / Regresión Logística | **0.92** | **0.9231** |

Con la ampliación de la muestra a 13 audios por clase, el modelo de representación profunda (*wav2vec2*) logra extraer mejores descriptores prosódicos y tímbricos, superando a los features acústicos clásicos en un **7.69%** de Balanced Accuracy.

### Gráficos de Resultados y Comparativa

A continuación se presentan las figuras generadas a partir del entrenamiento y evaluación del clasificador (`outputs/figuras/`):

#### 1. Matriz de Confusión y Desempeño (wav2vec2)
![Resultados v2](outputs/figuras/resultados_v2.png)

#### 2. Comparativa Features Clásicos vs wav2vec2
![Comparativa Features vs wav2vec2](outputs/figuras/comparativa_features_vs_wav2vec2.png)

---

## Aplicación Web Interactiva

Para tangibilizar los resultados del modelo, se desarrolló una aplicación web interactiva (Flask + HTML5 + CSS3 + JS Vanilla) con un diseño moderno y responsivo basado en *Glassmorphism*.

### Características principales:
* **Entrada de audio dual**: Permite subir archivos de audio (WAV, MP3, OGG, etc.) o grabar directamente desde el micrófono en tiempo real con una animación de onda y temporizador.
* **Selector interactivo de modelos**: Panel en cuadrícula que permite alternar instantáneamente entre los 6 clasificadores entrenados (SVM lineal, Regresión Logística, SVM RBF, Random Forest, KNN k=3 y KNN k=5) cargados desde archivos binarios `.joblib`.
* **Medidor de confianza SVG**: Un anillo de progreso vectorial animado que dibuja la confianza de la predicción y adopta colores y halos de brillo neón según la emoción predicha (Rojo para Enojo, Cian para Tristeza).
* **Explorador interactivo del dataset**: Lista interactiva para reproducir y testear individualmente los 26 mejores audios del dataset.
* **Análisis de Métricas en Tiempo Real**: Muestra la duración del audio, la energía (RMS), el Pitch medio (Hz) y los cruces por cero (ZCR) extraídos dinámicamente en el backend.

### Capturas de Pantalla de la Interfaz

#### 1. Panel de Control y Selector de Modelos
![Selector de Modelos](imgs/models.png)

#### 2. Resultado del Análisis y Medidor de Confianza SVG
![Resultado del Análisis](imgs/result.png)

#### 3. Explorador del Dataset Top-26
![Explorador del Dataset](imgs/audios.png)

---

## Dataset

| Parametro | Valor |
|---|---|
| Total de audios | 146 |
| Clases originales | Aburrido (36), Enojo (37), Tranquilidad (36), Tristeza (37) |
| Recolectores | MT, VZ, VA, ED |
| Duracion por audio | ~10-30 segundos |
| Formatos | .ogg, .mp4, .mpeg |

### Audios utiles por clase (score > 0.35)

| Clase | Total | Pasan (>0.35) | Excelentes (>0.45) |
|---|---|---|---|
| Enojo | 37 | 36 | 33 |
| Tranquilidad | 36 | 27 | 14 |
| Tristeza | 37 | 11 | 4 |

Tristeza es el cuello de botella: los recolectores VA y ED no expresaron
la emocion con suficiente intensidad acustica (energia muy alta, pitch
muy variable para ser tristeza).

---

## Estructura del proyecto

```
clasificador-audios/
├── app.py                                     # Servidor Flask (Backend API)
├── templates/
│   └── index.html                             # Interfaz de la aplicación (Frontend)
├── static/
│   ├── css/
│   │   └── style.css                          # Estilos CSS (Glassmorphism & Grids)
│   └── js/
│       └── main.js                            # Lógica interactiva en el cliente
├── notebooks/
│   ├── 01_clasificador_v1_features.ipynb      # Features manuales (144 dims)
│   ├── 02_clasificador_v1_embeddings.ipynb    # wav2vec2 embeddings (768 dims)
│   └── 03_clasificador_v2_balanceado.ipynb    # Comparativa final + conclusiones
├── scripts/
│   ├── filtrar_audios.py                      # Scoring acústico por audio
│   └── exportar_modelos.py                    # Script de entrenamiento y serialización de modelos
├── outputs/
│   ├── reporte_filtrado_v2.csv                # Scores de los 146 audios
│   ├── modelos/                               # Clasificadores serializados (.joblib)
│   │   ├── knn_k3.joblib
│   │   ├── knn_k5.joblib
│   │   ├── svm_lineal.joblib
│   │   ├── svm_rbf.joblib
│   │   ├── logreg.joblib
│   │   └── rf.joblib
│   └── figuras/                               # Gráficos de resultados y análisis
├── data/                                      # (no versionado)
│   └── AUDIOS MACHINE LEARNING/               # Dataset original (146 audios)
├── imgs/                                      # Capturas de la interfaz web
├── .gitignore
└── README.md
```

---

## Pipeline

```
1. Scoring acustico (scripts/filtrar_audios.py)
   - Calcula 4 scores por audio: enojo, tristeza, tranquilidad, neutro
   - NO reasigna etiquetas — cada audio mantiene su clase original
   - Genera reporte con ranking completo y estado (EXCELENTE/PASA/borderline/NO PASA)

2. Seleccion por ranking (en los notebooks)
   - Top-N audios de cada clase ordenados por su score correspondiente
   - Permite elegir la configuracion segun el experimento

3. Extraccion de features
   - Features manuales: pitch, energia, ZCR, spectral, MFCCs (144 dims)
   - wav2vec2: mean-pooling de la ultima capa oculta (768 dims)

4. Clasificacion con LOOCV
   - KNN, SVM lineal, SVM RBF, LogReg, Random Forest
   - Metrica principal: Balanced Accuracy
```

---

## Uso rápido

```bash
# 1. Activar entorno
source ../.venv/bin/activate

# 2. Calcular scores de todos los audios (opcional)
python scripts/filtrar_audios.py

# 3. Entrenar y exportar modelos serializados (.joblib)
python scripts/exportar_modelos.py

# 4. Iniciar la aplicación web interactiva
python app.py

# 5. Abrir en el navegador
# La interfaz estará disponible en http://127.0.0.1:5001
```

---

## Hallazgos del proceso

### Por que solo 2 clases funcionan bien?

El dataset tiene 4 clases originales. Tres de ellas (Aburrido, Tranquilidad,
Tristeza) son acusticamente similares — baja energia, pitch bajo, poca
variabilidad. Solo Enojo tiene una firma acustica marcada y consistente.

Al reducir a Enojo vs Tristeza (las dos mas opuestas), el problema se vuelve
resoluble con cualquier clasificador.

### ¿Por qué wav2vec2 supera a los features manuales?

Al ampliar el dataset a 26 muestras (13 por clase), los embeddings de wav2vec2 logran capitalizar su preentrenamiento en 960h de voz (LibriSpeech) para capturar dinámicas prosódicas y tímbricas de más alto nivel, logrando un **92.31%** de balanced accuracy frente al **84.62%** de los features acústicos tradicionales.

### Por que el filtrado es importante?

Muchos audios etiquetados como "Tristeza" o "Enojo" suenan neutros — la persona
no expreso la emocion con suficiente intensidad. El script de scoring permite
identificar y descartar esos audios antes de entrenar.

### Tristeza: el problema de los nuevos recolectores

Con 2 recolectores (MT y VZ): 10-11 audios de tristeza utiles.
Con 4 recolectores (MT, VZ, VA, ED): sigue habiendo 11 utiles.

VA y ED grabaron tristeza con voz activa (energia -22 a -27 dB, pitch variable),
cuando la firma acustica de tristeza requiere voz plana y baja (energia < -35 dB,
pitch_std < 2). Sus grabaciones no tienen la expresividad emocional necesaria.

---

## Proximos pasos

- Agregar clase Felicidad (alta energia + pitch alto + brillo espectral alto)
  para tener 3 clases acusticamente distintas: Enojo / Felicidad / Tranquilidad
- Regrabar audios de Tristeza con VA y ED con instrucciones mas especificas
  (voz mas lenta, mas baja, mas monotona)
- Evaluar con mas datos si wav2vec2 supera a features manuales

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
Dataset recolectado por 4 personas (MT, VZ, VA, ED), ~21 hablantes distintos
por recolector, ~10-30 segundos por audio.
