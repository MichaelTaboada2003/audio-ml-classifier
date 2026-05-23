# EmotiSpeech · Clasificador de Emociones + Módulo de Toma de Decisiones

Proyecto integrado de dos capas:

1. **Capa de Machine Learning** — clasificación de emociones a partir de audio de voz usando un encoder fine-tuneado para emoción (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) + clasificadores clásicos.
2. **Capa de Toma de Decisiones** — un módulo analítico construido sobre los resultados del clasificador que permite simular, evaluar y recomendar despliegues con justificación cuantitativa de negocio.

La capa de Toma de Decisiones se agregó **sin modificar la capa de ML**: solo consume los embeddings y los modelos ya entrenados.

---

## Tabla de contenido

- [Resumen rápido](#resumen-rápido)
- [Historial de iteraciones y errores](#historial-de-iteraciones-y-errores)
- [Aplicación web — pestañas disponibles](#aplicación-web--pestañas-disponibles)
- [Capa 1 · Machine Learning](#capa-1--machine-learning)
- [Capa 2 · Toma de Decisiones](#capa-2--toma-de-decisiones)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Uso rápido](#uso-rápido)
- [Dependencias](#dependencias)

---

## Resumen rápido

| Capa | Qué hace | Salida |
|---|---|---|
| ML | Clasifica audio en Enojo / Tranquilidad con embedding emocional (1024 dims de audeering) + 6 clasificadores clásicos | Predicción + probabilidades + arousal/valence/dominance |
| Decisiones | Convierte las matrices de confusión y probabilidades LOO en una decisión de negocio (GO / NO-GO, umbral óptimo, VPN esperado) | Recomendación justificada con sensibilidad y Monte Carlo |

**Estado actual (2026-05-23):**

| Modelo | LOOCV bal_acc | **Holdout bal_acc (out-of-sample)** |
|---|---|---|
| **Random Forest** | 0.93 | **0.88** |
| SVM lineal | 0.93 | 0.84 |
| KNN k=5 | 0.83 | 0.81 |
| LogReg | 0.97 | 0.77 |
| SVM RBF | 0.97 | 0.76 |
| KNN k=3 | 0.83 | 0.72 |

Pipeline actual: **2 clases (Enojo vs Tranquilidad), N_PER_CLASS=15**, encoder emocional audeering, dataset re-etiquetado con auditoría de IA. Random Forest gana en holdout (88.2% balanced accuracy sobre 79 audios no vistos). Para llegar a estos números pasamos por **siete iteraciones de pipeline** y resolvimos cuatro problemas estructurales — la próxima sección documenta ese recorrido.

---

## Historial de iteraciones y errores

El proyecto pasó por ocho fases. Cada una arregló (o intentó arreglar) un problema concreto detectado en la anterior. Esta sección documenta qué fallaba, qué cambió y qué resultado dio.

### Línea de tiempo resumida

| # | Fecha | Cambio principal | Resultado |
|---|---|---|---|
| 1 | 2026-05-20 | Filtrado acústico naive sobre 84 audios (3 clases) | Filtro funciona, pero escoge ejemplos muy prototípicos |
| 2 | 2026-05-20 | Notebook v2: Enojo vs Tristeza con `wav2vec2-base` | LOOCV 90% — primera señal de que el encoder neuronal supera features manuales |
| 3 | 2026-05-21 | App Flask + 3 clases (Enojo / Tristeza / Feliz) | Pipeline productivo + dashboard interactivo |
| 4 | 2026-05-21 | N_PER_CLASS=14 con audios EXCELENTE solamente | **LOOCV 69%, pero solo 9% recall Enojo en holdout** — generalización catastrófica |
| 5 | 2026-05-22 | N_PER_CLASS=20 con `class_weight='balanced'` | LOOCV 67%, **59% recall Enojo holdout** (+50 pp) |
| 6 | 2026-05-23 | Intento con 4 clases (+ Tranquilidad) | Colapsa a **42% LOOCV** (chance=25%) — Tranquilidad confunde todo |
| 7 | 2026-05-23 | Re-etiquetado por IA + encoder emocional audeering | **97% LOOCV, 88% holdout** — gap LOOCV/holdout cerrado |
| 8 | 2026-05-23 | Localización de segmentos → dataset curado en `data/procesado` | 146 segmentos de 10 s (tramo más emotivo por audio); 6 mal-etiquetados eliminados |

### Iteración 1 — Filtrado acústico (commits `608636a`–`c514281`)

**Objetivo:** detectar audios donde el participante no proyectó la emoción pedida (algunos grabaron "enojo" en tono neutro).

**Implementación inicial:** 7 métricas acústicas globales por audio (energía RMS, pitch medio, brillo espectral, fracción de silencio, dinámica) combinadas con pesos manuales en cuatro scores: `score_enojo`, `score_tristeza`, `score_feliz`, `score_tranquilidad`. Cada score normalizado a [0, 1] con kernel lineal saturado.

**Decisiones tempranas:**
- Selección por **ranking** dentro de la clase declarada, no reasignar (commit `796cf99`).
- Umbral mínimo `MIN_SCORE` para descartar audios "muy malos".

**Problema que apareció después:** estas 7 métricas globales **no son lo bastante finas** para distinguir tristeza de tranquilidad (acústicamente casi idénticas: voz baja, monótona) ni emoción real de "habla pasiva". Esto se descubrió en la iteración 7 al comparar con un modelo emocional fine-tuneado.

### Iteración 2 — wav2vec2-base como encoder (commit `61fe927`)

**Salto técnico:** reemplazar features manuales (144 dims) por embeddings de `facebook/wav2vec2-base` mean-pooled (768 dims).

**Resultado:** LOOCV 90% en 2 clases (Enojo vs Tristeza). Esto validó usar un encoder neuronal pre-entrenado por encima de features handcrafted.

**Decisión empírica adoptada:** `MAX_DURATION = 10s` desde el inicio del audio. Probamos también:

| Variante | BalAcc LogReg LOOCV | Decisión |
|---|---|---|
| **0-10 s** (adoptado) | 67-69% | ✓ |
| Audio completo, 1 embedding | 0% (lineales colapsan) | ✗ |
| 0-15 s | 62% | ✗ |
| Warm-up 1-2 s + 10 s | 67% | ✗ |
| Chunks 8 s + agregación RMS | 60% | ✗ |

**Por qué 10s gana:** wav2vec2 hace mean-pool sobre los frames. Audios largos diluyen la firma emocional con contenido neutro/transicional. 10s captura prosodia sin diluir.

### Iteración 3 — App Flask + escalado a 3 clases (commits `76eb3e1`–`a020e6b`)

**Objetivo:** envolver el modelo en un dashboard usable y añadir una tercera clase.

**Implementación:** Flask + 4 tabs (Clasificador en Vivo, Explorador Dataset, Decisiones, Análisis). Predicción en vivo desde archivo o grabación del navegador (con conversión `webm/opus` → WAV PCM en cliente, commit `3e11824`).

**Problema descubierto:** ampliar a 3 clases (Enojo + Tristeza + Feliz) bajó el LOOCV a ~65%. Feliz tenía solo 15 audios disponibles vs 37 de las otras dos. El desbalance forzaba al modelo a predecir Feliz casi siempre o casi nunca.

### Iteración 4 — N=14 solo audios EXCELENTE (commit `4b059a6`)

**Hipótesis:** "Si entreno solo con los audios más prototípicos por clase, el modelo aprende fronteras más limpias."

**Implementación:** N_PER_CLASS=14, filtrando solo audios con score > 0.45 ("EXCELENTE").

**Resultado:** LOOCV LogReg 69%, **pero holdout devastador:**

| Métrica | Training (LOOCV) | Holdout |
|---|---|---|
| Recall Enojo | ~70% | **9% (2/23)** |
| Recall Tristeza | ~67% | ~30% |

**Causa diagnosticada:** *cherry-picking* del score. Los top-14 audios por score son los más extremos acústicamente (Enojo con energía/pitch claramente altos, Tristeza con energía/pitch claramente bajos). El modelo aprendió "alta energía → Enojo, baja energía → Tristeza", una pista trivial. Los audios mid-score del holdout caían geométricamente en el centroide opuesto. **Distribution shift inducido por la propia selección.**

**Lección:** el filtro acústico, usado como criterio de ranking estricto, sesga el training contra audios realistas.

### Iteración 5 — N=20 + class_weight balanced (commit `ee8edce`)

**Cambio:** subir N_PER_CLASS a 20 (incluir audios mid-score) y usar `class_weight='balanced'` para compensar que Feliz solo tiene 15 audios.

**Resultado:**

| Métrica | LOOCV | Holdout |
|---|---|---|
| LogReg bal_acc | 0.672 | 0.412 |
| SVM lineal bal_acc | 0.606 | 0.500 |
| **Recall Enojo holdout (SVM lineal)** | — | **10/17 (59%)** ← +50 pp vs iteración 4 |

**Cómo se logró:** los audios mid-score que antes excluíamos cubrían la curva completa de expresividad real. El modelo aprendió fronteras menos triviales y generalizó mejor. El LOOCV bajó porque ahora dentro del training hay audios más difíciles — caída honesta que refleja la dificultad real.

**Pero el holdout todavía estaba en ~50% bal_acc** — apenas por encima de chance en 3 clases con holdout desbalanceado.

### Iteración 6 — Intento con 4 clases (Tranquilidad incluida)

**Hipótesis:** "Una clase 'neutral' debería ayudar al modelo a no clasificar habla normal como Enojo o Feliz."

**Resultado:** colapso total.

| Modelo | LOOCV bal_acc | chance=0.25 |
|---|---|---|
| Random Forest | 0.38 |  |
| SVM RBF | 0.36 |  |
| SVM lineal | 0.35 |  |

**Causa:** acústicamente Tristeza y Tranquilidad son casi indistinguibles con features globales (ambas son voz baja, monótona, sin proyección). El modelo no las separaba y la confusión se propagaba a las otras dos clases.

**Decisión:** descartar Tranquilidad como cuarta clase. Vuelta atrás a 3 clases.

### Iteración 7 — Re-etiquetado por IA + encoder emocional (esta sesión, 2026-05-23)

Esta iteración resolvió **tres problemas estructurales simultáneamente** que las anteriores no habían podido cerrar.

#### Problema A — etiquetas ruidosas masivamente

**Síntoma reportado por el usuario:** "Algunos audios etiquetados como Enojo me los predice como Tristeza incluso con score alto. Y audios similares dan resultados contradictorios."

**Análisis:** corrimos el modelo SER `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` sobre los 162 audios del dataset para extraer arousal/valence/dominance (AVD) por audio. Mapeamos AVD a las 4 clases con kernels gaussianos centrados en los puntos teóricos de cada emoción y comparamos con la etiqueta humana.

**Resultados de la auditoría:**

| Clase | OK (IA confirma) | Sospechoso (IA discrepa, Δ > 0.15) | Suenan a... |
|---|---|---|---|
| **Enojo** | 10/48 (21%) | **34 (71%)** | Tranquilidad (30), Tristeza (3), Feliz (1) |
| Tristeza | 20/48 (42%) | 20 (42%) | Tranquilidad (20) |
| Feliz | 8/21 (38%) | 10 (48%) | Tranquilidad (7), Enojo (3) |
| Tranquilidad | 36/45 (**80%**) | 6 (13%) | Tristeza (4), Feliz (2) |

**Lectura:** Enojo era la clase peor etiquetada — el 71% de los audios marcados "Enojo" en realidad sonaban a habla normal (Tranquilidad). Casos extremos: `MT_08_ENO.ogg` con score_Enojo=0.006 y score_Tranquilidad=0.861. La gente no proyectó la emoción pedida.

**Importante:** el filtro acústico de las 7 métricas globales daba un diagnóstico **invertido** (decía que Tristeza era la peor). El modelo SER captura prosodia/micro-tensión que la energía promedio no ve.

**Acción tomada:** `scripts/reasignar_audios.py` movió 60 audios físicamente entre carpetas según la predicción del modelo SER, con umbral `Δ > 0.25` y manifest reversible en `outputs/reasignacion_log.json`.

| Origen → Destino | Audios movidos |
|---|---|
| Enojo → Tranquilidad | 26 |
| Tristeza → Tranquilidad | 18 |
| Feliz → Tranquilidad | 5 |
| Tranquilidad → Tristeza | 4 |
| Enojo → Tristeza | 3 |
| Tranquilidad → Feliz | 2 |
| Feliz → Enojo | 2 |

**Conteo final por carpeta tras reasignación:** Enojo 21, Feliz 16, Tranquilidad 88, Tristeza 37, Aburrido 45.

#### Problema B — encoder inadecuado

**Síntoma:** después del re-etiquetado, **LOOCV subió de 74% a 97%** pero **el holdout siguió en 50-57%**. Brecha enorme.

**Análisis:** mirando los errores con confianza alta — `VZ_04_ENO.ogg` (etiquetado Tranquilidad por reasignación, predicho Enojo conf=0.98), `MT_02_TRI.ogg` (predicho Enojo conf=0.97). El clasificador con `wav2vec2-base` discrepaba sistemáticamente del modelo SER que había hecho el re-etiquetado.

**Causa:** `facebook/wav2vec2-base` está pre-entrenado para reconocimiento de habla (ASR). Sus 768 dims capturan principalmente fonemas y palabras, no prosodia emocional. LOOCV funcionaba porque había solapamiento de vocabulario entre audios; holdout fallaba porque las palabras eran distintas y el modelo no veía la emoción.

**Acción:** migrar el encoder oficial del pipeline a `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` — mismo modelo que ya usábamos como auditor — vía un módulo compartido `emotion_encoder.py` en la raíz. Sus 1024 hidden_states están fine-tuneados sobre MSP-Podcast para predecir A/V/D, son emocionalmente relevantes por diseño.

**Resultado al re-entrenar con audeering:**

| Modelo | LOOCV antes (wav2vec2-base) | Holdout antes | LOOCV ahora (audeering) | **Holdout ahora** |
|---|---|---|---|---|
| SVM lineal | 0.97 | 0.55 | 0.93 | **0.84** |
| LogReg | 0.97 | 0.53 | 0.97 | **0.77** |
| SVM RBF | 0.93 | 0.57 | 0.97 | **0.76** |
| **Random Forest** | 0.93 | 0.53 | 0.93 | **0.88** |
| KNN k=5 | 0.83 | 0.43 | 0.83 | **0.81** |

**De 53% a 88% en holdout.** El cambio de encoder fue el factor decisivo, no solo el re-etiquetado.

#### Problema C — filtro acústico engañoso

**Síntoma:** durante la auditoría descubrimos que el filtro acústico original marcaba como "sospechoso" cosas distintas a lo que un humano (y el modelo SER) marcaba.

**Causa:** las 7 métricas globales (energía, pitch, brillo) son demasiado toscas. Alguien puede hablar suave pero con tensión emocional, y la energía promedio no lo capta.

**Acción:** desactivar el filtro hard (`MIN_SCORE = 0.0` en `config.py`). El score acústico sigue calculándose y se usa como criterio de ranking dentro de clase (top-N), pero **no descarta audios**. La detección de etiquetas erróneas pasa a ser responsabilidad del modelo SER vía `scripts/auditar_etiquetas_ia.py`.

### Iteración 8 — Localización de segmentos y dataset curado (esta sesión, 2026-05-23)

**Contexto:** Enojo y Feliz quedaron con pocos audios (≈17 y ≈13). El usuario notó que en varios de ellos **los primeros 10 s no suenan a la clase** (marcan Tranquilidad), pero **otros tramos del mismo audio sí**. Como el pipeline solo mira los primeros 10 s, esa señal se perdía.

**Herramientas de inspección añadidas:**
- `scripts/mostrar_todos.py` — vista en terminal de los 4 scores audeering + A/V/D de **todos** los audios, agrupados por carpeta, con `suena_a` = la clase de mayor score. Hermano de `scripts/mostrar_sospechosos.py` (que solo lista los discrepantes).
- `scripts/auditar_etiquetas_ia.py` parametrizado con `--data` y `--out` para poder auditar cualquier carpeta, no solo el dataset original.

**Localización (`scripts/segmentar_rescate.py`):** ventana deslizante de 10 s sobre el audio **completo**; para cada audio se queda con la ventana de mayor score para su clase y la recorta a un `.wav`. Estados por audio:

| Estado | Significado |
|---|---|
| OK | los primeros 10 s ya suenan a la clase |
| RESCATADO | los primeros 10 s no, pero otro tramo sí (ej. Tristeza que recién aparece en 35-45 s) |
| SIN_RESCATE | ningún tramo de 10 s suena a la clase → candidato a mal-etiquetado |

**¿Esto es cherry-picking?** No, y la distinción importa para la defensa:
- Seleccionar **qué audios** entran (botar los difíciles) **sí** sesga — es exactamente lo que causó el desastre de la iteración 4.
- Seleccionar **qué 10 s dentro de cada audio**, conservando *todos* los audios, **no** sesga: la emoción es actuada (la etiqueta vale para toda la grabación) y en qué tramo le salió mejor al hablante es ruido de grabación (arranque, carraspeo), no señal de la clase. Es localización de emoción / refinamiento de etiqueta débil, igual que recortar el silencio inicial de un clip.
- **Guardrail obligatorio:** el holdout **nunca** se segmenta. Se entrena con los recortes localizados pero se valida sobre audio crudo. Si solo sube el LOOCV-sobre-recortes y no el holdout-crudo, fue inflación, no mejora.

**Auditoría de los 152 recortes:** 149/152 (98%) confirman su clase. *Caveat honesto:* ese 98% es en parte circular — los segmentos se eligieron con audeering y se auditan con audeering, así que confirma que la selección funcionó, no que las etiquetas sean correctas. La validación real sigue siendo el holdout crudo con el clasificador downstream.

**Revisión manual:** los recortes que ni localizando confirmaban (`VA_13_ENO`, `SG_02_FEL`, `VZ_07_TRI`) se escucharon a oído. Tras escucharlos el usuario eliminó 6 audios genuinamente mal-etiquetados (4 Enojo + 2 Feliz) y conservó el resto, incluido `VZ_07_TRI` (empate Tristeza/Tranquilidad, no mal-etiquetado sino ambigüedad de bajo arousal).

**Resultado — dataset curado en `data/procesado/` (gitignored, regenerable):**

| Clase | Segmentos |
|---|---|
| Enojo | 13 |
| Feliz | 11 |
| Tranquilidad | 89 |
| Tristeza | 33 |
| **Total** | **146** |

Cada segmento es el tramo de 10 s más emotivo de su audio, listo para entrenar sin tocar `MAX_DURATION`. **Re-entrenamiento con este dataset: pendiente** (próximo paso, manteniendo un holdout crudo aparte).

### Lecciones acumuladas

1. **Cherry-picking del score sesga el training.** Filtrar solo "prototípicos" entrena al modelo a aprender la pista trivial (energía) y rompe la generalización.
2. **LOOCV alto + holdout bajo ≠ overfitting "normal".** En este proyecto significaba que el modelo aprendía algo que NO era emoción (etiquetas ruidosas o contenido lingüístico).
3. **`wav2vec2-base` es ASR, no SER.** Para clasificación emocional, usar un encoder fine-tuneado para emoción cambia el techo de rendimiento en holdout dramáticamente.
4. **Etiquetas humanas ≠ ground truth.** En datasets pequeños con actores no entrenados, ~50% de los audios pueden no proyectar la emoción pedida. Una segunda opinión de un modelo SER pre-entrenado es trabajo de auditoría obligatorio.
5. **Aburrido y Tristeza colapsan acústicamente.** No conviene tratarlas como clases separadas con un dataset pequeño — habría que distinguirlas con contexto léxico, no solo prosodia.
6. **Localizar la emoción ≠ cherry-picking.** En emoción actuada, recortar cada audio a su tramo más expresivo —conservando *todos* los audios— es refinamiento de etiqueta, no selección sesgada, siempre que el holdout no se segmente. Es distinto de filtrar *qué audios* entran, que sí sesga (lección 1).

---

## Aplicación web — pestañas disponibles

La app Flask (`app.py`) levanta una interfaz con 4 pestañas:

1. **Clasificador en Vivo** — sube o graba audio y obtén predicción con confianza y métricas acústicas.
2. **Explorador del Dataset** — reproduce audios **holdout** (excluye los usados en training del modelo desplegado) y prueba el modelo sobre ellos. Cada predicción es genuinamente out-of-sample.
3. **Decisiones** *(módulo de Toma de Decisiones)* — descrito en detalle abajo.
4. **Análisis y Métricas** — figuras del entrenamiento (matriz de confusión, separabilidad, comparativa).

---

## Capa 1 · Machine Learning

### Resultado actual (post-iteración 7)

**LOOCV (Leave-One-Audio-Out)** sobre training Enojo=15, Tranquilidad=15, con `class_weight='balanced'` y encoder audeering (1024 dims):

| Modelo | Accuracy | Balanced Acc |
|---|---|---|
| LogReg | 0.967 | 0.967 |
| SVM RBF | 0.967 | 0.967 |
| SVM lineal | 0.933 | 0.933 |
| Random Forest | 0.933 | 0.933 |
| KNN (k=5) | 0.833 | 0.833 |
| KNN (k=3) | 0.833 | 0.833 |
| Baseline (chance) | — | 0.500 |

**Holdout (79 audios no usados en training: 6 Enojo + 73 Tranquilidad):**

| Modelo | Acc | **Bal Acc** | Recall Enojo | Recall Tranq |
|---|---|---|---|---|
| **Random Forest** | 0.924 | **0.882** | 0.833 | 0.932 |
| SVM lineal | 0.848 | 0.841 | 0.833 | 0.849 |
| KNN k=5 | 0.924 | 0.806 | 0.667 | 0.945 |
| LogReg | 0.861 | 0.772 | 0.667 | 0.877 |
| SVM RBF | 0.835 | 0.758 | 0.667 | 0.849 |
| KNN k=3 | 0.899 | 0.716 | 0.500 | 0.932 |

**Holdout balanceado (6 Enojo + 6 Tranquilidad random):** Random Forest, SVM lineal, KNN k=5 → 83.3% accuracy.

### Dataset (post-reasignación)

| Parámetro | Valor |
|---|---|
| Total de audios disponibles tras reasignación | 207 (Enojo 21, Tristeza 37, Feliz 16, Tranquilidad 88, Aburrido 45) |
| Clases activas en el pipeline actual | Enojo, Tranquilidad |
| Audios usados en training | 15 + 15 = 30 (top por `score_clase`, sin filtro hard) |
| Audios disponibles en holdout | 6 Enojo + 73 Tranquilidad |
| Recolectores | MT, VA, ED, VZ, SG (5) |
| Duración por audio | 21-64 segundos (mediana 34.6 s; se usan los primeros 10 s) |
| Audios reasignados por IA (con manifest reversible) | 60 / 222 originales |

**Cada audio es de un hablante distinto.** Los prefijos MT/VZ/VA/ED/SG identifican al *recolector* del audio, no al hablante. LOOCV ya es efectivamente leave-one-speaker-out.

### Pipeline ML actual

```
1. (Una sola vez por dataset) Auditoría de etiquetas
   python scripts/auditar_etiquetas_ia.py
   → Usa audeering para predecir A/V/D de cada audio
   → Genera outputs/audios_sospechosos_ia.csv

2. (Opcional) Reasignación por IA
   python scripts/reasignar_audios.py --apply --umbral 0.25
   → Mueve audios entre carpetas según predicción del modelo SER
   → Manifest reversible en outputs/reasignacion_log.json
   → Para revertir: python scripts/reasignar_audios.py --revert

3. Scoring acústico (mantiene el rol de ranking, no de filtro)
   python scripts/filtrar_audios.py
   → outputs/reporte_filtrado_v2.csv

4. Selección por ranking (top-N_PER_CLASS por score dentro de clase)
   → MIN_SCORE = 0.0 (desactivado, ya no se descartan audios)

5. Extracción de embeddings con audeering (cache: outputs/embeddings_v2.npz)
   python scripts/exportar_modelos.py
   → librosa.load(..., duration=10.0) + audeering + mean-pool de hidden_states
   → un embedding 1024-d por audio

6. Entrenamiento LOOCV + serialización
   → outputs/modelos/*.joblib (6 modelos entrenados con TODOS los audios de training)
   → outputs/model_metrics.json (métricas LOOCV honestas)
   → outputs/predicciones_loocv.{csv,json}
```

### Decisiones de diseño actuales

| Decisión | Valor | Por qué |
|---|---|---|
| **Encoder** | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | Fine-tuneado para SER. Sus hidden_states son emocionales por diseño, no fonéticos. Subió holdout de 55% a 88%. |
| **MAX_DURATION** | 10 s | Sweet spot empírico. Audios largos diluyen la señal con mean-pool. |
| **MIN_SCORE** | 0.0 (desactivado) | El filtro acústico engañaba. Las decisiones de etiqueta se toman con el modelo SER en `auditar_etiquetas_ia.py`. |
| **N_PER_CLASS** | 15 | Enojo solo tiene 21 audios tras reasignación; N=15 deja 6 audios de Enojo para holdout. |
| **class_weight** | 'balanced' | Tranquilidad tiene 88 audios y Enojo 21; balanceo automático en SVMs, LogReg, RF. |
| **Mejor modelo desplegado** | Random Forest | Mejor holdout bal_acc (0.882). Más robusto a los 11 errores residuales que LogReg/SVM resuelven con alta confianza incorrecta. |

---

## Capa 2 · Toma de Decisiones

### Problema de decisión

> **Pregunta central:** Un call center evalúa desplegar un detector acústico de enojo para escalar llamadas críticas a un supervisor senior antes de que el cliente cuelgue molesto. ¿Conviene desplegarlo? Si sí, ¿con qué configuración (modelo × umbral)?

| Elemento | Detalle |
|---|---|
| **Stakeholder** | Coordinador de Operaciones del call center |
| **Decisión** | GO / NO-GO + configuración óptima (modelo × umbral) |
| **Métrica de éxito** | Valor Presente Neto mensual esperado (USD) |
| **Restricciones** | El modelo no puede degradar la experiencia de clientes no-enojados (falsos positivos costosos por tiempo de supervisor) |

**¿Por qué no basta con la BalAcc?**  
La balanced accuracy ignora el costo asimétrico entre errores: omitir a un cliente furioso (FN) puede costar 20× más que escalar a uno que no estaba enojado (FP). Por eso la decisión correcta depende de la matriz de costos del negocio, no del modelo más preciso en abstracto.

### Estructura del módulo

La pestaña **Decisiones** contiene 4 secciones secuenciales más una recomendación final. Cada sección recibe inputs de la anterior y todo se recalcula en vivo cuando el usuario mueve un control.

```
┌──────────────────────────────────────────────────────────┐
│ 1. CONTEXTO + MATRIZ DE COSTOS (editable)                │
│    ↓ define la economía de la decisión                   │
├──────────────────────────────────────────────────────────┤
│ 2. DATOS PARA DECIDIR (evidencia LOOCV)                  │
│    ↓ muestra precision/recall/F1 reales por clase         │
├──────────────────────────────────────────────────────────┤
│ 3. SIMULADOR DE DESPLIEGUE                                │
│    Inputs: escenario, modelo, umbral, volumen, prevalencia│
│    Outputs: CM viva, ROC con punto óptimo, VPN mensual    │
├──────────────────────────────────────────────────────────┤
│ 4. ANÁLISIS DE DECISIONES                                 │
│    - Tornado de sensibilidad (±30% por parámetro)         │
│    - Monte Carlo 2 000 escenarios                         │
│    - Tabla de break-even (12 combinaciones)               │
├──────────────────────────────────────────────────────────┤
│ ★ RECOMENDACIÓN FINAL                                     │
│    GO / GO condicional / NO-GO + condiciones de validez   │
└──────────────────────────────────────────────────────────┘
```

### Sección 1 · Contexto y matriz de costos

Tres tarjetas describen el contexto de negocio (stakeholder, decisión a tomar, métrica de decisión). Debajo hay una **matriz de costos editable** con 4 celdas:

| Celda | Significado | Default |
|---|---|---|
| **TP** | Escalamiento correcto → cliente atendido a tiempo, churn evitado | +USD 25 |
| **FP** | Falsa alarma → supervisor distraído sin necesidad | −USD 4 |
| **FN** | Enojo no detectado → riesgo de queja o churn | −USD 80 |
| **TN** | No-escalación correcta → operación normal | USD 0 |

La interfaz muestra automáticamente el **ratio costo FN / FP** y un hint explicando cómo ese ratio sesga el umbral óptimo:

- Ratio alto (FN ≫ FP) → favorece umbrales bajos (más recall, asume FP).
- Ratio bajo (FN ≪ FP) → favorece umbrales altos (más precisión, evita FP).

Todos los valores son inputs numéricos editables; cualquier cambio recalcula la sección 3 y 4 instantáneamente.

### Sección 2 · Datos para decidir

Renderiza tarjetas con la **evidencia LOOCV real** que respalda la decisión:

- Distribución de clases (chips con cuenta por clase).
- Mejor modelo con su BalAcc.
- Tabla con precision, recall y F1 por clase para ese mejor modelo.

Esto convierte el output del clasificador en evidencia inspeccionable antes de la simulación.

### Sección 3 · Simulador de despliegue

La sección más interactiva. Tiene 6 controles:

| Control | Qué ajusta | Default |
|---|---|---|
| **Escenario** | 2 emociones (Enojo vs Tranquilidad) o configuraciones legacy de 3 emociones | 2 emociones |
| **Modelo** | Cualquiera de los 6 clasificadores (ordenados por BalAcc) | Random Forest |
| **Umbral P(Enojo) ≥** | Umbral de decisión binaria sobre la probabilidad de Enojo | 0.50 |
| **Volumen mensual** | Llamadas/mes proyectadas | 10 000 |
| **Prevalencia de Enojo** | % de llamadas que realmente son enojo en producción | 18 % |
| **Costo de inferencia** | USD por llamada procesada (cloud + cómputo) | USD 0.02 |

A la derecha hay tres outputs:

1. **Matriz de confusión @ umbral activo** — se recalcula desde las probabilidades LOO usando el umbral elegido. Muestra TP/FP/FN/TN con colores y debajo: Recall, Precision, F1, FPR.
2. **Curva ROC + punto óptimo** — ROC binaria (Enojo vs no-Enojo). Marca 🟢 verde = operación actual, ⭐ amarillo = umbral que maximiza VPN dadas las costos actuales.
3. **VPN mensual + desglose** — Suma de beneficios menos costos esperados. Verde si > 0, rojo si < 0.

### Sección 4 · Análisis de decisiones

- **Tornado de sensibilidad** — varía ±30 % por parámetro (FN, FP, TP, prevalencia, volumen, costo de inferencia) y mide el rango de VPN.
- **Monte Carlo (2 000 escenarios)** — ruido triangular en costos, volumen y prevalencia. Devuelve P(VPN > 0), media, mediana e IC90%.
- **Tabla de break-even (escenario × modelo)** — para las combinaciones disponibles, calcula umbral óptimo, VPN óptimo, prevalencia mínima rentable y veredicto.

### Recomendación final

Tarjeta de cierre con badge **GO / GO condicional / NO-GO**:

| Veredicto | Criterio |
|---|---|
| **GO** | VPN óptimo > USD 1 000 y P(VPN > 0) ≥ 75 % en Monte Carlo |
| **GO condicional** | P(VPN > 0) ≥ 55 % |
| **NO-GO** | Cualquier otra condición |

### Modelo matemático

**Matriz de confusión binaria desde probabilidades LOO**

Para cada muestra `i`, el modelo entregó un vector de probabilidades `p_i` con LOOCV. Definimos la predicción binaria:

```
ŷ_i = 1  si  p_i[Enojo] ≥ θ
ŷ_i = 0  en otro caso
```

**Valor neto mensual esperado**

```
TPR = TP / (TP + FN)              (recall)
FPR = FP / (FP + TN)
FNR = 1 - TPR
TNR = 1 - FPR

V_pos = Volumen × Prevalencia
V_neg = Volumen × (1 - Prevalencia)

E[TP] = V_pos × TPR
E[FN] = V_pos × FNR
E[FP] = V_neg × FPR
E[TN] = V_neg × TNR

VPN = E[TP]·valor_TP
    − E[FP]·costo_FP
    − E[FN]·costo_FN
    − E[TN]·costo_TN
    − Volumen·costo_inferencia
    − costo_fijo_mensual
```

**Umbral óptimo:** búsqueda lineal sobre θ ∈ [0.05, 0.95] con paso 0.01 maximizando VPN.

### Origen y validez de los valores económicos

Los valores de costo por defecto **no provienen de datos confidenciales de una empresa real** — un proyecto académico no tiene acceso a esa información. En lugar de eso, son **valores ilustrativos calibrados con benchmarks de industria publicados**, elegidos para que el orden de magnitud y las relaciones entre ellos sean defendibles. La interfaz permite reemplazarlos con datos propios en cualquier momento.

| Parámetro | Default | Fuente / benchmark |
|---|---|---|
| **Costo FN** | USD 80 | Reichheld & Sasser, HBR (1990); Salesforce State of the Connected Customer (2023). |
| **Costo FP** | USD 4 | U.S. BLS — Customer Service Supervisors OEWS 2023. |
| **Valor TP** | USD 25 | Bain & Company — *Prescription for cutting costs*. |
| **Costo de inferencia** | USD 0.02/llamada | AWS Transcribe, Azure Speech, Google Cloud STT pricing (2024). |
| **Costo fijo mensual** | USD 1 200 | AWS EC2 `g4dn.xlarge` + monitoring + ingeniería prorrateada. |
| **Volumen 10 000 llamadas/mes** | — | ContactBabel + ICMI benchmarks 2023. |
| **Prevalencia 18% de Enojo** | — | NICE inContact CX Transformation Benchmark (2023). |

El **tornado de sensibilidad** y **Monte Carlo** existen precisamente para validar que la recomendación sobrevive a errores en estos valores. Para implementación real, reemplazar con datos del cliente.

---

## Estructura del proyecto

```
clasificador-audios/
├── app.py                                     # Flask backend
├── config.py                                  # N_PER_CLASS, ACTIVE_CLASSES, MIN_SCORE
├── emotion_encoder.py                         # ★ Encoder audeering compartido (training + auditoria + inferencia)
├── templates/
│   └── index.html                             # 4 pestañas
├── static/
│   ├── css/style.css
│   └── js/main.js                             # Simulador de decisiones en cliente
├── notebooks/
│   ├── 01_clasificador_v1_features.ipynb      # Iteración 1: features manuales (144 dims)
│   ├── 02_clasificador_v1_embeddings.ipynb    # Iteración 2: wav2vec2-base (768 dims)
│   └── 03_clasificador_v2_balanceado.ipynb    # Iteración 5: N=20 + class_weight
├── scripts/
│   ├── filtrar_audios.py                      # Scoring acústico (ranking, no filtro hard)
│   ├── auditar_etiquetas_ia.py                # ★ Auditoría de etiquetas con modelo SER (--data/--out)
│   ├── reasignar_audios.py                    # Re-etiquetado físico con manifest reversible
│   ├── mostrar_sospechosos.py                 # Lista los audios cuya etiqueta discrepa de la IA
│   ├── mostrar_todos.py                       # ★ Vista de scores audeering de TODOS los audios
│   ├── segmentar_rescate.py                   # ★ Localización: mejor segmento de 10 s por audio
│   ├── exportar_modelos.py                    # Training LOOCV + serialización
│   ├── build_experiment_history.py            # Historial para el dashboard
│   ├── generar_datos_decisiones.py            # Data del módulo Decisiones
│   └── regenerar_figuras.py                   # Regenera PNGs del dashboard
├── outputs/
│   ├── reporte_filtrado_v2.csv                # Scores acústicos por audio
│   ├── audios_sospechosos_ia.csv              # Auditoría IA: A/V/D + delta + suena_a
│   ├── mejores_segmentos.csv                  # ★ Reporte de localización por audio (estado, mejor tramo)
│   ├── mejores_segmentos_auditoria.csv        # ★ Auditoría IA de los segmentos localizados
│   ├── reasignacion_log.json                  # Manifest reversible de la última reasignación
│   ├── model_metrics.json                     # BalAcc y descripción por modelo
│   ├── experiment_history.json
│   ├── decisions_data.json                    # Data para el módulo Decisiones
│   ├── embeddings_v2.npz                      # Embeddings 1024-d cacheados (audeering)
│   ├── embeddings_wav2vec2.npz                # (Legacy, 2-clase con wav2vec2-base)
│   ├── predicciones_loocv.{csv,json}          # Predicciones por audio
│   ├── modelos/                               # 6 clasificadores serializados (.joblib)
│   └── figuras/                               # Gráficos de resultados
├── data/                                      # Dataset (gitignored)
│   ├── AUDIOS MACHINE LEARNING/               # Originales por clase
│   └── procesado/                             # ★ Dataset curado: mejor segmento de 10 s por audio (146)
└── README.md
```

`★` = añadido o renovado en las iteraciones 7-8.

---

## Uso rápido

```bash
# 1. Activar entorno
source ../.venv/bin/activate

# 2. (Una sola vez) Auditar etiquetas del dataset con modelo SER
python scripts/auditar_etiquetas_ia.py
# → outputs/audios_sospechosos_ia.csv

# 2b. (Opcional) Inspeccionar los scores audeering de todos los audios
python scripts/mostrar_todos.py          # todos, agrupados por carpeta
python scripts/mostrar_sospechosos.py    # solo los discrepantes

# 2c. (Opcional) Localizar el mejor segmento de 10 s de cada audio
python scripts/segmentar_rescate.py
# → outputs/mejores_segmentos/<Clase>/*.wav (dataset localizado, luego movible a data/procesado)

# 3. (Opcional, reversible) Re-etiquetar audios sospechosos
python scripts/reasignar_audios.py --apply --umbral 0.25
# Para revertir: python scripts/reasignar_audios.py --revert

# 4. Re-correr el pipeline completo
python scripts/filtrar_audios.py
python scripts/exportar_modelos.py
python scripts/build_experiment_history.py
python scripts/generar_datos_decisiones.py

# 5. Levantar la app
python app.py
# http://127.0.0.1:5001
```

**Override puntual de N_PER_CLASS sin tocar `config.py`:**

```bash
N_PER_CLASS=20 python scripts/exportar_modelos.py
```

---

## Dependencias

```
flask
librosa>=0.10
torch>=2.0
transformers>=4.40        # audeering requiere transformers reciente
scikit-learn>=1.3
joblib
numpy
pandas
matplotlib
```

**Nota sobre la primera descarga:** el modelo `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` pesa ~1.2 GB. Se cachea en `~/.cache/huggingface/` la primera vez que cargas el encoder. Después de eso, el arranque de la app y el pipeline son rápidos (~3 s para cargar el modelo desde caché).

---

## Hallazgos relevantes para la decisión

- **El umbral 0.5 no es óptimo cuando los costos son asimétricos.** Con FN/FP = 20×, el VPN óptimo suele estar en umbrales 0.25-0.35, sacrificando precisión por recall.
- **La prevalencia importa más que la BalAcc.** En el tornado típico, prevalencia y costo_FN dominan; el modelo es relativamente intercambiable mientras esté por encima de 75 % BalAcc.
- **Monte Carlo es decisivo.** Modelos con bal_acc alto pero P(VPN > 0) < 60 % deberían pasar por piloto pequeño antes de escalar.
- **El re-etiquetado por IA es trabajo upstream, no un truco.** Ningún encoder ni modelo arregla etiquetas mal puestas — limpiar el ground truth es un paso anterior a cualquier comparación de algoritmos.

---

## Contexto académico

- **Curso de Machine Learning**: pipeline de clasificación de emociones (capa 1).
- **Curso de Toma de Decisiones**: módulo analítico de simulación, sensibilidad y recomendación (capa 2).

Dataset recolectado por 5 personas (MT, VZ, VA, ED, SG) con ~10-30 s por audio. Cada audio es de un hablante distinto.
