# EmotiSpeech · Clasificador de Emociones + Módulo de Toma de Decisiones

Proyecto integrado de dos capas:

1. **Capa de Machine Learning** — clasificación de emociones a partir de audio de voz (wav2vec2 + clasificadores clásicos).
2. **Capa de Toma de Decisiones** — un módulo analítico construido sobre los resultados del clasificador que permite simular, evaluar y recomendar despliegues con justificación cuantitativa de negocio.

La capa de Toma de Decisiones se agregó **sin modificar la capa de ML**: solo consume los embeddings y los modelos ya entrenados.

---

## Tabla de contenido

- [Resumen rápido](#resumen-rápido)
- [Aplicación web — pestañas disponibles](#aplicación-web--pestañas-disponibles)
- [Capa 1 · Machine Learning](#capa-1--machine-learning)
  - [Decisiones de diseño verificadas empíricamente](#decisiones-de-diseño-verificadas-empíricamente)
  - [Caveat sobre generalización a hablantes nuevos](#caveat-sobre-generalización-a-hablantes-nuevos)
- [Capa 2 · Toma de Decisiones](#capa-2--toma-de-decisiones)
  - [Problema de decisión](#problema-de-decisión)
  - [Estructura del módulo](#estructura-del-módulo)
  - [Sección 1 · Contexto y matriz de costos](#sección-1--contexto-y-matriz-de-costos)
  - [Sección 2 · Datos para decidir](#sección-2--datos-para-decidir)
  - [Sección 3 · Simulador de despliegue](#sección-3--simulador-de-despliegue)
  - [Sección 4 · Análisis de decisiones](#sección-4--análisis-de-decisiones)
  - [Recomendación final](#recomendación-final)
  - [Modelo matemático](#modelo-matemático)
  - [Origen y validez de los valores económicos](#origen-y-validez-de-los-valores-económicos)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Uso rápido](#uso-rápido)
- [Dependencias](#dependencias)

---

## Resumen rápido

| Capa | Qué hace | Salida |
|---|---|---|
| ML | Clasifica audio en Enojo / Tristeza / Feliz con 6 modelos clásicos sobre embeddings wav2vec2 (768 dims) | Predicción + probabilidades + métricas acústicas |
| Decisiones | Convierte las matrices de confusión y probabilidades LOO en una decisión de negocio (GO / NO-GO, umbral óptimo, VPN esperado) | Recomendación justificada con sensibilidad y Monte Carlo |

**Mejor modelo actual:** Regresión Logística con **67.2 %** de balanced accuracy LOOCV en el escenario de 3 emociones, training cap a 20 por clase con `class_weight='balanced'` (Enojo 20 + Tristeza 20 + Feliz 15; Feliz limitado por el dataset). Pipeline: wav2vec2-base sobre los primeros 10 segundos de cada audio (sweet spot empírico). El "Explorador del Dataset" muestra **solo audios holdout** (no usados en training) para permitir validación honesta out-of-sample desde la app. En holdout, **SVM lineal alcanza ~59% recall en Enojo** vs solo 9% con N=14 (ver [Decisiones de diseño](#decisiones-de-diseño-verificadas-empíricamente)).

---

## Aplicación web — pestañas disponibles

La app Flask (`app.py`) levanta una interfaz con 4 pestañas:

1. **Clasificador en Vivo** — sube o graba audio y obtén predicción con confianza y métricas acústicas.
2. **Explorador del Dataset** — reproduce audios **holdout** (excluye los usados en training del modelo desplegado) y prueba el modelo sobre ellos. Cada predicción es genuinamente out-of-sample.
3. **Decisiones** *(módulo de Toma de Decisiones)* — descrito en detalle abajo.
4. **Análisis y Métricas** — figuras del entrenamiento (matriz de confusión, separabilidad, comparativa).

---

## Capa 1 · Machine Learning

### Resultado principal

**LOOCV (Leave-One-Audio-Out)** sobre training Enojo=20, Tristeza=20, Feliz=15, con `class_weight='balanced'`:

| Modelo | Accuracy | Balanced Acc |
|---|---|---|
| **Regresión Logística** | **0.691** | **0.672** |
| SVM lineal | 0.618 | 0.606 |
| SVM RBF | 0.618 | 0.589 |
| Random Forest | 0.564 | 0.533 |
| KNN (k=5) | 0.491 | 0.478 |
| KNN (k=3) | 0.491 | 0.478 |
| Baseline (chance) | — | 0.333 |

**Holdout** (audios no usados en training, 17 Enojo + 17 Tristeza + 0 Feliz):

| Modelo | Recall Enojo | Recall Tristeza | BalAcc holdout |
|---|---|---|---|
| **SVM lineal** | **10/17 (59%)** | 7/17 (41%) | **50.0%** |
| LogReg | 8/17 (47%) | 6/17 (35%) | 41.2% |
| SVM RBF | 5/17 (29%) | 10/17 (59%) | 44.1% |
| KNN k=5 | 9/17 (53%) | 4/17 (24%) | 38.2% |

Antes (N=14 sin class_weight), SVM lineal acertaba solo 2/23 = 9% Enojo en holdout. El salto a 59% con N=20 confirma que **el problema previo era el rango de expresividad del training**, no el modelo.

### Dataset

| Parámetro | Valor |
|---|---|
| Total de audios disponibles | 89 (Enojo 37, Tristeza 37, Feliz **15**) + 36 Tranquilidad (no activos) |
| Clases activas | Enojo, Tristeza, Feliz |
| Audios usados en training | 20+20+15 = 55 (top por `score_clase`, Feliz cap a 15) |
| Audios disponibles en holdout (dashboard) | 17 Enojo + 17 Tristeza + 0 Feliz |
| Recolectores | MT, VZ, VA, ED |
| Duración por audio | 21-64 segundos (mediana 34.6s; se usan los primeros 10s) |

**Limitación reconocida del dataset:**
- **Feliz solo tiene 15 grabaciones** vs 37 de Enojo/Tristeza. Se usan todas para training; no quedan holdout Feliz. La métrica holdout no incluye Feliz por esta razón.
- **Cada audio es de un hablante distinto** (los prefijos MT/VZ/VA/ED identifican al *recolector* del audio, no al hablante). Esto significa que LOOCV ya es efectivamente leave-one-speaker-out: cada audio held-out es una persona que el modelo nunca escuchó. El número reportado refleja generalización razonable a nuevos hablantes.
- **Solo 4 recolectores = 4 condiciones de grabación** (mic, ambiente). El modelo podría haber aprendido huellas técnicas del recolector. Nuevos recolectores en futuras grabaciones probarían robustez a hardware/ambiente distintos.

### Pipeline ML

```
1. Scoring acústico (scripts/filtrar_audios.py)
   → Calcula score_enojo, score_tristeza, score_feliz sobre los primeros 10s
   → Genera outputs/reporte_filtrado_v2.csv

2. Selección por ranking (top-N_PER_CLASS por clase, ordenado por su score)

3. Extracción de embeddings con wav2vec2 (cache: outputs/embeddings_v2.npz)
   → librosa.load(..., duration=10.0) + wav2vec2-base + mean-pool
   → un embedding 768-d por audio

4. Entrenamiento LOOCV + serialización (scripts/exportar_modelos.py)
   → outputs/modelos/*.joblib (6 modelos entrenados con TODOS los 42 audios)
   → outputs/model_metrics.json (métricas LOOCV honestas, no del modelo final)
   → outputs/predicciones_loocv.{csv,json} (qué audios falló cada modelo)
```

### Decisiones de diseño verificadas empíricamente

**Ventana de audio (MAX_DURATION):**

| Variante probada | BalAcc LogReg LOOCV | Decisión |
|---|---|---|
| **0-10s** (actual) | 67-69% | ✓ Adoptado |
| Audio completo, 1 embedding | 0% (lineales colapsan) | Rechazado |
| 0-15s | 62% | Rechazado |
| Warm-up 1-2s + 10s | 67% | Rechazado |
| Chunks 8s + agregación RMS | 60% | Rechazado |

**Por qué 10s gana:** wav2vec2 hace mean-pool internamente sobre los frames del input. Audios largos diluyen la firma emocional al promediar contenido neutro/transicional. 10s es suficiente para capturar la prosodia emocional sin caer en dilución. Chunks introducen ruido de etiqueta y rompen modelos basados en distancia (KNN cae a 38%).

**Tamaño y composición del training (N_PER_CLASS):**

| Configuración | LOOCV LogReg | Holdout Enojo (SVM lineal) | Decisión |
|---|---|---|---|
| N=14 (solo EXCELENTE) | 69% | 9% (2/23) | Rechazado |
| **N=20 + class_weight balanced** | 67% | **59% (10/17)** | ✓ Adoptado |

**Por qué N=20 gana:** N=14 seleccionaba solo audios con score muy alto, creando un training set homogéneo en expresividad. El modelo aprendía "Enojo = audio muy gritado" y fallaba en holdout porque los audios mid-score caían geométricamente cerca del centroide de Tristeza en el espacio de embeddings wav2vec2. Expandir a N=20 incluye audios mid-score que cubren la curva completa de expresividad → modelo generaliza dramáticamente mejor (+50pp en recall holdout). El LOOCV baja levemente porque ahora hay audios más difíciles dentro del training, pero esa caída es honesta — refleja la dificultad real del problema. Feliz queda en 15 (todos los disponibles) y se compensa la asimetría con `class_weight='balanced'`.

### Caveat sobre generalización: hablantes vs condiciones de grabación

Cada audio del dataset es de un hablante **distinto** — el prefijo MT/VZ/VA/ED identifica quién recolectó la grabación, no quién habla. Esto hace que el LOOCV ya sea efectivamente leave-one-speaker-out: el 67% reportado es generalización a personas que el modelo nunca escuchó.

Lo que sí persiste es **condición-de-grabación overlap**: solo hay 4 recolectores, así que solo 4 configuraciones de micrófono/ambiente. El modelo podría haber capturado señal técnica del recolector. La forma rigurosa de medirlo sería leave-one-*recolector*-out (dejar fuera todos los audios de MT, etc.): probablemente daría algo menor que 67% pero no se puede estimar a priori. Una mejora futura sería incorporar audios de más recolectores.

---

## Capa 2 · Toma de Decisiones

### Problema de decisión

> **Pregunta central:** Un call center evalúa desplegar un detector acústico de enojo para escalar llamadas críticas a un supervisor senior antes de que el cliente cuelgue molesto. ¿Conviene desplegarlo? Si sí, ¿con qué escenario (2 ó 3 emociones), qué modelo y qué umbral operativo?

| Elemento | Detalle |
|---|---|
| **Stakeholder** | Coordinador de Operaciones del call center |
| **Decisión** | GO / NO-GO + configuración óptima (escenario × modelo × umbral) |
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

Renderiza dos tarjetas (una por escenario) con la **evidencia LOOCV real** que respalda la decisión:

- Distribución de clases (chips con cuenta por clase).
- Mejor modelo del escenario con su BalAcc.
- Tabla con precision, recall y F1 por clase para ese mejor modelo.

Esto convierte el output del clasificador en evidencia inspeccionable antes de la simulación. Por ejemplo, en 3 emociones uno ve que Tristeza tiene recall=0.70 pero F1 menor por menos precision (algunas confusiones con Enojo).

### Sección 3 · Simulador de despliegue

La sección más interactiva. Tiene 6 controles:

| Control | Qué ajusta | Default |
|---|---|---|
| **Escenario** | 2 emociones (Enojo vs Tristeza) o 3 emociones (Enojo, Tristeza, Feliz) | 3 emociones |
| **Modelo** | Cualquiera de los 6 clasificadores (ordenados por BalAcc) | SVM lineal |
| **Umbral P(Enojo) ≥** | Umbral de decisión binaria sobre la probabilidad de Enojo | 0.50 |
| **Volumen mensual** | Llamadas/mes proyectadas | 10 000 |
| **Prevalencia de Enojo** | % de llamadas que realmente son enojo en producción | 18 % |
| **Costo de inferencia** | USD por llamada procesada (cloud + cómputo) | USD 0.02 |

A la derecha hay tres outputs:

1. **Matriz de confusión @ umbral activo**  
   Se recalcula desde las probabilidades LOO de las 20-30 muestras usando el umbral elegido. Muestra TP/FP/FN/TN con colores (verde, ámbar, rojo, gris) y debajo: Recall, Precision, F1, FPR.

2. **Curva ROC + punto óptimo**  
   ROC binaria (Enojo vs no-Enojo) construida desde las probabilidades LOO. Marca dos puntos: 🟢 verde = operación actual, ⭐ amarillo = umbral que maximiza VPN dadas las costos actuales.

3. **VPN mensual + desglose**  
   Suma de beneficios menos costos esperados. Verde si > 0, rojo si < 0. El desglose línea por línea muestra cuánto aporta cada componente (+beneficio TP, −costo FN, −costo FP, −costo inferencia, −costos fijos).

### Sección 4 · Análisis de decisiones

Aquí no se ajusta más el modelo; se prueba si la decisión es **robusta** ante cambios en los inputs.

- **Tornado de sensibilidad**  
  Para cada parámetro (FN, FP, TP, prevalencia, volumen, costo de inferencia), varía ±30 % manteniendo los demás fijos y mide el rango de VPN resultante. Las barras más largas identifican los parámetros más críticos. Si la decisión cambia de GO a NO-GO en alguna barra, ese parámetro merece atención antes de desplegar.

- **Monte Carlo (2 000 escenarios)**  
  Cada simulación samplea ruido triangular en costos, volumen y prevalencia y calcula el VPN resultante. Devuelve:
  - **P(VPN > 0)**: probabilidad de que la decisión sea rentable bajo incertidumbre.
  - Media, mediana e intervalo de confianza del 90 %.
  - Histograma con la línea VPN=0 marcada en amarillo.

- **Tabla de break-even (escenario × modelo)**  
  Para las 12 combinaciones (2 escenarios × 6 modelos) calcula:
  - Umbral óptimo en VPN (búsqueda sobre 0.05 → 0.95).
  - VPN óptimo al mes.
  - **Prevalencia mínima de Enojo** que rentabiliza el modelo (break-even).
  - Veredicto: GO fuerte / GO / Marginal / NO-GO.
  
  La fila ganadora se marca con una estrella.

### Recomendación final

Tarjeta de cierre con un badge **GO / GO condicional / NO-GO** decidido así:

| Veredicto | Criterio |
|---|---|
| **GO** | VPN óptimo > USD 1 000 y P(VPN > 0) ≥ 75 % en Monte Carlo |
| **GO condicional** | P(VPN > 0) ≥ 55 % |
| **NO-GO** | Cualquier otra condición |

Acompañado de cinco bloques de justificación:

- **Por qué gana** — diferencia de VPN vs runner-up, relación con la asimetría de costos.
- **Riesgos / Trade-off** — qué se pierde con la configuración elegida.
- **Condiciones de la decisión** — bajo qué rangos de prevalencia / costos se mantiene la recomendación.
- **Siguiente acción** — piloto sugerido (volumen, duración, métricas de cierre).

### Modelo matemático

**Matriz de confusión binaria desde probabilidades LOO**

Para cada muestra `i`, el modelo entregó un vector de probabilidades `p_i` con LOOCV. Definimos la predicción binaria:

```
ŷ_i = 1  si  p_i[Enojo] ≥ θ
ŷ_i = 0  en otro caso
```

A partir de los 20 (2-clases) o 30 (3-clases) pares (y_i, ŷ_i) se obtiene TP, FP, FN, TN.

**Valor neto mensual esperado**

```
TPR = TP / (TP + FN)              (recall)
FPR = FP / (FP + TN)
FNR = 1 - TPR
TNR = 1 - FPR

V_pos = Volumen × Prevalencia          # llamadas realmente enojadas/mes
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

**Umbral óptimo**

Búsqueda lineal sobre θ ∈ [0.05, 0.95] con paso 0.01 maximizando VPN.

**Break-even de prevalencia**

Despejando `p` en `VPN(p) = 0` con TPR, FPR fijos:

```
VPN(p) = V·[ p·(TPR·v_TP − FNR·c_FN) + (1−p)·(−FPR·c_FP − TNR·c_TN) ] − V·c_inf − c_fix

→ p* = [ V·(c_inf − b) + c_fix ] / [ (a − b)·V ]
  donde a = TPR·v_TP − FNR·c_FN
        b = −FPR·c_FP − TNR·c_TN
```

**Monte Carlo**

Distribución triangular (low, mode, high) sobre cada input financiero. Mode = valor actual, low = mode×0.7, high = mode×1.3-1.6 según el parámetro. 2 000 muestras → cuantiles p5/p50/p95 y P(VPN > 0).

### Origen y validez de los valores económicos

Los valores de costo por defecto **no provienen de datos confidenciales de una empresa real** — un proyecto académico no tiene acceso a esa información. En lugar de eso, son **valores ilustrativos calibrados con benchmarks de industria publicados**, elegidos para que el orden de magnitud y las relaciones entre ellos sean defendibles. La interfaz permite al usuario reemplazarlos con datos propios en cualquier momento.

| Parámetro | Default | Razonamiento | Fuente / benchmark |
|---|---|---|---|
| **Costo FN** (Enojo no detectado) | USD 80 | Costo aproximado de churn de un cliente molesto que no recibió escalación. | Reichheld & Sasser, *"Zero Defections: Quality Comes to Services"*, Harvard Business Review (1990) — adquirir un cliente cuesta 5-7× más que retenerlo. Reportes Salesforce State of the Connected Customer (2023): 80% de los clientes considera la experiencia tan importante como el producto. |
| **Costo FP** (falsa alarma) | USD 4 | ~5 minutos de tiempo de un supervisor con costo cargado de ≈ USD 48/hora. | U.S. Bureau of Labor Statistics — *Customer Service Supervisors* OEWS 2023: salario mediano USD 25/hora + ~80% de costos de empleo cargado (impuestos, beneficios, facilities). |
| **Valor TP** (escalación acertada) | USD 25 | Aproximadamente 30% del costo de FN, asumiendo que la escalación a tiempo no garantiza retención total. | Bain & Company — *Prescription for cutting costs*: clientes cuyos problemas se resuelven bien tienen 70% probabilidad de quedarse. |
| **Costo de inferencia** | USD 0.02/llamada | Pricing público de servicios cloud de reconocimiento de voz para audio de ~10 segundos. | AWS Transcribe (USD 0.024/min, 2024); Azure Cognitive Services Speech (USD 0.016/min); Google Cloud Speech-to-Text (USD 0.024/min, primer tier). |
| **Costo fijo mensual** | USD 1 200 | Hosting de un servicio ML pequeño: instancia GPU + monitoring + parte proporcional de ingeniería. | AWS EC2 `g4dn.xlarge` ≈ USD 380/mes on-demand; AWS CloudWatch + S3 ≈ USD 50/mes; resto en mantenimiento prorrateado. |
| **Volumen 10 000 llamadas/mes** | — | Tamaño típico de un call center mediano (pyme). | ContactBabel — *The US Contact Center Decision-Makers' Guide* (anual); ICMI benchmarks 2023. |
| **Prevalencia 18% de Enojo** | — | Proporción de llamadas con expresión clara de enojo en operaciones de servicio. | NICE inContact CX Transformation Benchmark (2023): ~15-25% de llamadas escaladas por insatisfacción. Calabrio Sentiment Analysis benchmarks. |

**Validez de la decisión bajo incertidumbre.** El módulo no depende de que estos números sean exactos. Las dos herramientas críticas de Fase 4 — **análisis de sensibilidad (tornado)** y **simulación Monte Carlo** — están diseñadas precisamente para responder *"¿qué tan robusta es la recomendación si estos valores se mueven?"*:

- El **tornado** muestra qué parámetro tiene más impacto en el VPN si se varía ±30%. En la mayoría de configuraciones, prevalencia y costo FN dominan; el resto son secundarios.
- **Monte Carlo** samplea ±30-60% sobre cada parámetro y devuelve **P(VPN > 0)**: la probabilidad de que la decisión siga siendo rentable bajo incertidumbre.

> **Cómo leer esto en un informe.** Los valores por defecto **fijan un punto de partida razonable**; el análisis de sensibilidad y Monte Carlo **validan que la recomendación sobrevive a errores grandes en la calibración**. Para una implementación real, los valores deberían reemplazarse con datos del área de Finanzas / Operaciones del cliente concreto — la matriz de costos editable de Sección 1 fue construida con ese flujo de trabajo en mente.

---

## Estructura del proyecto

```
clasificador-audios/
├── app.py                                     # Servidor Flask (Backend API)
├── templates/
│   └── index.html                             # 4 pestañas (Clasificador, Dataset, Decisiones, Análisis)
├── static/
│   ├── css/style.css                          # Estilos Glassmorphism + módulo Decisiones
│   └── js/main.js                             # Lógica del cliente + simulador de decisiones
├── notebooks/
│   ├── 01_clasificador_v1_features.ipynb      # Features manuales (144 dims)
│   ├── 02_clasificador_v1_embeddings.ipynb    # wav2vec2 (768 dims)
│   └── 03_clasificador_v2_balanceado.ipynb    # Comparativa final + conclusiones
├── scripts/
│   ├── filtrar_audios.py                      # Scoring acústico
│   ├── exportar_modelos.py                    # Entrenamiento + serialización (.joblib)
│   ├── build_experiment_history.py            # Genera historial de experimentos
│   └── generar_datos_decisiones.py            # ★ NUEVO: extrae CM + probs LOO + ROC
├── outputs/
│   ├── reporte_filtrado_v2.csv                # Scores de los 146 audios
│   ├── model_metrics.json                     # BalAcc y descripción por modelo
│   ├── experiment_history.json                # Historial de experimentos
│   ├── decisions_data.json                    # ★ NUEVO: data para el módulo de Decisiones
│   ├── embeddings_v2.npz                      # Embeddings 3-clase cacheados
│   ├── embeddings_wav2vec2.npz                # Embeddings 2-clase cacheados
│   ├── modelos/                               # 6 clasificadores serializados
│   └── figuras/                               # Gráficos de resultados
├── data/                                      # Dataset original (no versionado)
├── imgs/                                      # Capturas de pantalla
└── README.md                                  # Este archivo
```

---

## Uso rápido

```bash
# 1. Activar entorno
source ../.venv/bin/activate

# 2. (Opcional) Reentrenar los 6 modelos desde cero
python scripts/exportar_modelos.py

# 3. Generar el dataset cuantitativo para el módulo de Decisiones
#    (extrae matrices de confusión y probabilidades LOO desde los embeddings)
python scripts/generar_datos_decisiones.py

# 4. Levantar la app
python app.py

# 5. Abrir en el navegador
# http://127.0.0.1:5001
#   → Tab "Decisiones" para el módulo de Toma de Decisiones
```

El paso 3 produce `outputs/decisions_data.json` (~53 KB) con todo lo necesario para que el cliente haga simulación y sensibilidad sin más roundtrips al backend.

---

## Dependencias

```
flask
librosa>=0.10
torch>=2.0
transformers>=4.30
scikit-learn>=1.3
joblib
numpy
pandas
matplotlib
```

---

## Hallazgos relevantes para la decisión

- **El umbral 0.5 no es óptimo cuando los costos son asimétricos.** Con FN/FP = 20×, el VPN óptimo suele estar en umbrales 0.25-0.35, sacrificando precisión por recall.
- **La prevalencia importa más que la BalAcc.** En el tornado típico, prevalencia y costo_FN dominan; el modelo es relativamente intercambiable mientras esté por encima de 70 % BalAcc.
- **2 emociones vence en decisión, no en cobertura.** El escenario binario consistentemente tiene mejor VPN por su mayor BalAcc (85 %) y menor confusión Enojo↔Feliz; sin embargo, sacrifica la capacidad analítica de detectar satisfacción positiva.
- **Monte Carlo es decisivo.** Algunos modelos (KNN k=5) tienen VPN positivo en promedio pero P(VPN > 0) < 60 % → recomendarían un piloto pequeño antes de escalar.

---

## Contexto académico

- **Curso de Machine Learning**: pipeline de clasificación de emociones (capa 1).
- **Curso de Toma de Decisiones**: módulo analítico de simulación, sensibilidad y recomendación (capa 2).

Dataset recolectado por 4 personas (MT, VZ, VA, ED) con ~21 hablantes distintos por recolector y ~10-30 s por audio.
