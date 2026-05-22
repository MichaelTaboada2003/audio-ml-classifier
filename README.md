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
- [Capa 2 · Toma de Decisiones](#capa-2--toma-de-decisiones)
  - [Problema de decisión](#problema-de-decisión)
  - [Estructura del módulo](#estructura-del-módulo)
  - [Sección 1 · Contexto y matriz de costos](#sección-1--contexto-y-matriz-de-costos)
  - [Sección 2 · Datos para decidir](#sección-2--datos-para-decidir)
  - [Sección 3 · Simulador de despliegue](#sección-3--simulador-de-despliegue)
  - [Sección 4 · Análisis de decisiones](#sección-4--análisis-de-decisiones)
  - [Recomendación final](#recomendación-final)
  - [Modelo matemático](#modelo-matemático)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Uso rápido](#uso-rápido)
- [Dependencias](#dependencias)

---

## Resumen rápido

| Capa | Qué hace | Salida |
|---|---|---|
| ML | Clasifica audio en Enojo / Tristeza / Feliz con 6 modelos clásicos sobre embeddings wav2vec2 (768 dims) | Predicción + probabilidades + métricas acústicas |
| Decisiones | Convierte las matrices de confusión y probabilidades LOO en una decisión de negocio (GO / NO-GO, umbral óptimo, VPN esperado) | Recomendación justificada con sensibilidad y Monte Carlo |

**Mejor modelo actual:** SVM lineal con **76.7 %** de balanced accuracy en el escenario de 3 emociones (chance = 33.3 %).

---

## Aplicación web — pestañas disponibles

La app Flask (`app.py`) levanta una interfaz con 4 pestañas:

1. **Clasificador en Vivo** — sube o graba audio y obtén predicción con confianza y métricas acústicas.
2. **Explorador del Dataset** — reproduce los audios mejor calificados por clase y prueba el modelo sobre ellos.
3. **Decisiones** *(nuevo módulo)* — el corazón de la capa de Toma de Decisiones, descrito en detalle abajo.
4. **Análisis y Métricas** — figuras del entrenamiento (matriz de confusión, separabilidad, comparativa).

---

## Capa 1 · Machine Learning

### Resultado principal

| Enfoque | Mejor modelo | Accuracy | Balanced Acc |
|---|---|---|---|
| Features manuales (144 dims) | SVM lineal / Random Forest | 0.85 | 0.8462 |
| wav2vec2 embeddings (768 dims, 3 clases) | SVM lineal | **0.7667** | **0.7667** |
| wav2vec2 embeddings (768 dims, 2 clases) | KNN k=5 / SVM RBF / RF | 0.85 | 0.85 |

### Dataset

| Parámetro | Valor |
|---|---|
| Total de audios | 146 |
| Clases originales | Aburrido (36), Enojo (37), Tranquilidad (36), Tristeza (37) |
| Recolectores | MT, VZ, VA, ED |
| Duración por audio | ~10-30 segundos |

### Pipeline ML

```
1. Scoring acústico (scripts/filtrar_audios.py)
   → Calcula 5 scores por audio (Enojo, Tristeza, Tranquilidad, Aburrido, Feliz)
   → Genera outputs/reporte_filtrado_v2.csv

2. Selección por ranking (top-N por clase)

3. Extracción de embeddings con wav2vec2 (cache: outputs/embeddings_v2.npz)

4. Entrenamiento LOOCV + serialización (scripts/exportar_modelos.py)
   → outputs/modelos/*.joblib (6 modelos)
   → outputs/model_metrics.json
```

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
