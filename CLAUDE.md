# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

**EmotiSpeech** — clasificador de emociones de voz (Enojo / Tristeza / Feliz) con dos capas conceptualmente separadas:

1. **Capa ML** (notebooks + `scripts/exportar_modelos.py`): embeddings `facebook/wav2vec2-base` (768 dims) → 6 clasificadores clásicos (SVM lineal/RBF, LogReg, RF, KNN k=3/k=5) evaluados con LOOCV.
2. **Capa Decisiones** (`scripts/generar_datos_decisiones.py` + tab "Decisiones" del front): consume probabilidades LOO + matrices de confusión ya calculadas y las convierte en simulador de VPN/sensibilidad/Monte Carlo para una decisión de negocio (call-center).

La capa de Decisiones **no modifica** la capa ML: lee los `.joblib` y los embeddings cacheados.

## Comandos comunes

```bash
# Activar entorno (vive un nivel arriba)
source ../.venv/bin/activate

# Pipeline ML completo (orden importa)
python scripts/filtrar_audios.py              # 1. scores acústicos por audio  → outputs/reporte_filtrado_v2.csv
python scripts/exportar_modelos.py            # 2. embeddings + entrena 6 modelos LOOCV → outputs/modelos/*.joblib + model_metrics.json
python scripts/build_experiment_history.py    # 3. historial para el dashboard → outputs/experiment_history.json
python scripts/generar_datos_decisiones.py    # 4. data del módulo Decisiones  → outputs/decisions_data.json

# Levantar la app Flask (puerto 5001)
python app.py

# Override de N_PER_CLASS sin tocar config.py
N_PER_CLASS=20 python scripts/exportar_modelos.py
```

No hay tests, lint ni typecheck configurados — los notebooks y métricas LOOCV en `model_metrics.json` actúan como verificación.

## Arquitectura

### Single source of truth: `config.py`

`config.py` expone `N_PER_CLASS` y `ACTIVE_CLASSES`. **Todos los scripts y `app.py` lo importan** y se reconfiguran automáticamente: cambiar `N_PER_CLASS` y re-correr el pipeline regenera embeddings, modelos, métricas y dashboard de forma consistente. Soporta override por env var (`N_PER_CLASS=20 python ...`).

### Flujo de datos (importante para entender el repo)

```
data/AUDIOS MACHINE LEARNING/{Clase}/*.ogg
        │
        ▼  filtrar_audios.py          (scoring acústico por audio, NO reasigna clases)
outputs/reporte_filtrado_v2.csv
        │
        ▼  exportar_modelos.py        (top-N_PER_CLASS por clase → wav2vec2 → LOOCV)
outputs/embeddings_v2.npz             ← cache de embeddings (768 dims × N_PER_CLASS × 3)
outputs/modelos/*.joblib              ← 6 modelos entrenados con TODO el data (no LOOCV-fold)
outputs/model_metrics.json            ← BalAcc LOOCV por modelo + best_model
        │
        ├─▶ generar_datos_decisiones.py
        │   outputs/decisions_data.json   ← CMs + probs LOO + ROC para el simulador
        │
        └─▶ build_experiment_history.py
            outputs/experiment_history.json
                                        │
                                        ▼
                                    app.py + index.html (4 tabs)
```

**`outputs/embeddings_v2.npz` actúa como caché crítico**: `exportar_modelos.py` lo reutiliza si las cuentas por clase coinciden con `N_PER_CLASS`, evitando reextraer embeddings (lento por wav2vec2). Si cambias `N_PER_CLASS` o `ACTIVE_CLASSES`, **fuerza rebuild** corriendo `exportar_modelos.py` directamente (su `__main__` ya pasa `force_rebuild=True`).

### Capa Decisiones — patrón

El front (`static/js/main.js`) recibe `decisions_data.json` precalculado y hace toda la simulación en cliente: recalcula matrices de confusión moviendo el umbral sobre las probabilidades LOO, busca el umbral óptimo en VPN, corre Monte Carlo (2 000 muestras) y construye el tornado de sensibilidad. **No hay roundtrips al backend para la simulación**; el único endpoint relacionado es `GET /api/decision/data` que sirve el JSON estático.

### Inferencia en vivo (`app.py`)

`init_models()` carga los 6 `.joblib` + wav2vec2 en memoria al arrancar. `extract_features_and_predict()` corre el pipeline completo (audio → embedding → modelo seleccionado) y devuelve también métricas acústicas (pitch, energía, ZCR) usadas en el front para mostrar contexto del audio. Sube audio vía `/api/predict_upload` o predice sobre un audio del dataset vía `/api/predict_dataset`.

## Convenciones del proyecto

- **N_PER_CLASS = 20, desbalance asumido en Feliz.** `seleccionar_audios()` cap a N pero tolera menos disponibles. Cache real: Enojo=20, Tristeza=20, Feliz=15 (todos los disponibles). Los modelos usan `class_weight='balanced'` para compensar. **No subir a N=25** sin agregar más audios — ya estás raspando NO PASA en Tristeza.
- **`class_weight='balanced'` en SVMs/LogReg/RF.** Tres scripts duplican MODEL_SPECS (`exportar_modelos.py`, `build_experiment_history.py`, `generar_datos_decisiones.py`). Si cambias hiperparámetros en uno, **propaga a los tres** o las métricas reportadas en distintos sitios divergen.
- **MAX_DURATION = 10.0 segundos.** Sweet spot empírico para este dataset. Probamos audio completo (modelos lineales colapsan a 0% por dilución del mean-pool), 15s, chunks de 8s con agregación, y warm-up de 1-2s — todas variantes empeoran el bal_acc. Ver tabla en README "Decisiones de diseño verificadas empíricamente".
- **No reasignar clases.** `filtrar_audios.py` calcula scores acústicos pero **mantiene la `clase_original`** de cada audio. La selección top-N usa el score de la clase declarada, no la mejor predicción acústica.
- **LOOCV es la métrica oficial intra-training.** El número que importa para el dashboard es **holdout BalAcc** sobre audios no usados en training. Hoy: ~50% SVM lineal en holdout vs ~67% LOOCV LogReg en training. La brecha refleja distribution shift entre training (top-score) y holdout (mid-score). No es bug.
- **El "Explorador del Dataset" del frontend muestra audios HOLDOUT.** `/api/dataset` excluye los archivos en el cache de embeddings (los que entraron al training). Predicciones desde esa pestaña son genuinamente out-of-sample. Hoy: ~17 Enojo + ~17 Tristeza + 0 Feliz (todos los Feliz están en training).
- **`outputs/figuras/*.png` se sirven desde la app** (`/outputs/figuras/<filename>`); muchos paths están hardcodeados en `experiment_history.json`. Si renombras una figura, actualiza también `build_experiment_history.py`.
- **Los `.npz` están gitignored** pero son regenerables corriendo el pipeline. `data/` también gitignored.

## Notas de modelo / pitfalls conocidos

- `wav2vec2-base` se descarga al primer uso (~360 MB). Se mueve a `mps` en Mac, `cuda` si está disponible, si no `cpu`.
- `app.py` siempre usa `cuda` o `cpu` (no `mps`); `exportar_modelos.py` sí usa `mps`. No es bug — la app es Flask en debug y `mps` ha dado problemas con `torch.no_grad()` en ese contexto.
- Los modelos en `outputs/modelos/*.joblib` están entrenados con **todo** el dataset (no con fold LOOCV) — son los que se exponen en producción. Las métricas LOOCV son una estimación honesta del desempeño, no se corresponden con un fold específico de los binarios.
- **Predicción en producción ≠ LOOCV.** Si pruebas un audio que está en `outputs/reporte_filtrado_v2.csv` en posición top-14 (es decir, está en training), el modelo lo memorizó y reportará confianza muy alta (95-99%). Eso NO valida nada. Para validar honestamente, usa audios fuera del cache (que es lo que sirve `/api/dataset`).
- Los audios de entrada se truncan a 10 s (`MAX_DURATION = 10.0`) y se resamplean a 16 kHz (`TARGET_SR`). El valor debe coincidir entre `exportar_modelos.py` y `app.py` o las predicciones serán inconsistentes.
- **Recolector ≠ hablante.** Los prefijos MT/VZ/VA/ED identifican a la persona que **recolectó** cada audio, no quién habla. Cada audio es de un hablante distinto, así que LOOCV es efectivamente leave-one-speaker-out. Lo que sí persiste es condición-de-grabación overlap (solo 4 micrófonos/ambientes), pero ese efecto es de orden secundario respecto al speaker overlap que NO existe en este dataset.
- El navegador graba en formato `webm/opus`; el front convierte a WAV PCM antes de subir porque el backend no decodifica opus directamente (ver commit `3e11824`).
- Los **notebooks** en `notebooks/` documentan iteraciones históricas (N=10, configuraciones distintas) y NO son la fuente de verdad. El pipeline canónico vive en `scripts/`.

## Contexto académico

Proyecto integrado de dos cursos: Machine Learning (capa 1) y Toma de Decisiones (capa 2). Dataset recolectado por 4 personas (prefijos `MT`, `VZ`, `VA`, `ED` en los nombres de archivo) con ~10-30 s por audio.
