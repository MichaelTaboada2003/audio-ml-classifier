"""Configuración global del proyecto.

Centraliza los parámetros que afectan al entrenamiento y al dashboard para
poder escalar el dataset (más audios por clase) cambiando un solo valor.
"""

import os


# Cantidad de audios por clase a usar en el experimento "actual" (3 clases).
# Cambia este valor cuando agregues más audios y vuelve a correr el pipeline:
#   1. python scripts/filtrar_audios.py
#   2. python scripts/exportar_modelos.py           (regenera cache + modelos)
#   3. python scripts/build_experiment_history.py   (refresca historial)
#   4. python scripts/generar_datos_decisiones.py   (refresca módulo Decisiones)
#
# Puedes override puntualmente vía variable de entorno:
#   N_PER_CLASS=20 python scripts/exportar_modelos.py
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 15))

# Score minimo para considerar un audio "valido" en su clase.
# 0.0 = filtro acustico desactivado: el score sigue siendo el criterio de
# ranking (top-N por score) pero NO se descartan audios por umbral.
# Confiamos en que wav2vec2 captura senal que las 7 metricas globales no ven.
MIN_SCORE = 0.0

# Clases activas del experimento actual.
# Arrancamos con 2 clases (Enojo vs Tristeza) como baseline limpio. Chance=50%.
# Despues escalamos a 3 (+ Feliz) y a 4 (+ Tranquilidad).
ACTIVE_CLASSES = ["Enojo", "Feliz", "Tranquilidad", "Tristeza"]

# Holdout para el "Explorador del Dataset" del front: audios reservados que NO
# entran al entrenamiento del modelo desplegado (.joblib), para que predecir
# sobre ellos en el dashboard sea genuinamente out-of-sample.
# Solo clases con datos de sobra (Enojo/Feliz son escasas -> 0; usar el microfono).
# Las metricas honestas de Analisis (LOO/leave-one-collector-out) se siguen
# estimando sobre TODOS los segmentos; esto solo afecta al .joblib y al explorador.
DASHBOARD_HOLDOUT = {"Tranquilidad": 18, "Tristeza": 12}
DASHBOARD_HOLDOUT_SEED = 42
