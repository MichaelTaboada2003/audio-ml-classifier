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
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 14))

# Score mínimo para considerar un audio válido por su clase
MIN_SCORE = 0.35

# Clases activas del experimento actual
ACTIVE_CLASSES = ["Enojo", "Tristeza", "Feliz"]
