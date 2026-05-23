"""Muestra en terminal los audios cuya etiqueta no coincide con lo que oye la IA.

Lee outputs/audios_sospechosos_ia.csv (generado por scripts/auditar_etiquetas_ia.py)
y los presenta agrupados por la clase que la IA propone ("suena_a"), ordenados
por delta descendente (los mas sospechosos primero). Util para escuchar y decidir
reasignaciones manuales.

Uso:
    python scripts/mostrar_sospechosos.py                  # todas las clases
    python scripts/mostrar_sospechosos.py --suena-a Feliz  # solo los que suenan a Feliz
    python scripts/mostrar_sospechosos.py --min-delta 0.10 # solo los muy sospechosos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "outputs" / "audios_sospechosos_ia.csv"
DATA_DIR = ROOT / "data" / "AUDIOS MACHINE LEARNING"
CLASES = ["Enojo", "Feliz", "Tranquilidad", "Tristeza"]


def encontrar_ruta(archivo: str) -> Path | None:
    """Busca el archivo en cualquiera de las carpetas de clase."""
    for clase in CLASES:
        candidato = DATA_DIR / clase / archivo
        if candidato.exists():
            return candidato
    return None


def imprimir_grupo(df_grupo: pd.DataFrame, clase_destino: str) -> None:
    if df_grupo.empty:
        return
    print()
    print("=" * 120)
    print(f"  LA IA DICE QUE SUENAN A: {clase_destino.upper()}  ({len(df_grupo)} audios)")
    print("=" * 120)
    header = (
        f'{"#":>3}  {"Archivo":<22} {"Etiq.actual":<13} {"Enojo":>6} {"Feliz":>6} '
        f'{"Tranq":>6} {"Triste":>6}  {"Delta":>6}  Ruta'
    )
    print(header)
    print("-" * 120)
    for i, (_, row) in enumerate(df_grupo.iterrows(), 1):
        ruta = encontrar_ruta(row["archivo"])
        ruta_str = str(ruta.relative_to(ROOT)) if ruta else "[NO ENCONTRADO]"
        print(
            f'{i:>3}  {row["archivo"]:<22} {row["clase_etiqueta"]:<13} '
            f'{row["score_Enojo"]:>6.3f} {row["score_Feliz"]:>6.3f} '
            f'{row["score_Tranquilidad"]:>6.3f} {row["score_Tristeza"]:>6.3f}  '
            f'{row["delta"]:>6.3f}  {ruta_str}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suena-a",
        choices=CLASES,
        default=None,
        help="Filtrar solo audios que la IA dice que suenan a esta clase.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Mostrar solo audios con delta >= este valor (default 0.0).",
    )
    parser.add_argument(
        "--etiqueta",
        choices=CLASES,
        default=None,
        help="Filtrar solo audios con esta etiqueta actual.",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(
            f"No existe {CSV_PATH}. Corre primero: python scripts/auditar_etiquetas_ia.py"
        )

    df = pd.read_csv(CSV_PATH)
    df = df[df["delta"] >= args.min_delta]
    if args.etiqueta:
        df = df[df["clase_etiqueta"] == args.etiqueta]
    if args.suena_a:
        df = df[df["suena_a"] == args.suena_a]

    df = df.sort_values("delta", ascending=False)

    print()
    print("#" * 120)
    print(
        f'  AUDIOS SOSPECHOSOS SEGUN AUDEERING  —  {len(df)} de {len(pd.read_csv(CSV_PATH))} totales'
    )
    if args.min_delta > 0:
        print(f"  Filtro: delta >= {args.min_delta}")
    if args.etiqueta:
        print(f"  Filtro: etiqueta_actual = {args.etiqueta}")
    if args.suena_a:
        print(f"  Filtro: suena_a = {args.suena_a}")
    print("#" * 120)

    print()
    print("RESUMEN (etiqueta_actual -> lo que oye la IA):")
    tabla = pd.crosstab(df["clase_etiqueta"], df["suena_a"], margins=True, margins_name="TOTAL")
    print(tabla.to_string())

    if args.suena_a:
        imprimir_grupo(df, args.suena_a)
    else:
        for clase in CLASES:
            grupo = df[df["suena_a"] == clase]
            imprimir_grupo(grupo, clase)

    print()
    print("Tip: para escuchar un audio en macOS usa:")
    print('   open "data/AUDIOS MACHINE LEARNING/<Clase>/<archivo>"')


if __name__ == "__main__":
    main()
