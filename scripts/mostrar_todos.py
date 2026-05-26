"""Muestra en terminal los scores emocionales de TODOS los audios de TODAS las carpetas.

Hermano de scripts/mostrar_sospechosos.py, pero sin filtrar por delta: lista todos
los audios (bien etiquetados, borderline y sospechosos por igual), agrupados por su
etiqueta actual (la carpeta donde viven) y mostrando los 4 scores por clase del modelo
emocional audeering, los valores AVD crudos, lo que "suena_a" la IA y el delta.

Por defecto lee el dataset original (data/AUDIOS MACHINE LEARNING +
outputs/audios_sospechosos_ia.csv). El CSV contiene un row por audio (generado por
scripts/auditar_etiquetas_ia.py). Si el CSV no coincide con las carpetas, lo avisa.

Para ver otro dataset (p. ej. segmentos localizados) pasa --csv y --data apuntando
a su CSV de auditoria y a su carpeta raiz de clases.

Uso:
    python scripts/mostrar_todos.py                    # dataset original (default)
    python scripts/mostrar_todos.py --clase Tristeza   # solo la carpeta Tristeza
    python scripts/mostrar_todos.py --orden delta      # ordenar cada grupo por delta desc

    # Otro dataset (p. ej. segmentos recortados):
    python scripts/mostrar_todos.py \\
        --csv outputs/mi_auditoria.csv \\
        --data outputs/mi_carpeta_segmentos
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "outputs" / "audios_sospechosos_ia.csv"
DATA_DIR = ROOT / "data" / "AUDIOS MACHINE LEARNING"
EXTS = {".ogg", ".mp3", ".mp4", ".mpeg", ".wav", ".flac", ".m4a"}


def carpetas() -> list[str]:
    """Subcarpetas de clase presentes en data/ (incluye 'borderline' si existe)."""
    if not DATA_DIR.exists():
        return []
    return sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir())


def encontrar_ruta(archivo: str, clases: list[str]) -> Path | None:
    """Busca el archivo en cualquiera de las carpetas de clase."""
    for clase in clases:
        candidato = DATA_DIR / clase / archivo
        if candidato.exists():
            return candidato
    return None


def fmt(val, ancho: int = 6, dec: int = 3) -> str:
    """Formatea un float o muestra '—' si esta vacio/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == "":
        return f'{"—":>{ancho}}'
    return f"{float(val):>{ancho}.{dec}f}"


def imprimir_grupo(df_grupo: pd.DataFrame, clase: str, clases_disp: list[str]) -> None:
    if df_grupo.empty:
        return
    print()
    print("=" * 124)
    print(f"  CARPETA / ETIQUETA: {clase.upper()}  ({len(df_grupo)} audios)")
    print("=" * 124)
    header = (
        f'{"#":>3}  {"Archivo":<22} {"Rec":<4} '
        f'{"A":>5} {"V":>5} {"D":>5}  '
        f'{"Enojo":>6} {"Feliz":>6} {"Tranq":>6} {"Triste":>6}  '
        f'{"Suena_a":<13} {"Delta":>6}'
    )
    print(header)
    print("-" * 124)
    for i, (_, row) in enumerate(df_grupo.iterrows(), 1):
        suena = row.get("suena_a", "")
        suena = "—" if (pd.isna(suena) or suena == "") else str(suena)
        print(
            f'{i:>3}  {str(row["archivo"]):<22} {str(row.get("recolector","")):<4} '
            f'{fmt(row.get("arousal"),5)} {fmt(row.get("valence"),5)} {fmt(row.get("dominance"),5)}  '
            f'{fmt(row.get("score_Enojo"))} {fmt(row.get("score_Feliz"))} '
            f'{fmt(row.get("score_Tranquilidad"))} {fmt(row.get("score_Tristeza"))}  '
            f'{suena:<13} {fmt(row.get("delta"))}'
        )


def aviso_desincronizado(df: pd.DataFrame, clases_disp: list[str]) -> None:
    """Compara los archivos del CSV con los de las carpetas y avisa si difieren."""
    en_csv = set(df["archivo"].astype(str))
    en_disco = {
        p.name
        for clase in clases_disp
        for p in (DATA_DIR / clase).iterdir()
        if p.suffix.lower() in EXTS
    }
    faltan = en_disco - en_csv
    sobran = en_csv - en_disco
    if faltan or sobran:
        print()
        print("!" * 124)
        print("  AVISO: el CSV no coincide con las carpetas (puede estar desactualizado).")
        print(f"    audios en disco no presentes en el CSV: {len(faltan)}")
        if faltan:
            print("      " + ", ".join(sorted(faltan)[:12]) + (" ..." if len(faltan) > 12 else ""))
        print(f"    filas en el CSV sin archivo en disco:   {len(sobran)}")
        if sobran:
            print("      " + ", ".join(sorted(sobran)[:12]) + (" ..." if len(sobran) > 12 else ""))
        print("  Re-genera con: python scripts/auditar_etiquetas_ia.py")
        print("!" * 124)


def main() -> None:
    global CSV_PATH, DATA_DIR
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--clase",
        default=None,
        help="Filtrar solo audios de esta carpeta/etiqueta.",
    )
    parser.add_argument(
        "--orden",
        choices=["score", "delta", "archivo"],
        default="score",
        help="Orden dentro de cada grupo: por score de la clase etiquetada desc (default), "
             "por delta desc, o por nombre de archivo.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="CSV de auditoria a leer (default: outputs/audios_sospechosos_ia.csv).",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Carpeta raiz con subcarpetas de clase (default: data/AUDIOS MACHINE LEARNING).",
    )
    args = parser.parse_args()

    if args.csv:
        CSV_PATH = Path(args.csv)
    if args.data:
        DATA_DIR = Path(args.data)

    clases_disp = carpetas()
    if args.clase and args.clase not in clases_disp:
        raise SystemExit(f"--clase '{args.clase}' no existe en {DATA_DIR}. Disponibles: {clases_disp}")

    if not CSV_PATH.exists():
        raise SystemExit(
            f"No existe {CSV_PATH}. Corre primero: python scripts/auditar_etiquetas_ia.py"
        )

    df = pd.read_csv(CSV_PATH)

    # 'suena_a' del CSV es la mejor clase DISTINTA a la etiqueta (para detectar discrepancias).
    # Aqui lo redefinimos como la clase con el score mas alto de las 4, sin excluir nada.
    SCORE_COLS = {
        "score_Enojo": "Enojo",
        "score_Feliz": "Feliz",
        "score_Tranquilidad": "Tranquilidad",
        "score_Tristeza": "Tristeza",
    }
    cols_score = [c for c in SCORE_COLS if c in df.columns]
    df["suena_a"] = df[cols_score].idxmax(axis=1).map(SCORE_COLS)

    print()
    print("#" * 124)
    print(f"  SCORES EMOCIONALES (audeering) — TODOS los audios  —  {len(df)} filas en el CSV")
    print("#" * 124)

    aviso_desincronizado(df, clases_disp)

    if args.clase:
        df = df[df["clase_etiqueta"] == args.clase]

    print()
    print("RESUMEN (etiqueta_actual -> lo que oye la IA):")
    tabla = pd.crosstab(df["clase_etiqueta"], df["suena_a"], margins=True, margins_name="TOTAL")
    print(tabla.to_string())

    # Orden de los grupos: el orden de las carpetas en disco, luego cualquier extra del CSV.
    orden_clases = [c for c in clases_disp if c in set(df["clase_etiqueta"])]
    orden_clases += [c for c in df["clase_etiqueta"].unique() if c not in orden_clases]

    for clase in orden_clases:
        grupo = df[df["clase_etiqueta"] == clase].copy()
        if args.orden == "score":
            # Score de la clase etiquetada; borderline (sin clase propia) cae al score mas alto.
            col = f"score_{clase}" if f"score_{clase}" in grupo.columns else "score_suena_a"
            grupo = grupo.sort_values(col, ascending=False, na_position="last")
        elif args.orden == "delta":
            grupo = grupo.sort_values("delta", ascending=False, na_position="last")
        else:
            grupo = grupo.sort_values("archivo")
        imprimir_grupo(grupo, clase, clases_disp)

    print()
    print("Tip: para escuchar un audio en macOS usa:")
    try:
        base = DATA_DIR.relative_to(ROOT)
    except ValueError:
        base = DATA_DIR
    print(f'   open "{base}/<Clase>/<archivo>"')


if __name__ == "__main__":
    main()
