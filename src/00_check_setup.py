"""
Fase 0 - Verifica dell'ambiente e prima ispezione dei dati.

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/00_check_setup.py

Cosa fa:
  1. controlla che le librerie necessarie siano installate;
  2. legge il CSV dei metadati e riassume coorti, split, pCR e sottotipi;
  3. ispeziona i file NIfTI dei dati demo (nomi, dimensioni, valori);
  4. salva tutto in outputs/check_setup_report.txt.
"""

import importlib
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = DATA / "BreastDCEDL_metadata_min_crop.csv"
DEMO = DATA / "demo"
OUT = ROOT / "outputs"

LIBS = ["numpy", "scipy", "pandas", "nibabel", "sklearn",
        "statsmodels", "dowhy", "matplotlib"]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_libraries():
    section("1. LIBRERIE")
    print(f"Python {sys.version.split()[0]}")
    missing = []
    for name in LIBS:
        try:
            mod = importlib.import_module(name)
            print(f"  OK       {name:12s} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  MANCANTE {name:12s} ({type(e).__name__}: {e})")
            missing.append(name)
    if missing:
        print("\n  -> Installa con: pip install -r requirements.txt")
    return not missing


def check_metadata():
    section("2. METADATI (CSV)")
    if not CSV.exists():
        print(f"  File non trovato: {CSV}")
        print("  -> Esegui prima: bash scripts/download_data.sh")
        return
    import pandas as pd

    df = pd.read_csv(CSV)
    print(f"Righe (pazienti): {len(df)}   Colonne: {df.shape[1]}")
    print("Colonne:", ", ".join(df.columns))

    print("\nPazienti per coorte (colonna 'dataset'):")
    print(df["dataset"].value_counts(dropna=False).to_string())

    if "test" in df.columns:
        print("\nValori della colonna 'test' (split) per coorte:")
        print(pd.crosstab(df["dataset"], df["test"], dropna=False).to_string())

    print("\nDisponibilita' della pCR per coorte:")
    print(df.groupby("dataset")["pCR"].agg(
        totale="size", con_pCR="count", tasso_pCR="mean").round(3).to_string())

    if "HR_HER2_STATUS" in df.columns:
        print("\nSottotipo x coorte:")
        print(pd.crosstab(df["HR_HER2_STATUS"], df["dataset"], dropna=False).to_string())
        print("\nTasso di pCR per sottotipo (solo pazienti con pCR):")
        print(df.dropna(subset=["pCR"]).groupby("HR_HER2_STATUS")["pCR"]
              .agg(n="size", tasso_pCR="mean").round(3).to_string())

    print("\nValori mancanti per colonna (solo colonne con mancanti):")
    na = df.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "  nessuno")

    print("\nPrime 3 righe:")
    print(df.head(3).to_string())


def check_demo_images():
    section("3. DATI DEMO (NIfTI)")
    if not DEMO.exists():
        print(f"  Cartella non trovata: {DEMO}")
        print("  -> Esegui prima: bash scripts/download_data.sh")
        return
    files = sorted(list(DEMO.rglob("*.nii.gz")) + list(DEMO.rglob("*.nii")))
    others = sorted(p for p in DEMO.rglob("*") if p.is_file() and p not in files)
    print(f"File NIfTI trovati: {len(files)}   Altri file: {len(others)}")

    print("\nStruttura delle cartelle (primi 2 livelli):")
    dirs = sorted({p.relative_to(DEMO).parts[:2] for p in DEMO.rglob("*") if p.is_dir()})
    for d in dirs[:30]:
        print("  ", "/".join(d))

    if others:
        print("\nAltri file (primi 15):")
        for p in others[:15]:
            print("  ", p.relative_to(DEMO))

    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        print("  nibabel/numpy non installati: salto l'ispezione delle immagini.")
        return

    print("\nDettaglio dei primi 25 file NIfTI:")
    for p in files[:25]:
        try:
            img = nib.load(str(p))
            arr = np.asanyarray(img.dataobj)
            zooms = tuple(round(float(z), 3) for z in img.header.get_zooms())
            print(f"  {str(p.relative_to(DEMO)):60s} shape={arr.shape} "
                  f"dtype={arr.dtype} voxel={zooms} "
                  f"min={arr.min():.1f} max={arr.max():.1f} "
                  f"valori_unici={'<=2 (maschera?)' if np.unique(arr).size <= 2 else 'molti'}")
        except Exception as e:  # noqa: BLE001
            print(f"  {p.relative_to(DEMO)}: errore di lettura ({e})")

    total = sum(p.stat().st_size for p in DEMO.rglob("*") if p.is_file())
    print(f"\nSpazio occupato dai dati demo: {total / 1e6:.1f} MB")


def main():
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buf.write(s)

        def flush(self):
            sys.__stdout__.flush()

    with redirect_stdout(Tee()):
        ok = check_libraries()
        check_metadata()
        check_demo_images()
        section("ESITO")
        print("Ambiente OK." if ok else "Mancano alcune librerie (vedi sezione 1).")

    OUT.mkdir(exist_ok=True)
    report = OUT / "check_setup_report.txt"
    report.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nReport salvato in: {report}")


if __name__ == "__main__":
    main()
