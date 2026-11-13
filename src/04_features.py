"""
Fase 4 - Biomarcatori di eterogeneita' dagli habitat (una riga per paziente).

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/04_features.py

Cosa fa:
  1. legge gli habitat di outputs/habitats_k<k>/<pid>.npz (Fase 3) per le pazienti con
     exclude == False in outputs/maps_qc.csv, e da outputs/cohort.csv solo pid, coorte, split,
     tum_vol e mask_frac_in_crop (la pCR non viene mai letta);
  2. per ogni paziente calcola:
     - frazione di volume di ogni habitat;
     - entropia di Shannon delle frazioni, divisa per log k (0 = un solo habitat, 1 = parti uguali);
     - per habitat, numero di regioni connesse (26 vicini) di almeno 50 mm^3 (descrittivo) e
       frammentazione = 1 - frazione dell'habitat nella sua regione piu' grande (per i modelli;
       NaN se l'habitat e' assente o piu' piccolo di 50 mm^3 in tutto);
     - adiacenza = quota della superficie di contatto tra voxel validi vicini (facce condivise,
       pesate con l'area in mm^2) che separa due habitat diversi; le quote per coppia di habitat
       sono solo descrittive;
     - tum_vol dal CSV (cm^3) e il flag crop_flag = mask_frac_in_crop < 0,95;
  3. filtro delle feature per i modelli: correlazione di Spearman sulle sole pazienti del
     TRAINING, soglia |rho| > 0,9; in una coppia sopra soglia si scarta la feature con priorita'
     piu' bassa (tum_vol > frazioni > entropia > adiacenza > frammentazione);
  4. ripete tutto per le analisi di sensibilita': maschera erosa nel piano (`eroded2d`, stesso
     elenco di feature dell'analisi principale) e k = 2 (filtro ricalcolato sul training);
  5. calcola le stesse feature (k = 3, maschera intera) per le 6 pazienti escluse in Fase 2,
     in un file a parte, solo per l'analisi di sensibilita' della Fase 5;
  6. salva outputs/features.csv (analisi principale), outputs/features_eroded2d.csv,
     outputs/features_k2.csv, outputs/features_qc_excluded.csv, outputs/features_selection.csv
     e outputs/features_report.txt.

Decisioni (concordate, vedi docs/roadmap.md, Fase 4):
  - frazioni: tutte nel file; nei modelli si toglie H1 (riferimento);
  - le regioni sotto 50 mm^3 (soglia in mm^3 perche' il volume del voxel varia di 8 volte tra le
    pazienti) non entrano nel conteggio: sono in gran parte rumore dell'assegnazione voxel per
    voxel. La frammentazione usa invece tutti i voxel dell'habitat;
  - connettivita' a 26 vicini: con fette di 2-4 mm il vicino in diagonale sulla fetta accanto e'
    lontano quasi quanto quello dritto sopra;
  - contatti pesati con l'area della faccia: l'indice non dipende dallo spessore delle fette.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

K_MAIN = 3
MIN_REGION_MM3 = 50.0       # regioni piu' piccole escluse dal conteggio
CROP_THR = 0.95             # mask_frac_in_crop sotto soglia = tumore tagliato dal ritaglio
CORR_THR = 0.9              # |rho di Spearman| oltre cui due feature sono "doppie"
STRUCT26 = ndimage.generate_binary_structure(3, 3)
COHORT_COLS = ["pid", "split", "tum_vol", "mask_frac_in_crop"]   # mai la pCR

# (nome, file, k, maschera, elenco delle feature)
ANALYSES = [
    ("principale", "features.csv", 3, "intera", "filtro"),
    ("eroded2d", "features_eroded2d.csv", 3, "eroded2d", "principale"),
    ("k2", "features_k2.csv", 2, "intera", "filtro"),
]


# --------------------------------------------------------------------------
# Calcolo (funzioni pure, testate in tests/test_04_features.py)
# --------------------------------------------------------------------------
def fractions(hab, k):
    """Frazione dei voxel validi (hab > 0) in ciascun habitat 1..k (NaN se nessun voxel)."""
    v = np.asarray(hab)[np.asarray(hab) > 0]
    if v.size == 0:
        return np.full(k, np.nan)
    return np.bincount(v, minlength=k + 1)[1:k + 1] / v.size


def entropy_norm(fr):
    """Entropia di Shannon delle frazioni divisa per log k: 0 = un habitat, 1 = parti uguali."""
    fr = np.asarray(fr, dtype=np.float64)
    if np.any(np.isnan(fr)):
        return np.nan
    p = fr[fr > 0]
    return float(-(p * np.log(p)).sum() / np.log(fr.size)) + 0.0   # + 0.0: niente "-0.0"


def regions(mask, spacing_zyx, min_mm3=MIN_REGION_MM3, structure=STRUCT26):
    """Regioni connesse di una maschera binaria.

    Restituisce (numero di regioni con volume >= min_mm3, frammentazione), dove
    frammentazione = 1 - voxel nella regione piu' grande / voxel totali. La frammentazione e'
    NaN se la maschera e' vuota o se in tutto e' piu' piccola di min_mm3: un habitat di pochi
    voxel non ha una forma da misurare e il valore sarebbe rumore."""
    lab, n = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    if n == 0:
        return 0, np.nan
    sizes = np.bincount(lab.ravel())[1:]
    vox_mm3 = float(np.prod(spacing_zyx))
    n_big = int((sizes * vox_mm3 >= min_mm3).sum())
    if sizes.sum() * vox_mm3 < min_mm3:
        return n_big, np.nan
    return n_big, float(1 - sizes.max() / sizes.sum())


def contact_areas(hab, spacing_zyx, k):
    """Matrice k x k della superficie di contatto (mm^2) tra voxel validi vicini per faccia.

    C[i, j] (i != j) = area delle facce tra habitat i+1 e j+1 (simmetrica);
    C[i, i] = area delle facce interne all'habitat i+1. I voxel 0 (non validi) non contano."""
    hab = np.asarray(hab)
    dz, dy, dx = (float(s) for s in spacing_zyx)
    face = (dy * dx, dz * dx, dz * dy)          # area della faccia perpendicolare all'asse
    C = np.zeros((k + 1, k + 1))
    for ax in range(3):
        a = np.moveaxis(hab, ax, 0)
        x, y = a[:-1].ravel(), a[1:].ravel()
        m = (x > 0) & (y > 0)
        np.add.at(C, (x[m], y[m]), face[ax])
    C = C[1:, 1:]
    return C + C.T - np.diag(np.diag(C))


def adjacency(C):
    """(indice globale, {(i, j): quota}) con quote rispetto alla superficie di contatto totale.

    Indice globale = superficie tra habitat diversi / superficie totale (NaN se nessun contatto)."""
    k = C.shape[0]
    tot = np.triu(C).sum()
    pairs = {(i + 1, j + 1): (C[i, j] / tot if tot > 0 else np.nan)
             for i in range(k) for j in range(i + 1, k)}
    glob = (np.triu(C, 1).sum() / tot) if tot > 0 else np.nan
    return float(glob), pairs


def patient_features(hab, spacing_zyx, k, min_mm3=MIN_REGION_MM3):
    """Dizionario delle feature di una paziente dalla mappa degli habitat (0 = non valido)."""
    hab = np.asarray(hab)
    fr = fractions(hab, k)
    n_valid = int((hab > 0).sum())
    row = {"n_valid": n_valid,
           "vol_valid_cm3": n_valid * float(np.prod(spacing_zyx)) / 1000}
    row.update({f"frac_h{h + 1}": fr[h] for h in range(k)})
    row["entropy"] = entropy_norm(fr)
    for h in range(1, k + 1):
        n, frag = regions(hab == h, spacing_zyx, min_mm3)
        row[f"nreg_h{h}"] = n
        row[f"frag_h{h}"] = frag
    glob, pairs = adjacency(contact_areas(hab, spacing_zyx, k))
    row["adj"] = glob
    if k > 2:                                   # con k = 2 l'unica coppia coincide con adj
        row.update({f"adj_h{i}h{j}": v for (i, j), v in pairs.items()})
    return row


def model_candidates(k):
    """Feature candidate per i modelli, in ordine di priorita' (prima = mai scartata)."""
    return (["tum_vol"] + [f"frac_h{h}" for h in range(2, k + 1)] + ["entropy", "adj"]
            + [f"frag_h{h}" for h in range(1, k + 1)])


def corr_filter(df, cols, thr=CORR_THR):
    """Filtro per correlazione di Spearman sulle righe di df (da passare SOLO il training).

    cols e' in ordine di priorita': si scorre l'elenco e si tiene una feature se |rho| <= thr
    con tutte quelle gia' tenute. Restituisce (tenute, scartate) con scartate = lista di
    (feature, feature tenuta che la rende doppia, rho)."""
    rho = df[cols].corr(method="spearman")
    kept, dropped = [], []
    for c in cols:
        hit = [(kc, rho.loc[c, kc]) for kc in kept if abs(rho.loc[c, kc]) > thr]
        if hit:
            kc, r = max(hit, key=lambda t: abs(t[1]))
            dropped.append((c, kc, float(r)))
        else:
            kept.append(c)
    return kept, dropped


# --------------------------------------------------------------------------
# Dati
# --------------------------------------------------------------------------
def load_patients(excluded=False):
    """Pazienti con coorte, split, tum_vol, mask_frac_in_crop: le incluse (exclude == False)
    oppure, con excluded=True, le sole escluse dal controllo di qualita' della Fase 2."""
    qc = pd.read_csv(OUT / "maps_qc.csv")
    cohort = pd.read_csv(OUT / "cohort.csv", usecols=COHORT_COLS)
    sel = qc["exclude"] if excluded else ~qc["exclude"]
    df = qc.loc[sel, ["pid", "cohort"]].merge(cohort, on="pid", how="left",
                                              validate="one_to_one")
    if df[["split", "tum_vol", "mask_frac_in_crop"]].isna().any().any():
        raise ValueError("pazienti senza split, tum_vol o mask_frac_in_crop in cohort.csv")
    df["crop_flag"] = df["mask_frac_in_crop"] < CROP_THR
    return df.sort_values("pid").reset_index(drop=True)


def build_table(df, k, mask):
    rows = []
    for _, r in df.iterrows():
        z = np.load(OUT / f"habitats_k{k}" / f"{r['pid']}.npz")
        if int(z["k"]) != k:
            raise ValueError(f"{r['pid']}: k del file diverso da {k}")
        hab = z["habitat"]
        if mask == "eroded2d":
            hab = np.where(z["eroded2d"], hab, 0).astype(hab.dtype)
        row = r.to_dict()
        row.update(patient_features(hab, z["spacing_zyx"], k))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def med_iqr(s):
    s = s.dropna()
    return f"{s.median():.3g} [{s.quantile(.25):.3g}-{s.quantile(.75):.3g}]"


def describe(tab, k):
    cols = (["tum_vol", "vol_valid_cm3"] + [f"frac_h{h}" for h in range(1, k + 1)]
            + ["entropy", "adj"] + [c for c in tab if c.startswith("adj_h")]
            + [f"frag_h{h}" for h in range(1, k + 1)] + [f"nreg_h{h}" for h in range(1, k + 1)])
    out = {s: [med_iqr(g[c]) for c in cols] for s, g in tab.groupby("split")}
    out["NaN"] = [int(tab[c].isna().sum()) for c in cols]
    return pd.DataFrame(out, index=cols)


def report_analysis(name, k, mask, tab, cand, kept, dropped, rho_train):
    print("=" * 74)
    print(f"ANALISI: {name}  (k = {k}, maschera {mask})")
    print("=" * 74)
    print(f"Pazienti: {len(tab)}  " + "  ".join(
        f"{s}: {n}" for s, n in tab["split"].value_counts().sort_index().items()))
    print("\nFeature per paziente: mediana [p25-p75] per split; ultima colonna = NaN")
    print(describe(tab, k).to_string())
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
        print("\nSpearman sul training tra le feature candidate per i modelli")
        print(rho_train.to_string())
    print(f"\nFiltro |rho| > {CORR_THR} (ordine di priorita': {', '.join(cand)})")
    print("  tenute:   " + ", ".join(kept))
    print("  scartate: " + ("; ".join(f"{d} (rho {r:+.2f} con {kc})" for d, kc, r in dropped)
                            if dropped else "nessuna"))
    print("\nCorrelazione di Spearman con tum_vol (tutte le pazienti): " + ", ".join(
        f"{c} {tab[c].corr(tab['tum_vol'], method='spearman'):+.2f}" for c in cand[1:]))
    nr = [f"nreg_h{h}" for h in range(1, k + 1)]
    print("Numero di regioni >= 50 mm^3 (descrittivo) con tum_vol: " + ", ".join(
        f"{c} {tab[c].corr(tab['tum_vol'], method='spearman'):+.2f}" for c in nr))
    print()


def report_sensitivity(main, er, cols):
    print("=" * 74)
    print("CONFRONTO: maschera erosa contro maschera intera (k = 3, stesse pazienti)")
    print("=" * 74)
    m = main.set_index("pid")
    e = er.set_index("pid").loc[m.index]
    print(f"Voxel validi rimasti dopo l'erosione: {e['n_valid'].sum() / m['n_valid'].sum():.1%};"
          f" pazienti con < 50%: {(e['n_valid'] < 0.5 * m['n_valid']).sum()};"
          f" minimo per paziente: {int(e['n_valid'].min())} voxel")
    rows = []
    for c in cols:
        rows.append({"feature": c,
                     "rho_intera_erosa": m[c].corr(e[c], method="spearman"),
                     "mediana_intera": m[c].median(), "mediana_erosa": e[c].median(),
                     "diff_mediana_paziente": (e[c] - m[c]).median()})
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(pd.DataFrame(rows).to_string(index=False))
    print()


# --------------------------------------------------------------------------
def main():
    df = load_patients()
    tables, sel_rows, kept_main = {}, [], None
    buf = io.StringIO()
    for name, fname, k, mask, mode in ANALYSES:
        print(f"Analisi {name}: calcolo...", flush=True)
        tab = build_table(df, k, mask)
        cand = model_candidates(k)
        train = tab[tab["split"] == "train"]
        rho_train = train[cand].corr(method="spearman")
        if mode == "filtro":
            kept, dropped = corr_filter(train, cand)
        else:                                   # stesso elenco dell'analisi principale
            kept, dropped = list(kept_main), []
        if name == "principale":
            kept_main = kept
        tab.to_csv(OUT / fname, index=False)
        tables[name] = tab
        drop_map = {d: (kc, r) for d, kc, r in dropped}
        for c in cand:
            sel_rows.append({"analisi": name, "k": k, "maschera": mask, "feature": c,
                             "nei_modelli": c in kept,
                             "scartata_per": drop_map.get(c, ("", np.nan))[0],
                             "rho": drop_map.get(c, ("", np.nan))[1],
                             "filtro": "training" if mode == "filtro" else "da principale"})
        with redirect_stdout(buf):
            report_analysis(name, k, mask, tab, cand, kept, dropped, rho_train)

    with redirect_stdout(buf):
        report_sensitivity(tables["principale"], tables["eroded2d"],
                           model_candidates(K_MAIN)[1:])
        print("=" * 74)
        print(f"TUMORI TAGLIATI DAL RITAGLIO (crop_flag = mask_frac_in_crop < {CROP_THR}),")
        print("da escludere nell'analisi di sensibilita' della Fase 5")
        print("=" * 74)
        fl = tables["principale"].loc[tables["principale"]["crop_flag"]]
        for _, r in fl.iterrows():
            print(f"  {r['pid']}  ({r['split']})  mask_frac_in_crop = {r['mask_frac_in_crop']:.4f}")
        print(f"  totale: {len(fl)}")

    # pazienti escluse in Fase 2 (problemi di acquisizione): solo per la sensibilita' in Fase 5
    exc = build_table(load_patients(excluded=True), K_MAIN, "intera")
    exc.to_csv(OUT / "features_qc_excluded.csv", index=False)
    with redirect_stdout(buf):
        print("\n" + "=" * 74)
        print("PAZIENTI ESCLUSE IN FASE 2 (solo sensibilita'; k = 3, maschera intera)")
        print("=" * 74)
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print(exc[["pid", "split", "n_valid"] + model_candidates(K_MAIN)].to_string(index=False))

    pd.DataFrame(sel_rows).to_csv(OUT / "features_selection.csv", index=False)
    head = ("FASE 4 - BIOMARCATORI DI ETEROGENEITA' (I-SPY1, pazienti con exclude == False)\n"
            f"Regioni: 26 vicini, conteggio solo >= {MIN_REGION_MM3:g} mm^3; frammentazione = "
            "1 - frazione nella regione piu' grande;\nadiacenza = quota della superficie di "
            "contatto (facce pesate in mm^2) tra habitat diversi; entropia / log k.\n"
            f"Filtro: Spearman sul training, |rho| > {CORR_THR}. La pCR non viene letta.\n\n")
    (OUT / "features_report.txt").write_text(head + buf.getvalue(), encoding="utf-8")
    print(head + buf.getvalue())
    print("Salvati: outputs/features.csv, outputs/features_eroded2d.csv, outputs/features_k2.csv,"
          " outputs/features_qc_excluded.csv, outputs/features_selection.csv,"
          " outputs/features_report.txt")


if __name__ == "__main__":
    main()
