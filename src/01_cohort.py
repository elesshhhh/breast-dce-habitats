"""
Fase 1 - Definizione della coorte e Tabella 1.

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/01_cohort.py

Cosa fa:
  1. applica i criteri di inclusione e salva outputs/cohort.csv
     (con colonna `split` esplicita: train / val / test);
  2. produce la Tabella 1 (totale, per coorte, per split)
     -> outputs/table1.csv e outputs/table1.md;
  3. calcola il tasso di pCR per sottotipo e coorte -> outputs/pcr_by_subtype.csv;
  4. verifica sui pazienti demo I-SPY che il PE minimo dentro la maschera sia ~0,7
     e che tum_vol sia in cm^3 -> outputs/demo_mask_check.csv;
  5. salva il riepilogo (diagramma di flusso della coorte e controlli)
     in outputs/cohort_report.txt.

Decisioni (concordate):
  - coorti I-SPY1 + I-SPY2 (Duke esclusa: solo bounding box);
  - esclusi i pazienti senza HER2 (5 in I-SPY1);
  - esclusi i pazienti con fase early e late sulla stessa acquisizione
    (ISPY1_1139: SER e variazione tardiva non calcolabili);
  - eta' mancante (3 pazienti I-SPY2): `age` resta originale, `age_imputed`
    usa la mediana del training set, `age_was_imputed` segnala i casi;
  - menopausa: solo descrittiva (Tabella 1), non entra nei modelli;
  - sottotipo a 3 categorie (principale, come HR_HER2_STATUS) e a 4 categorie;
  - Tabella 1: mediana [IQR] per le continue, n (%) per le categoriche,
    mancanti come riga esplicita, nessun p-value.
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CSV = DATA / "BreastDCEDL_metadata_min_crop.csv"
DEMO = DATA / "demo" / "BreastDCEDL_demo_data"
COHORT_DIRS = {"spy1": DATA / "ispy1", "spy2": DATA / "ispy2"}
OUT = ROOT / "outputs"

SPLIT_MAP = {0: "train", 1: "test", 2: "val"}          # come nel paper Scientific Data
COHORT_LABEL = {"spy1": "I-SPY1", "spy2": "I-SPY2"}
SUBTYPE3_LABEL = {"HRposHER2neg": "HR+/HER2-", "HER2pos": "HER2+", "TripleNeg": "HR-/HER2- (TN)"}
SUBTYPE3_ORDER = ["HR+/HER2-", "HER2+", "HR-/HER2- (TN)"]
SUBTYPE4_ORDER = ["HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "HR-/HER2- (TN)"]
PE_THRESHOLD = 0.70                                   # soglia FTV nominale del trial


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------
# 0. Confronto con i conteggi pubblicati (Fridman et al. 2026, Scientific Data)
# --------------------------------------------------------------------------
PAPER = {
    "Pazienti totali": 2070, "I-SPY1": 172, "I-SPY2": 982, "Duke": 916,
    "pCR disponibile": 1452, "Split train": 1532, "Split val": 268, "Split test": 270,
}


def compare_with_paper(df):
    section("0. CONFRONTO CON I CONTEGGI DEL PAPER (dataset completo, Duke inclusa)")
    split = df["test"].map(SPLIT_MAP)
    ours = {
        "Pazienti totali": len(df),
        "I-SPY1": (df["dataset"] == "spy1").sum(), "I-SPY2": (df["dataset"] == "spy2").sum(),
        "Duke": (df["dataset"] == "duke").sum(), "pCR disponibile": df["pCR"].notna().sum(),
        "Split train": (split == "train").sum(), "Split val": (split == "val").sum(),
        "Split test": (split == "test").sum(),
    }
    for k, v in PAPER.items():
        print(f"{k:20s} paper = {v:5d}   CSV = {ours[k]:5d}   {'OK' if v == ours[k] else 'DIVERSO'}")
    print("\nTasso di pCR per split (pazienti con pCR, tutte le coorte; paper: ~29-30%):")
    has = df["pCR"].notna()
    print(df[has].groupby(split[has])["pCR"].agg(n="size", tasso_pCR="mean")
          .reindex(["train", "val", "test"]).rename_axis("split").round(3).to_string())


# --------------------------------------------------------------------------
# 1. Coorte
# --------------------------------------------------------------------------
def build_cohort(df):
    section("1. CRITERI DI INCLUSIONE")
    steps = []

    d = df.copy()
    steps.append(("Tutti i pazienti BreastDCEDL", d))

    d = d[d["dataset"].isin(["spy1", "spy2"])]
    steps.append(("Coorti I-SPY1 + I-SPY2 (Duke esclusa)", d))

    d = d[d["pCR"].notna()]
    steps.append(("pCR disponibile", d))

    excl_her2 = d.loc[d["HER2"].isna(), "pid"].tolist()
    d = d[d["HER2"].notna()]
    steps.append(("HER2 disponibile", d))

    excl_phase = d.loc[d["post_early"] == d["post_late"], "pid"].tolist()
    d = d[d["post_early"] != d["post_late"]]
    steps.append(("Fasi early e late distinte", d))

    prev = None
    for label, s in steps:
        n_by = s["dataset"].value_counts()
        extra = f"  (I-SPY1 {n_by.get('spy1', 0)}, I-SPY2 {n_by.get('spy2', 0)})" \
            if label != steps[0][0] else ""
        removed = f"  [-{prev - len(s)}]" if prev is not None else ""
        print(f"{label:42s} n = {len(s):5d}{removed}{extra}")
        prev = len(s)
    print(f"\nEsclusi per HER2 mancante: {', '.join(excl_her2)}")
    print(f"Esclusi per early = late: {', '.join(excl_phase)}")

    # controlli: le variabili cliniche di base devono essere complete (tranne l'eta')
    assert d["HR"].notna().all() and d["HR_HER2_STATUS"].notna().all()

    d = d.copy()
    d["cohort"] = d["dataset"].map(COHORT_LABEL)
    d["split"] = d["test"].astype(int).map(SPLIT_MAP)
    assert d["split"].notna().all(), "valori inattesi nella colonna test"

    d["subtype3"] = d["HR_HER2_STATUS"].map(SUBTYPE3_LABEL)
    hr = d["HR"].astype(int)
    her2 = d["HER2"].astype(int)
    d["subtype4"] = np.select(
        [(hr == 1) & (her2 == 0), (hr == 1) & (her2 == 1),
         (hr == 0) & (her2 == 1), (hr == 0) & (her2 == 0)],
        SUBTYPE4_ORDER, default="?")
    # coerenza tra HR/HER2 e HR_HER2_STATUS
    check = d["subtype4"].replace({"HR+/HER2+": "HER2+", "HR-/HER2+": "HER2+"})
    assert (check == d["subtype3"]).all(), "HR/HER2 non coerenti con HR_HER2_STATUS"

    # eta': mediana del solo training set (niente informazione da val/test)
    age_median_train = d.loc[d["split"] == "train", "age"].median()
    d["age_was_imputed"] = d["age"].isna().astype(int)
    d["age_imputed"] = d["age"].fillna(age_median_train)
    print(f"\nEta' mancante: {d['age'].isna().sum()} pazienti "
          f"({', '.join(d.loc[d['age'].isna(), 'pid'])}); "
          f"imputata con la mediana del training = {age_median_train:.2f} anni")

    print("\nPazienti per coorte e split:")
    print(pd.crosstab(d["cohort"], d["split"], margins=True, margins_name="Totale")
          [["train", "val", "test", "Totale"]].to_string())

    front = ["pid", "cohort", "split", "pCR", "age", "age_imputed", "age_was_imputed",
             "HR", "HER2", "subtype3", "subtype4", "menopause", "tum_vol"]
    d = d[front + [c for c in d.columns if c not in front]]
    return d.reset_index(drop=True), age_median_train


# --------------------------------------------------------------------------
# 2. Tabella 1
# --------------------------------------------------------------------------
def fmt_median_iqr(x):
    x = x.dropna()
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return f"{med:.1f} [{q1:.1f}-{q3:.1f}]"


def fmt_n_pct(k, n):
    return f"{k} ({100 * k / n:.1f}%)" if n else "0"


def table1_column(s):
    n = len(s)
    rows = [("N", "", str(n))]
    rows.append(("Eta' (anni), mediana [IQR]", "", fmt_median_iqr(s["age"])))
    rows.append(("Eta' (anni)", "Mancante", fmt_n_pct(s["age"].isna().sum(), n)))
    for var, lab in [("HR", "HR"), ("HER2", "HER2")]:
        rows.append((lab, "Positivo", fmt_n_pct((s[var] == 1).sum(), n)))
        rows.append((lab, "Negativo", fmt_n_pct((s[var] == 0).sum(), n)))
    for lev in SUBTYPE3_ORDER:
        rows.append(("Sottotipo (3 categorie)", lev, fmt_n_pct((s["subtype3"] == lev).sum(), n)))
    for lev in SUBTYPE4_ORDER:
        rows.append(("Sottotipo (4 categorie)", lev, fmt_n_pct((s["subtype4"] == lev).sum(), n)))
    rows.append(("Menopausa", "Pre (0)", fmt_n_pct((s["menopause"] == 0).sum(), n)))
    rows.append(("Menopausa", "Post (1)", fmt_n_pct((s["menopause"] == 1).sum(), n)))
    rows.append(("Menopausa", "Mancante", fmt_n_pct(s["menopause"].isna().sum(), n)))
    rows.append(("Volume tumorale (cm^3), mediana [IQR]", "", fmt_median_iqr(s["tum_vol"])))
    rows.append(("pCR", "Si'", fmt_n_pct((s["pCR"] == 1).sum(), n)))
    rows.append(("pCR", "No", fmt_n_pct((s["pCR"] == 0).sum(), n)))
    return rows


def build_table1(c):
    section("2. TABELLA 1")
    groups = [("Totale", c)]
    groups += [(lab, c[c["cohort"] == lab]) for lab in ["I-SPY1", "I-SPY2"]]
    groups += [(sp.capitalize(), c[c["split"] == sp]) for sp in ["train", "val", "test"]]

    cols = {}
    index = None
    for name, s in groups:
        rows = table1_column(s)
        index = [(r[0], r[1]) for r in rows]
        cols[name] = [r[2] for r in rows]
    t1 = pd.DataFrame(cols, index=pd.MultiIndex.from_tuples(index, names=["Variabile", "Livello"]))
    print(t1.to_string())
    return t1


def table1_markdown(t1):
    header = "| Variabile | Livello | " + " | ".join(t1.columns) + " |"
    sep = "|---|---|" + "---|" * len(t1.columns)
    lines = ["# Tabella 1 - Caratteristiche della coorte", "",
             "Coorte I-SPY1 + I-SPY2 (BreastDCEDL MinCrop). Variabili continue: mediana [IQR]; "
             "categoriche: n (%). Nessun test statistico. Eta' riportata senza imputazione. "
             "Menopausa: solo descrittiva (0 = pre, 1 = post, codifica del CSV).", "",
             header, sep]
    for (var, lev), row in t1.iterrows():
        lines.append(f"| {var} | {lev} | " + " | ".join(row.values) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 3. Tasso di pCR per sottotipo e coorte
# --------------------------------------------------------------------------
def pcr_by_subtype(c):
    section("3. TASSO DI pCR PER SOTTOTIPO E COORTE")
    records = []
    for scheme, col, order in [("3 categorie", "subtype3", SUBTYPE3_ORDER),
                               ("4 categorie", "subtype4", SUBTYPE4_ORDER)]:
        for coh in ["I-SPY1", "I-SPY2", "Totale"]:
            s = c if coh == "Totale" else c[c["cohort"] == coh]
            for lev in order + ["Tutti"]:
                g = s if lev == "Tutti" else s[s[col] == lev]
                n, k = len(g), int(g["pCR"].sum())
                records.append({"schema": scheme, "sottotipo": lev, "coorte": coh,
                                "n": n, "n_pCR": k, "tasso_pCR": round(k / n, 3) if n else np.nan})
    res = pd.DataFrame(records)
    for scheme in ["3 categorie", "4 categorie"]:
        r = res[res["schema"] == scheme].copy()
        r["cella"] = r.apply(lambda x: f"{x.n_pCR}/{x.n} ({100 * x.tasso_pCR:.1f}%)", axis=1)
        print(f"\nSottotipo, {scheme}:")
        print(r.pivot(index="sottotipo", columns="coorte", values="cella")
              .reindex([*(SUBTYPE3_ORDER if scheme == "3 categorie" else SUBTYPE4_ORDER), "Tutti"])
              [["I-SPY1", "I-SPY2", "Totale"]].to_string())
    return res


# --------------------------------------------------------------------------
# 4. Controlli su maschere e volumi
# --------------------------------------------------------------------------
def phase_file(pid, dataset, n):
    return DEMO / "dce" / f"{pid}_{dataset}_vis1_dce_aqc_{int(n)}.nii.gz"


def check_demo_masks(c, df):
    section("4. CONTROLLO MASCHERE SUI PAZIENTI DEMO I-SPY")
    print(f"PE = (early - pre) / pre nei voxel della maschera; soglia FTV nominale = {PE_THRESHOLD}")
    print("Volume maschera = n voxel x slice_thick x xy_spacing^2 / 1000, confrontato con tum_vol\n")
    records = []
    for mpath in sorted((DEMO / "mask").glob("*_spy[12]_vis1_mask.nii.gz")):
        pid, dataset = mpath.name.split("_vis1_mask")[0].rsplit("_", 1)
        r = df[df["pid"] == pid].iloc[0]
        mask = nib.load(mpath).get_fdata() > 0
        vol_cm3 = mask.sum() * r["slice_thick"] * r["xy_spacing"] ** 2 / 1000
        rec = {"pid": pid, "cohort": COHORT_LABEL[dataset], "in_cohort": pid in set(c["pid"]),
               "fasi_DCE": "si'", "n_voxel": int(mask.sum()), "n_pre_le_0": np.nan,
               "PE_min": np.nan, "PE_p01": np.nan, "PE_p05": np.nan, "PE_mediana": np.nan,
               "frac_PE_lt_0.70": np.nan}

        f_pre, f_early = phase_file(pid, dataset, r["pre"]), phase_file(pid, dataset, r["post_early"])
        if f_pre.exists() and f_early.exists():
            pre, early = nib.load(f_pre).get_fdata(), nib.load(f_early).get_fdata()
            assert pre.shape == early.shape == mask.shape
            p, e = pre[mask], early[mask]
            valid = p > 0
            pe = (e[valid] - p[valid]) / p[valid]
            rec.update({"n_pre_le_0": int((~valid).sum()), "PE_min": pe.min(),
                        "PE_p01": np.percentile(pe, 1), "PE_p05": np.percentile(pe, 5),
                        "PE_mediana": np.median(pe), "frac_PE_lt_0.70": (pe < PE_THRESHOLD).mean()})
        else:
            rec["fasi_DCE"] = "assenti nella demo"
        rec.update({"vol_maschera_cm3": vol_cm3, "tum_vol": r["tum_vol"],
                    "rapporto_vol": vol_cm3 / r["tum_vol"]})
        records.append(rec)
    res = pd.DataFrame(records)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
        print(res.to_string(index=False))
    ok_vol = np.allclose(res["rapporto_vol"], 1, atol=1e-3)
    print(f"\ntum_vol coincide con il volume della maschera in cm^3: {'SI' if ok_vol else 'NO'}")
    return res


def check_full_masks(c):
    """Controlla maschere e 3 fasi sulle coorti scaricate e calcola mask_frac_in_crop.

    mask_frac_in_crop = volume della maschera nel ritaglio MinCrop / tum_vol (arrotondato a
    4 decimali). < 1 = tumore tagliato in z dal ritaglio; NaN = coorte non ancora scaricata.
    """
    section("5. FILE MASCHERA E FASI (coorti complete)")
    frac = pd.Series(np.nan, index=c.index, name="mask_frac_in_crop")
    for dataset, folder in COHORT_DIRS.items():
        lab = COHORT_LABEL[dataset]
        sub = c[c["cohort"] == lab]
        if not folder.exists():
            print(f"{lab}: cartella {folder.relative_to(ROOT)} assente -> controllo non eseguito "
                  f"({len(sub)} pazienti NON verificati, mask_frac_in_crop vuota)")
            continue
        masks = {p.name: p for p in folder.rglob("*_vis1_mask.nii.gz")}
        phases = {p.name for p in folder.rglob("*_vis1_dce_aqc_*.nii.gz")}
        no_mask, no_phase, bad_shape = [], [], []
        for i, r in sub.iterrows():
            mname = f"{r['pid']}_{dataset}_vis1_mask.nii.gz"
            if mname not in masks:
                no_mask.append(r["pid"])
                continue
            if any(f"{r['pid']}_{dataset}_vis1_dce_aqc_{int(r[p])}.nii.gz" not in phases
                   for p in ["pre", "post_early", "post_late"]):
                no_phase.append(r["pid"])
            img = nib.load(masks[mname])
            if img.shape[1:] != (256, 256):
                bad_shape.append(r["pid"])
            vol = (np.asarray(img.dataobj) > 0).sum() * r["slice_thick"] * r["xy_spacing"] ** 2 / 1000
            frac[i] = round(vol / r["tum_vol"], 4)
        print(f"{lab}: {len(sub) - len(no_mask)}/{len(sub)} maschere trovate"
              + (f"; mancanti: {', '.join(no_mask)}" if no_mask else ""))
        print(f"{lab}: pazienti con fasi mancanti: {', '.join(no_phase) or 'nessuno'}; "
              f"shape xy diversa da 256x256: {', '.join(bad_shape) or 'nessuno'}")
        cut = frac[sub.index][frac[sub.index] < 1]
        print(f"{lab}: tumore tagliato dal ritaglio (mask_frac_in_crop < 1): {len(cut)} pazienti"
              + "".join(f"\n    {c.loc[i, 'pid']}: {v:.4f}" for i, v in cut.items()))
    return frac


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(CSV)

    buf = io.StringIO()
    with redirect_stdout(buf):
        compare_with_paper(df)
        cohort, _ = build_cohort(df)
        t1 = build_table1(cohort)
        pcr = pcr_by_subtype(cohort)
        demo = check_demo_masks(cohort, df)
        cohort.insert(cohort.columns.get_loc("tum_vol") + 1, "mask_frac_in_crop",
                      check_full_masks(cohort))

    cohort.to_csv(OUT / "cohort.csv", index=False)
    t1.to_csv(OUT / "table1.csv")
    (OUT / "table1.md").write_text(table1_markdown(t1), encoding="utf-8")
    pcr.to_csv(OUT / "pcr_by_subtype.csv", index=False)
    demo.to_csv(OUT / "demo_mask_check.csv", index=False)
    (OUT / "cohort_report.txt").write_text(buf.getvalue(), encoding="utf-8")

    print(buf.getvalue())
    print("\nFile salvati in outputs/: cohort.csv, table1.csv, table1.md, "
          "pcr_by_subtype.csv, demo_mask_check.csv, cohort_report.txt")


if __name__ == "__main__":
    main()
