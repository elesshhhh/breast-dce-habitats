"""
Fase 2 - Mappe di enhancement voxel per voxel: PE, SER e variazione tardiva.

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/02_maps.py                              # tutte le coorti scaricate
    python src/02_maps.py --pids ISPY1_1001 ISPY1_1002 # prova su pochi pazienti
    python src/02_maps.py --no-figures                 # senza figure di controllo

Cosa fa, per ogni paziente di outputs/cohort.csv con i file presenti:
  1. carica le fasi pre, early e late (acquisizioni indicate dal CSV) e la maschera;
  2. dentro la maschera calcola voxel per voxel
        PE                  = (early - pre) / pre
        SER                 = (early - pre) / (late - pre)
        variazione tardiva  = (late - early) / early
  3. salva outputs/maps/<pid>.npz, ritagliato sul riquadro (bounding box) della maschera;
  4. raccoglie i controlli in outputs/maps_qc.csv e outputs/maps_report.txt;
  5. disegna le figure di controllo in outputs/figures/maps_qc/<pid>.png
     (pazienti segnalate + 6 pazienti di dimensioni diverse). Le figure non mostrano la pCR.

Decisioni (concordate, vedi docs/roadmap.md, Fase 2):
  - voxel VALIDI = maschera & PE >= 0,70 (definizione FTV del trial, Hylton 2012):
    lo stesso tipo di tessuto per tutte le pazienti;
  - le tre mappe sono calcolate su tutta la maschera (anche dove PE < 0,70), cosi' un'analisi
    di sensibilita' "tutta la maschera" non richiede di ricalcolarle; la Fase 3 usa `valid`;
  - dove un rapporto non e' calcolabile il valore e' NaN (mai 0 o un valore di comodo):
    PE e variazione tardiva se pre <= 0, SER se late - pre <= 0. Un voxel con SER NaN
    resta valido per PE e variazione tardiva; i casi sono contati nel report;
  - nessun taglio dei valori estremi: si decide in Fase 3, insieme al clustering;
  - ESCLUSIONE per problemi di acquisizione (colonna `exclude`), regola oggettiva decisa
    dopo il controllo visivo e senza guardare la pCR. Una paziente e' esclusa se
      * il rapporto early/pre o late/pre dell'intensita' mediana del tessuto FUORI dal tumore
        esce da [0,9; 2,5] (un'intera fase piu' chiara o piu' scura: guadagno o serie
        sbagliata; nella coorte I-SPY1 l'1-99 percentile e' 0,93-1,97), oppure
      * meno del 50% dei voxel della maschera e' valido.
    Le mappe delle escluse vengono comunque salvate (analisi di sensibilita');
  - colonna `flag` (solo informativa): > 10% della maschera con PE < 0,70 o late <= pre;
  - spacing preso dal CSV (slice_thick, xy_spacing): l'header NIfTI dichiara 1x1x1 mm;
  - le immagini hanno assi (z, y, x);
  - la tolleranza +-5% sulla variazione tardiva serve solo per le frazioni descrittive
    (wash-out / plateau / crescita continua), non per escludere voxel.
"""

import argparse
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COHORT_DIRS = {"spy1": DATA / "ispy1", "spy2": DATA / "ispy2"}
OUT = ROOT / "outputs"
MAPS_DIR = OUT / "maps"
FIG_DIR = OUT / "figures" / "maps_qc"

PE_MIN = 0.70        # soglia FTV nominale del trial
PE_EPS = 1e-9        # tolleranza numerica sul confronto PE >= PE_MIN
LATE_TOL = 0.05      # +-5%: plateau (solo descrittivo)
FLAG_FRAC = 0.10     # flag se > 10% dei voxel della maschera ha PE < 0,70 oppure late <= pre
BG_DILATE = 5        # voxel di margine attorno al tumore esclusi dal "fondo"
BG_RANGE = (0.9, 2.5)  # rapporti early/pre e late/pre del fondo accettati (estremi inclusi)
MIN_FRAC_VALID = 0.5   # frazione minima di voxel validi nella maschera
N_FIG_EXTRA = 6      # pazienti non segnalate nelle figure di controllo

# motivo di ogni voxel (mappa `reason`)
REASON_OUTSIDE = 0   # fuori dalla maschera
REASON_VALID = 1     # valido
REASON_PE_LOW = 2    # nella maschera ma PE < 0,70
REASON_PRE_LE0 = 3   # nella maschera ma pre <= 0 (rapporti non calcolabili)
REASON_LABELS = {REASON_OUTSIDE: "fuori maschera", REASON_VALID: "valido",
                 REASON_PE_LOW: "PE < 0,70", REASON_PRE_LE0: "pre <= 0"}


# --------------------------------------------------------------------------
# Calcolo (funzioni pure, testate in tests/test_02_maps.py)
# --------------------------------------------------------------------------
def compute_maps(pre, early, late, mask):
    """Mappe PE, SER, variazione tardiva, voxel validi e motivo, sulla shape di ingresso.

    I calcoli sono in float64; fuori dalla maschera tutte le mappe sono NaN.
    """
    pre, early, late = (np.asarray(x, dtype=np.float64) for x in (pre, early, late))
    mask = np.asarray(mask, dtype=bool)
    defined = mask & (pre > 0)
    d_early, d_late = early - pre, late - pre
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = np.where(defined, d_early / pre, np.nan)
        ser = np.where(defined & (d_late > 0), d_early / d_late, np.nan)
        late_change = np.where(defined & (early > 0), (late - early) / early, np.nan)

    valid = defined & (pe >= PE_MIN - PE_EPS)
    reason = np.full(pre.shape, REASON_OUTSIDE, dtype=np.uint8)
    reason[mask & ~defined] = REASON_PRE_LE0
    reason[defined & ~valid] = REASON_PE_LOW
    reason[valid] = REASON_VALID
    return {"pe": pe, "ser": ser, "late": late_change, "valid": valid, "reason": reason}


def background_ratios(pre, early, late, mask, dilate=BG_DILATE):
    """Rapporti early/pre e late/pre dell'intensita' mediana del tessuto fuori dal tumore.

    Fondo = voxel a piu' di `dilate` voxel dalla maschera, con pre sopra la mediana del
    volume (cosi' si esclude l'aria). Se l'intera fase cambia luminosita' (guadagno diverso,
    serie sbagliata) il rapporto si allontana molto dal valore tipico (~1,3).
    """
    outside = ~ndimage.binary_dilation(mask, iterations=dilate)
    body = outside & (pre > np.median(pre))
    p = np.median(pre[body])
    return np.median(early[body]) / p, np.median(late[body]) / p


def exclusion_reasons(bg_early_pre, bg_late_pre, frac_valid):
    """Motivi di esclusione (lista vuota = paziente inclusa)."""
    lo, hi = BG_RANGE
    why = []
    for name, r in (("early/pre", bg_early_pre), ("late/pre", bg_late_pre)):
        if not lo <= r <= hi:
            why.append(f"fondo {name} = {r:.2f} fuori da [{lo}; {hi}]")
    if frac_valid < MIN_FRAC_VALID:
        why.append(f"voxel validi {frac_valid:.1%} < {MIN_FRAC_VALID:.0%}")
    return why


def mask_bbox(mask):
    """Riquadro minimo che contiene la maschera: (start, stop), stop escluso."""
    idx = np.nonzero(mask)
    return (np.array([i.min() for i in idx]), np.array([i.max() + 1 for i in idx]))


def crop(a, start, stop):
    return a[tuple(slice(s, e) for s, e in zip(start, stop))]


def curve_fractions(late, tol=LATE_TOL):
    """Frazioni di voxel wash-out (< -tol), plateau (|x| <= tol), crescita continua (> tol)."""
    x = np.asarray(late, dtype=np.float64)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"washout": np.nan, "plateau": np.nan, "persistent": np.nan}
    wo, pers = (x < -tol).mean(), (x > tol).mean()
    return {"washout": wo, "plateau": 1 - wo - pers, "persistent": pers}


# --------------------------------------------------------------------------
# File
# --------------------------------------------------------------------------
def index_files():
    """{nome file: percorso} per maschere e fasi delle coorti scaricate."""
    files = {}
    for folder in COHORT_DIRS.values():
        if folder.exists():
            files.update({p.name: p for p in folder.rglob("*_vis1_*.nii.gz")})
    return files


def patient_files(row, files):
    """Percorsi di pre, early, late e maschera; None se manca qualcosa."""
    pid, ds = row["pid"], row["dataset"]
    names = {k: f"{pid}_{ds}_vis1_dce_aqc_{int(row[c])}.nii.gz"
             for k, c in [("pre", "pre"), ("early", "post_early"), ("late", "post_late")]}
    names["mask"] = f"{pid}_{ds}_vis1_mask.nii.gz"
    if any(n not in files for n in names.values()):
        return None
    return {k: files[n] for k, n in names.items()}


def load_patient(row, files):
    paths = patient_files(row, files)
    vols = {k: nib.load(p).get_fdata() for k, p in paths.items()}
    vols["mask"] = vols["mask"] > 0
    shapes = {v.shape for v in vols.values()}
    if len(shapes) != 1:
        raise ValueError(f"{row['pid']}: shape diverse tra fasi e maschera: {shapes}")
    spacing = np.array([row["slice_thick"], row["xy_spacing"], row["xy_spacing"]])  # (z, y, x)
    return vols, spacing


def pct(a, q):
    a = a[~np.isnan(a)]
    return np.percentile(a, q) if a.size else np.full(len(q), np.nan)


# --------------------------------------------------------------------------
# Paziente
# --------------------------------------------------------------------------
def process_patient(row, files, save=True):
    """Calcola e salva le mappe di un paziente; restituisce (riga di controllo, voxel validi)."""
    vols, spacing = load_patient(row, files)
    m = compute_maps(vols["pre"], vols["early"], vols["late"], vols["mask"])
    mask, valid = vols["mask"], m["valid"]

    start, stop = mask_bbox(mask)
    if save:
        np.savez_compressed(
            MAPS_DIR / f"{row['pid']}.npz",
            pe=crop(m["pe"], start, stop).astype(np.float32),
            ser=crop(m["ser"], start, stop).astype(np.float32),
            late=crop(m["late"], start, stop).astype(np.float32),
            valid=crop(valid, start, stop),
            reason=crop(m["reason"], start, stop),
            bbox_start=start, bbox_stop=stop, full_shape=np.array(mask.shape),
            spacing_zyx=spacing, pe_min=PE_MIN, pid=np.array(row["pid"]))

    n_mask = int(mask.sum())
    n_valid = int(valid.sum())
    pre_m, early_m, late_m = (vols[k][mask] for k in ("pre", "early", "late"))
    v = {k: m[k][valid] for k in ("pe", "ser", "late")}
    frac_pe_low = 1 - n_valid / n_mask
    frac_late_le_pre = float((late_m <= pre_m).mean())
    flag_why = []
    if frac_pe_low > FLAG_FRAC:
        flag_why.append(f"PE<{PE_MIN:.2f} nel {100 * frac_pe_low:.1f}% della maschera")
    if frac_late_le_pre > FLAG_FRAC:
        flag_why.append(f"late<=pre nel {100 * frac_late_le_pre:.1f}% della maschera")

    qc = {"pid": row["pid"], "cohort": row["cohort"], "split": row["split"],
          "n_mask": n_mask, "n_valid": n_valid, "frac_valid": n_valid / n_mask,
          "n_pe_low": int((m["reason"] == REASON_PE_LOW).sum()),
          "n_pre_le0": int((m["reason"] == REASON_PRE_LE0).sum()),
          "n_ser_nan_valid": int(np.isnan(v["ser"]).sum()),
          "frac_late_le_pre": frac_late_le_pre,
          "mean_pre": pre_m.mean(), "mean_early": early_m.mean(), "mean_late": late_m.mean(),
          "pre_min": pre_m.min()}
    for name in ("pe", "ser", "late"):
        for q, x in zip((5, 50, 95), pct(v[name], [5, 50, 95])):
            qc[f"{name}_p{q:02d}"] = x
    qc.update({f"frac_{k}": x for k, x in curve_fractions(v["late"]).items()})
    bg_early_pre, bg_late_pre = background_ratios(vols["pre"], vols["early"], vols["late"], mask)
    excl_why = exclusion_reasons(bg_early_pre, bg_late_pre, n_valid / n_mask)
    qc.update({"bbox_shape": "x".join(map(str, stop - start)),
               "mask_frac_in_crop": row["mask_frac_in_crop"],
               "bg_early_pre": bg_early_pre, "bg_late_pre": bg_late_pre,
               "flag": bool(flag_why), "flag_motivo": "; ".join(flag_why),
               "exclude": bool(excl_why), "exclude_motivo": "; ".join(excl_why)})
    return qc, {k: x.astype(np.float32) for k, x in v.items()}


# --------------------------------------------------------------------------
# Figure di controllo (senza pCR)
# --------------------------------------------------------------------------
def qc_figure(row, files, qc_row):
    vols, _ = load_patient(row, files)
    m = compute_maps(vols["pre"], vols["early"], vols["late"], vols["mask"])
    mask = vols["mask"]
    z = int(mask.sum(axis=(1, 2)).argmax())                 # fetta con piu' tumore
    _, ys, xs = np.nonzero(mask)
    mg = 15
    y0, y1 = max(ys.min() - mg, 0), min(ys.max() + mg + 1, mask.shape[1])
    x0, x1 = max(xs.min() - mg, 0), min(xs.max() + mg + 1, mask.shape[2])
    sl = lambda a: a[z, y0:y1, x0:x1]

    fig, ax = plt.subplots(2, 4, figsize=(17, 8.5))
    vmax = np.percentile(vols["early"][z], 99.5)
    for a, k, t in zip(ax[0, :3], ("pre", "early", "late"),
                       (f"pre (acq {int(row['pre'])})", f"early (acq {int(row['post_early'])})",
                        f"late (acq {int(row['post_late'])})")):
        a.imshow(sl(vols[k]), cmap="gray", vmin=0, vmax=vmax)
        a.contour(sl(mask), levels=[0.5], colors="red", linewidths=0.8)
        a.set_title(t + " - contorno maschera")

    a = ax[0, 3]
    for sel, lab, st in ((mask, "tutta la maschera", "o-"), (m["valid"], "voxel validi", "s--")):
        if sel.any():
            a.plot([0, 1, 2], [vols[k][sel].mean() for k in ("pre", "early", "late")], st, label=lab)
    a.set_xticks([0, 1, 2], ["pre", "early", "late"])
    a.set_title("segnale medio")
    a.legend()

    panels = [("pe", "PE", "viridis", 0, 3), ("ser", "SER", "coolwarm", 0, 2),
              ("late", "variazione tardiva", "coolwarm_r", -0.5, 0.5)]
    for a, (k, t, cm, lo, hi) in zip(ax[1, :3], panels):
        a.imshow(sl(vols["early"]), cmap="gray", vmin=0, vmax=vmax)
        im = a.imshow(sl(np.where(m["valid"], m[k], np.nan)), cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_title(f"{t} (voxel validi)")
    a = ax[1, 3]
    cmap = ListedColormap(["black", "limegreen", "orange", "magenta"])
    im = a.imshow(sl(m["reason"]), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    cb = fig.colorbar(im, ax=a, fraction=0.046, ticks=list(REASON_LABELS))
    cb.ax.set_yticklabels(list(REASON_LABELS.values()))
    a.set_title("motivo")
    for a in ax.flat:
        if a is not ax[0, 3]:          # il grafico del segnale medio tiene gli assi
            a.axis("off")

    flag = (f" - ESCLUSA: {qc_row['exclude_motivo']}" if qc_row["exclude"]
            else f" - FLAG: {qc_row['flag_motivo']}" if qc_row["flag"] else "")
    fig.suptitle(f"{row['pid']}  ({row['subtype3']}, fetta z={z}, {qc_row['n_valid']}/"
                 f"{qc_row['n_mask']} voxel validi){flag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{row['pid']}.png", dpi=90)
    plt.close(fig)


def pick_figure_pids(qc):
    """Pazienti escluse o segnalate + N_FIG_EXTRA altre a quantili diversi di n_valid."""
    odd = qc["flag"] | qc["exclude"]
    flagged = qc.loc[odd, "pid"].tolist()
    rest = qc[~odd].sort_values("n_valid").reset_index(drop=True)
    if rest.empty:
        return flagged
    idx = np.unique(np.round(np.linspace(0.1, 0.9, N_FIG_EXTRA) * (len(rest) - 1)).astype(int))
    return flagged + rest.loc[idx, "pid"].tolist()


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def report(qc, pooled, skipped):
    print("=" * 70)
    print("FASE 2 - MAPPE DI ENHANCEMENT")
    print("=" * 70)
    print(f"Parametri: PE_MIN = {PE_MIN}, tolleranza variazione tardiva = +-{LATE_TOL:.0%} "
          f"(solo descrittiva), flag se > {FLAG_FRAC:.0%} della maschera ha PE < {PE_MIN} "
          "oppure late <= pre")
    print(f"Esclusione: rapporto del fondo early/pre o late/pre fuori da {list(BG_RANGE)} "
          f"(fondo = tessuto a > {BG_DILATE} voxel dal tumore con pre sopra la mediana), "
          f"oppure voxel validi < {MIN_FRAC_VALID:.0%}")
    print(f"\nPazienti elaborati: {len(qc)}  "
          + "  ".join(f"{k}: {v}" for k, v in qc["cohort"].value_counts().sort_index().items()))
    if skipped:
        print(f"Pazienti saltati (file assenti): {len(skipped)}"
              + (f" -> {', '.join(skipped[:10])}{' ...' if len(skipped) > 10 else ''}"))

    ex = qc[qc["exclude"]]
    print(f"\nPazienti ESCLUSE (problemi di acquisizione): {len(ex)} -> incluse {len(qc) - len(ex)}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250,
                           "display.max_colwidth", 100):
        print(ex[["pid", "n_mask", "frac_valid", "bg_early_pre", "bg_late_pre", "exclude_motivo"]]
              .to_string(index=False) if len(ex) else "  nessuna")
    print("Rapporti del fondo nelle incluse: mediana [min - max]  "
          + "  ".join(f"{c}: {qc.loc[~qc['exclude'], c].median():.2f} "
                      f"[{qc.loc[~qc['exclude'], c].min():.2f} - {qc.loc[~qc['exclude'], c].max():.2f}]"
                      for c in ("bg_early_pre", "bg_late_pre")))

    print("\n" + "-" * 70)
    print("Da qui in poi: SOLO le pazienti incluse")
    print("-" * 70)
    keep = ~qc["exclude"].to_numpy()
    qc = qc[keep]
    pooled = {k: [a for a, kp in zip(v, keep) if kp] for k, v in pooled.items()}

    n_mask, n_valid = qc["n_mask"].sum(), qc["n_valid"].sum()
    print("\nVoxel")
    print(f"  nella maschera:           {n_mask:>10,}")
    print(f"  validi (PE >= {PE_MIN}):     {n_valid:>10,}  ({n_valid / n_mask:.2%})")
    print(f"  esclusi, PE < {PE_MIN}:       {qc['n_pe_low'].sum():>10,}  "
          f"in {(qc['n_pe_low'] > 0).sum()} pazienti")
    print(f"  esclusi, pre <= 0:        {qc['n_pre_le0'].sum():>10,}")
    n_sn = qc["n_ser_nan_valid"].sum()
    print(f"  validi con SER NaN (late <= pre): {n_sn:,} ({n_sn / n_valid:.2%}) "
          f"in {(qc['n_ser_nan_valid'] > 0).sum()} pazienti")
    print(f"  pazienti con 100% voxel validi: {(qc['n_valid'] == qc['n_mask']).sum()}")

    q = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    print("\nDistribuzione sui voxel validi di tutte le pazienti (percentili)")
    tab = pd.DataFrame({k: pct(np.concatenate(pooled[k]), q) for k in ("pe", "ser", "late")},
                       index=[f"p{x:g}" for x in q]).T
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(tab.to_string())
    f = curve_fractions(np.concatenate(pooled["late"]))
    print(f"\nForma della curva (variazione tardiva, voxel validi, tolleranza +-{LATE_TOL:.0%}): "
          f"wash-out {f['washout']:.1%}, plateau {f['plateau']:.1%}, "
          f"crescita continua {f['persistent']:.1%}")
    print(f"SER > 1 (wash-out secondo il SER): {np.nanmean(np.concatenate(pooled['ser']) > 1):.1%} "
          "dei voxel validi con SER definito")

    print("\nPer paziente: mediana [min - max]")
    for c, lab in [("n_valid", "voxel validi"), ("frac_valid", "frazione validi"),
                   ("pe_p50", "PE mediano"), ("ser_p50", "SER mediano"),
                   ("late_p50", "variazione tardiva mediana")]:
        s = qc[c]
        print(f"  {lab:<28} {s.median():.3f} [{s.min():.3f} - {s.max():.3f}]")

    fl = qc[qc["flag"]]
    print(f"\nPazienti incluse ma segnalate (flag informativo): {len(fl)}")
    cols = ["pid", "n_mask", "n_valid", "frac_valid", "frac_late_le_pre",
            "mean_pre", "mean_early", "mean_late", "flag_motivo"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250,
                           "display.max_colwidth", 80):
        print(fl[cols].to_string(index=False) if len(fl) else "  nessuna")
    cut = qc[qc["mask_frac_in_crop"] < 1]
    print(f"\nPromemoria Fase 1 - tumore tagliato dal ritaglio (mask_frac_in_crop < 1): "
          f"{', '.join(cut['pid']) or 'nessuno'}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--pids", nargs="+", help="elabora solo questi pazienti")
    ap.add_argument("--no-figures", action="store_true", help="non disegnare le figure")
    args = ap.parse_args()

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(OUT / "cohort.csv")
    if args.pids:
        cohort = cohort[cohort["pid"].isin(args.pids)]
    files = index_files()

    rows, pooled, skipped = [], {"pe": [], "ser": [], "late": []}, []
    todo = [r for _, r in cohort.iterrows()]
    for i, r in enumerate(todo, 1):
        if patient_files(r, files) is None:
            skipped.append(r["pid"])
            continue
        qc_row, vox = process_patient(r, files)
        rows.append(qc_row)
        for k in pooled:
            pooled[k].append(vox[k])
        if i % 20 == 0:
            print(f"  {i}/{len(todo)} pazienti", flush=True)
    qc = pd.DataFrame(rows)

    buf = io.StringIO()
    with redirect_stdout(buf):
        report(qc, pooled, skipped)
    if not args.pids:                  # le prove su pochi pazienti non sovrascrivono i riepiloghi
        qc.to_csv(OUT / "maps_qc.csv", index=False)

    if not args.no_figures:
        fig_pids = pick_figure_pids(qc)
        qc_by_pid = qc.set_index("pid")
        for pid in fig_pids:
            qc_figure(cohort[cohort["pid"] == pid].iloc[0], files, qc_by_pid.loc[pid])
        buf.write(f"\nFigure di controllo (outputs/figures/maps_qc/): {', '.join(fig_pids)}\n")

    if not args.pids:
        (OUT / "maps_report.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print(f"Salvati: outputs/maps/<pid>.npz ({len(qc)} file)"
          + ("" if args.pids else ", outputs/maps_qc.csv, outputs/maps_report.txt"))


if __name__ == "__main__":
    main()
