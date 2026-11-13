"""
Fase 3 - Clustering degli habitat cinetici a livello di popolazione.

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python src/03_habitats.py            # passo 1: solo le curve per scegliere k (k = 2..6)
    python src/03_habitats.py --k 3      # passo 2: modello finale (analisi principale, k = 3)
    python src/03_habitats.py --k 2      # sensibilita': stesso procedimento con k = 2
    python src/03_habitats.py --k 3 --no-figures

Cosa fa:
  1. legge i voxel `valid` di outputs/maps/<pid>.npz (Fase 2) per le pazienti con
     exclude == False in outputs/maps_qc.csv;
  2. campiona 2.000 voxel per paziente (tutti se ne ha meno) dalle sole pazienti del TRAINING
     (colonna `split` di outputs/cohort.csv), con seme fisso;
  3. variabili: log(PE) e variazione tardiva; taglio ai percentili 0,5 e 99,5 e
     standardizzazione (media e deviazione standard), con parametri stimati SOLO sul campione
     del training e poi congelati;
  4. passo 1: per k = 2..6 calcola inerzia (gomito), silhouette, stabilita' (ARI tra modelli
     adattati su due meta' diverse delle pazienti del training) e BIC della GMM;
     passo 2: K-means finale con il k scelto, ordinamento fisso degli habitat, nomi da regola,
     GMM come confronto, assegnazione di ogni voxel valido di TUTTE le pazienti al centroide
     piu' vicino (stesso modello per train, val, test e per le escluse);
  5. salva, con <k> nel nome per non sovrascrivere le analisi con k diversi:
     outputs/habitats_k<k>/<pid>.npz, outputs/habitat_model_k<k>.npz,
     outputs/habitat_fractions_k<k>.csv, outputs/habitats_report_k<k>.txt e le figure in
     outputs/figures/habitats/k<k>/ (passo 1: outputs/habitat_k_selection.* e
     outputs/figures/habitats/k_selection_*.png).
La pCR non viene mai letta.

Decisioni (concordate, vedi docs/roadmap.md, Fase 3):
  - variabili PE e variazione tardiva: le tre mappe hanno solo due gradi di liberta' e la
    variazione tardiva e' sempre definita (il SER esplode quando late ~ pre ed e' NaN proprio
    nei voxel con il wash-out piu' forte);
  - log(PE) per la coda lunga a destra; taglio ai percentili 0,5-99,5 e z-score del training;
  - nessuna erosione della maschera nell'analisi principale; la maschera erosa di 1 voxel nel
    piano (fetta per fetta) e' salvata come `eroded2d` per l'analisi di sensibilita' (Fase 4);
  - K-means principale (n_init = 10), GMM (covarianza piena) solo come confronto;
  - k scelto guardando le curve insieme (regola indicativa: il k piu' piccolo con silhouette
    alta e ARI di stabilita' >= 0,8), mai guardando la pCR. Scelto k = 3 (analisi principale;
    ARI 0,76, le tre forme di curva di Kuhl) con k = 2 come sensibilita' (unico con ARI >= 0,8);
  - habitat numerati per PE crescente del centroide (poi variazione tardiva); 0 = voxel non
    valido o fuori maschera;
  - nome = perfusione "alta"/"bassa" rispetto al PE mediano del campione del training +
    forma della curva di Kuhl dalla variazione tardiva mediana dell'habitat (+-5%).
"""

import argparse
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("maps", ROOT / "src" / "02_maps.py")
maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maps)

OUT = ROOT / "outputs"
MAPS_DIR = OUT / "maps"
FIG_DIR = OUT / "figures" / "habitats"
K_MAIN = 3                 # k dell'analisi principale (k = 2 come sensibilita')


def out_paths(k):
    """Percorsi degli output del passo 2 per un dato k."""
    return {"hab": OUT / f"habitats_k{k}", "model": OUT / f"habitat_model_k{k}.npz",
            "fractions": OUT / f"habitat_fractions_k{k}.csv",
            "report": OUT / f"habitats_report_k{k}.txt", "fig": FIG_DIR / f"k{k}"}

FEATURES = ("log_pe", "late")
N_PER_PATIENT = 2000       # voxel campionati per paziente del training
CLIP_Q = (0.5, 99.5)       # percentili del taglio
K_RANGE = range(2, 7)
SEED = 0
N_INIT = 10                # inizializzazioni di K-means
N_SIL = 10_000             # voxel per ogni calcolo della silhouette
N_SIL_REP = 3              # ripetizioni della silhouette (media)
N_STAB = 10                # ripetizioni della stabilita' (meta' delle pazienti)
ARI_MIN = 0.8              # soglia indicativa di stabilita'
LATE_TOL = maps.LATE_TOL   # +-5%: wash-out / plateau / crescita continua (Kuhl)
N_FIG_PIDS = 6
FIG_PIDS_FIXED = ["ISPY1_1107"]   # bordo di falso wash-out visto in Fase 2


# --------------------------------------------------------------------------
# Calcolo (funzioni pure, testate in tests/test_03_habitats.py)
# --------------------------------------------------------------------------
def features(pe, late):
    """Matrice (n, 2) delle variabili del clustering: log(PE), variazione tardiva."""
    pe = np.asarray(pe, dtype=np.float64)
    late = np.asarray(late, dtype=np.float64)
    if np.any(~(pe > 0)) or np.any(np.isnan(late)):
        raise ValueError("PE deve essere > 0 e la variazione tardiva definita (voxel validi)")
    return np.column_stack([np.log(pe), late])


def sample_indices(n, n_max, rng):
    """Indici di al massimo n_max voxel su n, senza ripetizioni, ordinati."""
    if n <= n_max:
        return np.arange(n)
    return np.sort(rng.choice(n, size=n_max, replace=False))


def fit_scaler(X, q=CLIP_Q):
    """Parametri della trasformazione stimati su X: taglio ai percentili, poi media e DS."""
    lo, hi = np.percentile(X, q, axis=0)
    Xc = np.clip(X, lo, hi)
    return {"lo": lo, "hi": hi, "mean": Xc.mean(axis=0), "sd": Xc.std(axis=0)}


def apply_scaler(X, s):
    return (np.clip(X, s["lo"], s["hi"]) - s["mean"]) / s["sd"]


def fit_kmeans(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(Z)


def nearest_center(Z, centers):
    """Etichetta 0..k-1 del centroide piu' vicino (distanza euclidea)."""
    d = ((Z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1)


def order_centers(centers):
    """Permutazione che ordina i centroidi per log(PE) crescente, poi variazione tardiva."""
    return np.lexsort((centers[:, 1], centers[:, 0]))


def curve_shape(late, tol=LATE_TOL):
    if late < -tol:
        return "wash-out"
    if late > tol:
        return "crescita continua"
    return "plateau"


def habitat_names(med_pe, med_late, pe_ref, tol=LATE_TOL):
    """Nomi da regola; i nomi ripetuti ricevono un suffisso (1), (2)... in ordine di habitat."""
    base = [f"perfusione {'alta' if p > pe_ref else 'bassa'}, {curve_shape(l, tol)}"
            for p, l in zip(med_pe, med_late)]
    names = []
    for i, b in enumerate(base):
        if base.count(b) > 1:
            names.append(f"{b} ({base[:i + 1].count(b)})")
        else:
            names.append(b)
    return names


def stability(Z, groups, k, n_rep=N_STAB, seed=SEED):
    """ARI tra K-means adattati su due meta' disgiunte delle pazienti, valutati su tutto Z."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    out = []
    for r in range(n_rep):
        perm = rng.permutation(ids)
        half = np.isin(groups, perm[: len(ids) // 2])
        la = fit_kmeans(Z[half], k, seed + r).predict(Z)
        lb = fit_kmeans(Z[~half], k, seed + r).predict(Z)
        out.append(adjusted_rand_score(la, lb))
    return np.array(out)


def mean_silhouette(Z, labels, n=N_SIL, rep=N_SIL_REP, seed=SEED):
    return float(np.mean([silhouette_score(Z, labels, sample_size=min(n, len(Z)),
                                           random_state=seed + r) for r in range(rep)]))


def erode_inplane(valid):
    """Erosione di 1 voxel nel piano (y, x), fetta per fetta; i voxel sul bordo del
    ritaglio contano come bordo (fuori = non valido)."""
    st = np.zeros((3, 3, 3), dtype=bool)
    st[1] = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_erosion(np.asarray(valid, dtype=bool), structure=st,
                                  border_value=0)


def labels_to_volume(valid, labels):
    """Mappa degli habitat: 0 fuori dai voxel validi, 1..k dentro (labels sono 0..k-1)."""
    hab = np.zeros(valid.shape, dtype=np.uint8)
    hab[valid] = np.asarray(labels) + 1
    return hab


def fractions(hab, k):
    """Frazione dei voxel validi in ciascun habitat 1..k."""
    v = hab[hab > 0]
    return np.bincount(v, minlength=k + 1)[1:] / max(v.size, 1)


# --------------------------------------------------------------------------
# Dati
# --------------------------------------------------------------------------
def load_patients():
    """Tabella delle pazienti con mappe: pid, cohort, split, exclude, n_valid."""
    qc = pd.read_csv(OUT / "maps_qc.csv")
    cohort = pd.read_csv(OUT / "cohort.csv")[["pid", "split"]]
    df = qc[["pid", "cohort", "exclude", "n_valid"]].merge(cohort, on="pid", how="left",
                                                          validate="one_to_one")
    if df["split"].isna().any():
        raise ValueError("pazienti senza split in cohort.csv")
    return df.sort_values("pid").reset_index(drop=True)


def training_pids(df):
    return df.loc[(df["split"] == "train") & ~df["exclude"], "pid"].tolist()


def load_map(pid):
    z = np.load(MAPS_DIR / f"{pid}.npz")
    return {k: z[k] for k in z.files}


def training_sample(pids, n_max=N_PER_PATIENT, seed=SEED):
    """Campione bilanciato del training: (X, groups, pe, late) nelle unita' originali."""
    rng = np.random.default_rng(seed)
    X, groups, pe_all, late_all = [], [], [], []
    for g, pid in enumerate(pids):
        m = load_map(pid)
        pe, late = m["pe"][m["valid"]], m["late"][m["valid"]]
        idx = sample_indices(pe.size, n_max, rng)
        X.append(features(pe[idx], late[idx]))
        groups.append(np.full(idx.size, g))
        pe_all.append(pe[idx].astype(np.float64))
        late_all.append(late[idx].astype(np.float64))
    return (np.concatenate(X), np.concatenate(groups),
            np.concatenate(pe_all), np.concatenate(late_all))


# --------------------------------------------------------------------------
# Passo 1: scelta di k
# --------------------------------------------------------------------------
def select_k(Z, groups, pe, late):
    rows, labels = [], {}
    for k in K_RANGE:
        km = fit_kmeans(Z, k)
        perm = order_centers(km.cluster_centers_)
        rank = np.empty(k, dtype=int)
        rank[perm] = np.arange(k)
        lab = rank[km.labels_]
        labels[k] = lab
        stab = stability(Z, groups, k)
        gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": mean_silhouette(Z, lab),
                     "ari_mean": stab.mean(), "ari_min": stab.min(),
                     "gmm_bic": gmm.bic(Z),
                     "ari_kmeans_gmm": adjusted_rand_score(lab, gmm.predict(Z)),
                     "min_cluster_frac": np.bincount(lab).min() / lab.size})
        print(f"  k = {k}: fatto", flush=True)
    return pd.DataFrame(rows), labels


def describe_clusters(lab, pe, late, groups, k):
    """Per habitat: frazione del campione, PE e variazione tardiva mediani, pazienti presenti."""
    rows = []
    for h in range(k):
        s = lab == h
        rows.append({"habitat": h + 1, "frac_campione": s.mean(),
                     "pe_mediano": np.median(pe[s]), "late_mediana": np.median(late[s]),
                     "pe_p25": np.percentile(pe[s], 25), "pe_p75": np.percentile(pe[s], 75),
                     "late_p25": np.percentile(late[s], 25), "late_p75": np.percentile(late[s], 75),
                     "pazienti_con_habitat_5pct": int(sum(
                         (lab[groups == g] == h).mean() >= 0.05 for g in np.unique(groups)))})
    return pd.DataFrame(rows)


def plot_k_selection(tab, labels, pe, late, path):
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, (c, t) in zip(ax, [("inertia", "inerzia (gomito)"), ("silhouette", "silhouette"),
                              ("ari_mean", "stabilita' (ARI medio, min)"),
                              ("gmm_bic", "BIC della GMM")]):
        a.plot(tab["k"], tab[c], "o-")
        if c == "ari_mean":
            a.plot(tab["k"], tab["ari_min"], "s--", alpha=0.6)
            a.axhline(ARI_MIN, color="grey", lw=0.8, ls=":")
        a.set_title(t)
        a.set_xlabel("k")
        a.set_xticks(list(tab["k"]))
    fig.tight_layout()
    fig.savefig(path / "k_selection_curves.png", dpi=100)
    plt.close(fig)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(pe.size, size=min(30_000, pe.size), replace=False)
    ks = list(labels)
    fig, ax = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 4), sharey=True)
    for a, k in zip(ax, ks):
        a.scatter(pe[idx], late[idx], c=labels[k][idx], cmap="tab10", vmin=0, vmax=9,
                  s=1, alpha=0.4, rasterized=True)
        a.set_xscale("log")
        a.axhline(-LATE_TOL, color="k", lw=0.6, ls=":")
        a.axhline(LATE_TOL, color="k", lw=0.6, ls=":")
        a.set_title(f"k = {k}")
        a.set_xlabel("PE (scala log)")
    ax[0].set_ylabel("variazione tardiva")
    ax[0].set_ylim(-0.7, 1.0)
    fig.tight_layout()
    fig.savefig(path / "k_selection_scatter.png", dpi=90)
    plt.close(fig)


def run_selection(df):
    pids = training_pids(df)
    X, groups, pe, late = training_sample(pids)
    scaler = fit_scaler(X)
    Z = apply_scaler(X, scaler)
    print(f"Campione del training: {len(pids)} pazienti, {len(Z):,} voxel", flush=True)
    tab, labels = select_k(Z, groups, pe, late)
    tab.to_csv(OUT / "habitat_k_selection.csv", index=False)
    plot_k_selection(tab, labels, pe, late, FIG_DIR)

    buf = io.StringIO()
    with redirect_stdout(buf):
        print("=" * 70)
        print("FASE 3 - PASSO 1: SCELTA DEL NUMERO DI HABITAT k (solo training, senza pCR)")
        print("=" * 70)
        print_setup(pids, X, scaler)
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print("\nCurve per k (silhouette su sotto-campioni di "
                  f"{N_SIL:,} voxel x {N_SIL_REP}; stabilita' su {N_STAB} divisioni a meta' "
                  "delle pazienti; inerzia e BIC sul campione)")
            print(tab.to_string(index=False))
            pe_ref = np.median(pe)
            for k, lab in labels.items():
                d = describe_clusters(lab, pe, late, groups, k)
                d.insert(1, "nome", habitat_names(d["pe_mediano"], d["late_mediana"], pe_ref))
                print(f"\nk = {k}: habitat (unita' originali, voxel del campione)")
                print(d.to_string(index=False))
        print(f"\nRegola indicativa: il k piu' piccolo con silhouette alta e ARI >= {ARI_MIN}.")
    (OUT / "habitat_k_selection.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print("Salvati: outputs/habitat_k_selection.csv, outputs/habitat_k_selection.txt, "
          "outputs/figures/habitats/k_selection_*.png")


def print_setup(pids, X, scaler):
    print(f"Pazienti del training (exclude == False): {len(pids)}")
    print(f"Voxel campionati: {len(X):,} (al massimo {N_PER_PATIENT} per paziente, seme {SEED})")
    print("Trasformazione (stimata sul campione del training):")
    for j, f in enumerate(FEATURES):
        print(f"  {f:<7} taglio [{scaler['lo'][j]:.4f}; {scaler['hi'][j]:.4f}] "
              f"(p{CLIP_Q[0]}-p{CLIP_Q[1]}), media {scaler['mean'][j]:.4f}, "
              f"DS {scaler['sd'][j]:.4f}")
    print(f"  -> in unita' originali: PE tagliato in [{np.exp(scaler['lo'][0]):.3f}; "
          f"{np.exp(scaler['hi'][0]):.3f}]")


# --------------------------------------------------------------------------
# Passo 2: modello finale
# --------------------------------------------------------------------------
def fit_final(Z, k):
    km = fit_kmeans(Z, k)
    perm = order_centers(km.cluster_centers_)
    return km.cluster_centers_[perm]


def assign_patient(pid, scaler, centers):
    """Habitat di tutti i voxel validi di una paziente con il modello congelato."""
    m = load_map(pid)
    valid = m["valid"]
    Z = apply_scaler(features(m["pe"][valid], m["late"][valid]), scaler)
    hab = labels_to_volume(valid, nearest_center(Z, centers))
    return m, hab, erode_inplane(valid)


def run_final(df, k, figures=True):
    pids = training_pids(df)
    X, groups, pe, late = training_sample(pids)
    scaler = fit_scaler(X)
    Z = apply_scaler(X, scaler)
    centers = fit_final(Z, k)
    lab = nearest_center(Z, centers)
    pe_ref = float(np.median(pe))
    desc = describe_clusters(lab, pe, late, groups, k)
    names = habitat_names(desc["pe_mediano"], desc["late_mediana"], pe_ref)
    desc.insert(1, "nome", names)

    gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=SEED).fit(Z)
    g_lab = gmm.predict(Z)
    ari_gmm = adjusted_rand_score(lab, g_lab)
    cont = pd.crosstab(pd.Series(lab + 1, name="K-means"), pd.Series(g_lab, name="GMM"),
                       normalize="index")

    P = out_paths(k)
    np.savez(P["model"], k=k, features=np.array(FEATURES),
             centers_z=centers, clip_lo=scaler["lo"], clip_hi=scaler["hi"],
             mean=scaler["mean"], sd=scaler["sd"], names=np.array(names),
             pe_ref=pe_ref, late_tol=LATE_TOL, seed=SEED, n_per_patient=N_PER_PATIENT,
             train_pids=np.array(pids))

    P["hab"].mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in df.iterrows():
        m, hab, er = assign_patient(r["pid"], scaler, centers)
        np.savez_compressed(P["hab"] / f"{r['pid']}.npz", habitat=hab, eroded2d=er,
                            bbox_start=m["bbox_start"], bbox_stop=m["bbox_stop"],
                            full_shape=m["full_shape"], spacing_zyx=m["spacing_zyx"],
                            k=k, pid=np.array(r["pid"]))
        f = fractions(hab, k)
        f_er = fractions(np.where(er, hab, 0), k)
        row = {"pid": r["pid"], "cohort": r["cohort"], "split": r["split"],
               "exclude": r["exclude"], "n_valid": int((hab > 0).sum()),
               "n_eroded2d": int(er.sum())}
        row.update({f"frac_h{h + 1}": x for h, x in enumerate(f)})
        row.update({f"frac_h{h + 1}_eroded2d": x for h, x in enumerate(f_er)})
        rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(df)} pazienti", flush=True)
    fr = pd.DataFrame(rows)
    fr.to_csv(P["fractions"], index=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        final_report(pids, X, scaler, k, desc, pe_ref, ari_gmm, cont, fr)
    if figures:
        P["fig"].mkdir(parents=True, exist_ok=True)
        plot_final_scatter(pe, late, lab, names, P["fig"] / "habitats_scatter.png")
        fig_pids = pick_figure_pids(fr)
        files = maps.index_files()
        cohort = pd.read_csv(OUT / "cohort.csv").set_index("pid")
        for pid in fig_pids:
            habitat_figure(pid, cohort.loc[pid].copy(), files, names, k, P)
        buf.write(f"\nFigure (outputs/figures/habitats/k{k}/): habitats_scatter.png, "
                  f"{', '.join(fig_pids)}\n")
    P["report"].write_text(buf.getvalue(), encoding="utf-8")
    print(buf.getvalue())
    print("Salvati: " + ", ".join(str(P[x].relative_to(ROOT)) for x in
                                  ("hab", "model", "fractions", "report", "fig")))


def final_report(pids, X, scaler, k, desc, pe_ref, ari_gmm, cont, fr):
    print("=" * 70)
    ruolo = "analisi principale" if k == K_MAIN else "analisi di sensibilita'"
    print(f"FASE 3 - HABITAT CINETICI (K-means, k = {k}, {ruolo}; adattato solo sul training)")
    print("=" * 70)
    print_setup(pids, X, scaler)
    print(f"Riferimento per 'perfusione alta/bassa': PE mediano del campione = {pe_ref:.3f}; "
          f"forma della curva con tolleranza +-{LATE_TOL:.0%} sulla variazione tardiva")
    fp = OUT / "habitat_k_selection.csv"
    if fp.exists():
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            print("\nCurve per la scelta di k (passo 1)")
            print(pd.read_csv(fp).to_string(index=False))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 250):
        print("\nHabitat (unita' originali, voxel del campione del training)")
        print(desc.to_string(index=False))
        rep = [n for n in desc["nome"] if n.endswith(")")]
        if rep:
            print("ATTENZIONE: nomi ripetuti, da rivedere insieme: " + "; ".join(rep))
        print(f"\nConfronto con la GMM (stesso k, covarianza piena): ARI = {ari_gmm:.3f}")
        print("Righe = habitat K-means, colonne = componente GMM (frazione della riga)")
        print(cont.to_string())

        print("\n" + "-" * 70)
        print("Assegnazione a tutte le pazienti (modello congelato)")
        print("-" * 70)
        inc = fr[~fr["exclude"]]
        print(f"Pazienti incluse: {len(inc)}  "
              + "  ".join(f"{s}: {n}" for s, n in inc["split"].value_counts().sort_index().items())
              + f"; escluse (solo per sensibilita'): {fr['exclude'].sum()}")
        cols = [f"frac_h{h + 1}" for h in range(k)]
        print("\nFrazione di volume per habitat, mediana per paziente [p25 - p75], per split")
        out = {}
        for s, g in inc.groupby("split"):
            out[s] = [f"{g[c].median():.2f} [{g[c].quantile(.25):.2f}-{g[c].quantile(.75):.2f}]"
                      for c in cols]
        print(pd.DataFrame(out, index=[f"H{h + 1}" for h in range(k)]).to_string())
        print("\nPazienti con almeno il 5% del tumore nell'habitat (incluse)")
        print("  " + "  ".join(f"H{h + 1}: {(inc[c] >= 0.05).sum()}" for h, c in enumerate(cols)))
        n_hab = (inc[cols] >= 0.05).sum(axis=1)
        print("Numero di habitat presenti (>= 5%) per paziente: "
              + ", ".join(f"{n}: {c}" for n, c in n_hab.value_counts().sort_index().items()))
        tot = inc["n_valid"].sum()
        print(f"\nErosione 2D (solo sensibilita'): restano {inc['n_eroded2d'].sum():,} voxel "
              f"su {tot:,} ({inc['n_eroded2d'].sum() / tot:.1%}); pazienti con < 50% dopo "
              f"l'erosione: {(inc['n_eroded2d'] < 0.5 * inc['n_valid']).sum()}")
        dif = pd.Series({f"H{h + 1}": (inc[f"{c}_eroded2d"] - inc[c]).median()
                         for h, c in enumerate(cols)})
        print("Differenza mediana di frazione (erosa - intera): "
              + "  ".join(f"{i}: {v:+.3f}" for i, v in dif.items()))


def plot_final_scatter(pe, late, lab, names, path):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(pe.size, size=min(40_000, pe.size), replace=False)
    fig, a = plt.subplots(figsize=(7.5, 5.5))
    cmap = plt.get_cmap("tab10")
    for h, n in enumerate(names):
        s = idx[lab[idx] == h]
        a.scatter(pe[s], late[s], s=1, alpha=0.4, color=cmap(h), label=f"H{h + 1}: {n}",
                  rasterized=True)
    a.set_xscale("log")
    for y in (-LATE_TOL, LATE_TOL):
        a.axhline(y, color="k", lw=0.6, ls=":")
    a.set_ylim(-0.7, 1.0)
    a.set_xlabel("PE (scala log)")
    a.set_ylabel("variazione tardiva")
    a.legend(markerscale=8, fontsize=8, loc="upper right")
    a.set_title("Habitat sul campione del training")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def pick_figure_pids(fr):
    """Pazienti incluse a quantili diversi di n_valid (tutti gli split) + quelle fisse."""
    inc = fr[~fr["exclude"]].sort_values("n_valid").reset_index(drop=True)
    idx = np.unique(np.round(np.linspace(0.1, 0.9, N_FIG_PIDS) * (len(inc) - 1)).astype(int))
    pids = inc.loc[idx, "pid"].tolist()
    return pids + [p for p in FIG_PIDS_FIXED if p in set(fr["pid"]) and p not in pids]


def habitat_figure(pid, row, files, names, k, P):
    row["pid"] = pid
    vols, _ = maps.load_patient(row, files)
    m = load_map(pid)
    h = np.load(P["hab"] / f"{pid}.npz")
    hab, er = h["habitat"], h["eroded2d"]
    s0 = m["bbox_start"]
    zc = int((hab > 0).sum(axis=(1, 2)).argmax())
    z = s0[0] + zc
    mg = 10
    y0, x0 = max(s0[1] - mg, 0), max(s0[2] - mg, 0)
    y1 = min(m["bbox_stop"][1] + mg, vols["early"].shape[1])
    x1 = min(m["bbox_stop"][2] + mg, vols["early"].shape[2])

    def full(a, fill):
        out = np.full(vols["early"].shape[1:], fill, dtype=np.float64)
        out[s0[1]:m["bbox_stop"][1], s0[2]:m["bbox_stop"][2]] = a[zc]
        return out[y0:y1, x0:x1]

    early = vols["early"][z, y0:y1, x0:x1]
    vmax = np.percentile(vols["early"][z], 99.5)
    valid = m["valid"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1, 0.6]})
    cmap = ListedColormap([plt.get_cmap("tab10")(i) for i in range(k)])
    hb = full(np.where(hab > 0, hab, np.nan), np.nan)
    ax[0].imshow(early, cmap="gray", vmin=0, vmax=vmax)
    im = ax[0].imshow(hb, cmap=cmap, vmin=0.5, vmax=k + 0.5, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, ticks=range(1, k + 1))
    cb.ax.set_yticklabels([f"H{i + 1}" for i in range(k)])
    ax[0].contour(full(er.astype(float), 0), levels=[0.5], colors="white", linewidths=0.5)
    ax[0].set_title("habitat (bianco = maschera erosa 2D)", fontsize=10)
    for a, (key, t, cm, lo, hi) in zip(ax[1:3], [("pe", "PE", "viridis", 0, 3),
                                                 ("late", "variazione tardiva", "coolwarm_r",
                                                  -0.5, 0.5)]):
        a.imshow(early, cmap="gray", vmin=0, vmax=vmax)
        im = a.imshow(full(np.where(valid, m[key], np.nan), np.nan), cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=a, fraction=0.046)
        a.set_title(f"{t} (voxel validi)")
    f = fractions(hab, k)
    ax[3].barh([f"H{i + 1}" for i in range(k)][::-1], f[::-1],
               color=[cmap(i) for i in range(k)][::-1])
    ax[3].set_xlim(0, 1)
    ax[3].set_title("frazione di volume\n(tutto il tumore)", fontsize=10)
    for a in ax[:3]:
        a.axis("off")
    fig.suptitle(f"{pid}  (split {row['split']}, fetta z={z}, {int(valid.sum())} voxel validi)\n"
                 + "   ".join(f"H{i + 1}: {n}" for i, n in enumerate(names)), fontsize=10)
    fig.tight_layout()
    fig.savefig(P["fig"] / f"{pid}.png", dpi=90)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--k", type=int, help="numero di habitat scelto (senza: solo le curve di k)")
    ap.add_argument("--no-figures", action="store_true", help="non disegnare le figure")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_patients()
    if args.k is None:
        run_selection(df)
    else:
        run_final(df, args.k, figures=not args.no_figures)


if __name__ == "__main__":
    main()
