"""
Test di src/04_features.py su piccoli array sintetici (valori calcolabili a mano).

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python tests/test_04_features.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("feat", ROOT / "src" / "04_features.py")
feat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feat)

ISO = (1.0, 1.0, 1.0)


def test_frazioni_ignorano_i_non_validi():
    hab = np.array([[[0, 1, 1, 2]], [[3, 3, 0, 1]]], dtype=np.uint8)
    fr = feat.fractions(hab, 3)
    assert np.allclose(fr, [3 / 6, 1 / 6, 2 / 6]) and np.isclose(fr.sum(), 1)
    assert np.all(np.isnan(feat.fractions(np.zeros((2, 2, 2), np.uint8), 3)))


def test_entropia_normalizzata():
    assert np.isclose(feat.entropy_norm([1 / 3, 1 / 3, 1 / 3]), 1.0)
    assert np.isclose(feat.entropy_norm([0.5, 0.5]), 1.0)            # k = 2: stesso massimo
    assert np.isclose(feat.entropy_norm([1.0, 0.0, 0.0]), 0.0)        # un solo habitat
    e = feat.entropy_norm([0.5, 0.5, 0.0])
    assert np.isclose(e, np.log(2) / np.log(3))                       # habitat assente = 0 log 0


def test_regioni_diagonali_unite_con_26_vicini():
    m = np.zeros((3, 3, 3), dtype=bool)
    m[0, 0, 0] = m[1, 1, 1] = m[2, 2, 2] = True                       # solo angoli in comune
    assert feat.regions(m, ISO, min_mm3=0)[0] == 1
    s6 = feat.ndimage.generate_binary_structure(3, 1)
    assert feat.regions(m, ISO, min_mm3=0, structure=s6)[0] == 3


def test_soglia_in_mm3_dipende_dallo_spacing():
    m = np.zeros((1, 1, 20), dtype=bool)
    m[0, 0, :10] = True                                               # regione da 10 voxel
    m[0, 0, 15] = True                                                # regione da 1 voxel
    assert feat.regions(m, ISO, min_mm3=5)[0] == 1                    # 10 mm^3 si', 1 mm^3 no
    assert feat.regions(m, (2.0, 1.0, 1.0), min_mm3=15)[0] == 1      # 20 mm^3 si', 2 no
    assert feat.regions(m, (0.5, 1.0, 1.0), min_mm3=15)[0] == 0      # 5 mm^3: nessuna


def test_frammentazione():
    m = np.zeros((1, 1, 20), dtype=bool)
    m[0, 0, :8] = True
    m[0, 0, 10:12] = True                                             # 8 + 2 voxel
    assert np.isclose(feat.regions(m, ISO, min_mm3=0)[1], 1 - 8 / 10)
    blocco = np.ones((2, 2, 2), dtype=bool)
    assert feat.regions(blocco, ISO, min_mm3=0)[1] == 0.0
    n, frag = feat.regions(np.zeros((2, 2, 2), dtype=bool), ISO)      # habitat assente
    assert n == 0 and np.isnan(frag)


def test_frammentazione_mancante_se_habitat_minuscolo():
    m = np.zeros((1, 1, 20), dtype=bool)
    m[0, 0, [0, 5, 10, 15]] = True                                    # 4 voxel sparsi
    assert np.isclose(feat.regions(m, ISO, min_mm3=0)[1], 0.75)       # senza soglia: rumore
    assert np.isnan(feat.regions(m, ISO, min_mm3=5)[1])               # 4 mm^3 < 5: mancante
    assert np.isclose(feat.regions(m, (2.0, 1.0, 1.0), min_mm3=5)[1], 0.75)   # 8 mm^3: ok
    grande = np.ones((4, 4, 4), dtype=bool)                           # 64 mm^3 >= 50
    assert feat.regions(grande, ISO)[1] == 0.0


def test_area_delle_facce_con_spacing_anisotropo():
    sp = (2.5, 1.0, 0.5)                                              # (z, y, x) in mm
    # due voxel vicini lungo x: faccia di area dz * dy
    hab = np.array([[[1, 2]]], dtype=np.uint8)
    C = feat.contact_areas(hab, sp, 2)
    assert np.isclose(C[0, 1], 2.5 * 1.0) and np.isclose(C[1, 0], 2.5 * 1.0)
    # due voxel vicini lungo z: faccia di area dy * dx
    hab = np.array([[[1]], [[2]]], dtype=np.uint8)
    assert np.isclose(feat.contact_areas(hab, sp, 2)[0, 1], 1.0 * 0.5)
    # lungo y: dz * dx
    hab = np.array([[[1], [1]]], dtype=np.uint8)
    C = feat.contact_areas(hab, sp, 2)
    assert np.isclose(C[0, 0], 2.5 * 0.5) and C[0, 1] == 0


def test_contatti_con_voxel_non_validi_non_contano():
    hab = np.array([[[1, 0, 2]]], dtype=np.uint8)
    assert feat.contact_areas(hab, ISO, 2).sum() == 0
    glob, pairs = feat.adjacency(feat.contact_areas(hab, ISO, 2))
    assert np.isnan(glob) and np.isnan(pairs[(1, 2)])


def test_adiacenza_blocchi_contro_mescolati():
    blocchi = np.array([[[1, 1, 1, 2, 2, 2]]], dtype=np.uint8)        # 5 contatti, 1 misto
    misti = np.array([[[1, 2, 1, 2, 1, 2]]], dtype=np.uint8)          # 5 contatti, 5 misti
    a, _ = feat.adjacency(feat.contact_areas(blocchi, ISO, 2))
    b, _ = feat.adjacency(feat.contact_areas(misti, ISO, 2))
    assert np.isclose(a, 1 / 5) and np.isclose(b, 1.0)


def test_adiacenza_per_coppia_somma_al_globale():
    hab = np.array([[[1, 2, 3, 3, 1, 1]]], dtype=np.uint8)
    # contatti: 1-2, 2-3, 3-3, 3-1, 1-1 -> 5 in tutto, 3 tra habitat diversi
    glob, pairs = feat.adjacency(feat.contact_areas(hab, ISO, 3))
    assert np.isclose(glob, 3 / 5)
    assert np.isclose(pairs[(1, 2)], 1 / 5) and np.isclose(pairs[(1, 3)], 1 / 5)
    assert np.isclose(pairs[(2, 3)], 1 / 5)
    assert np.isclose(sum(pairs.values()), glob)


def test_feature_della_paziente():
    hab = np.zeros((2, 4, 4), dtype=np.uint8)
    hab[:, :, :2] = 1
    hab[:, :, 2:] = 2                                                 # H3 assente
    r = feat.patient_features(hab, ISO, 3, min_mm3=1)
    assert r["n_valid"] == 32 and np.isclose(r["vol_valid_cm3"], 32 / 1000)
    assert np.isclose(r["frac_h1"], 0.5) and r["frac_h3"] == 0
    assert np.isclose(r["entropy"], np.log(2) / np.log(3))
    assert r["nreg_h1"] == 1 and r["frag_h1"] == 0 and r["nreg_h3"] == 0
    assert np.isnan(r["frag_h3"])
    assert {"adj", "adj_h1h2", "adj_h1h3", "adj_h2h3"} <= set(r)
    r2 = feat.patient_features(np.clip(hab, 0, 2), ISO, 2)
    assert "adj_h1h2" not in r2                                       # con k = 2 coincide con adj


def test_candidate_senza_h1_e_tum_vol_per_primo():
    assert feat.model_candidates(3) == ["tum_vol", "frac_h2", "frac_h3", "entropy", "adj",
                                        "frag_h1", "frag_h2", "frag_h3"]
    assert feat.model_candidates(2) == ["tum_vol", "frac_h2", "entropy", "adj",
                                        "frag_h1", "frag_h2"]


def test_filtro_correlazione_segue_la_priorita():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    df = pd.DataFrame({"tum_vol": x, "a": rng.normal(size=100),
                       "copia_vol": x * 3 + 1e-3 * rng.normal(size=100),
                       "b": rng.normal(size=100)})
    df["copia_a"] = -df["a"]                                          # rho = -1: conta |rho|
    kept, dropped = feat.corr_filter(df, ["tum_vol", "copia_vol", "a", "copia_a", "b"])
    assert kept == ["tum_vol", "a", "b"]                              # tum_vol mai scartato
    assert [(d, kc) for d, kc, _ in dropped] == [("copia_vol", "tum_vol"), ("copia_a", "a")]
    assert np.isclose(dropped[1][2], -1.0)


def test_filtro_usa_solo_le_righe_passate():
    # nel "training" a e b sono identiche, nel resto no: il filtro guarda solo il training
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8], "b": [1, 2, 3, 4, 8, 1, 7, 2],
                       "split": ["train"] * 4 + ["test"] * 4})
    kept, _ = feat.corr_filter(df[df["split"] == "train"], ["a", "b"])
    assert kept == ["a"]
    kept, _ = feat.corr_filter(df, ["a", "b"])
    assert kept == ["a", "b"]


def test_la_pcr_non_viene_letta():
    assert not any("pcr" in c.lower() for c in feat.COHORT_COLS)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test superati")
