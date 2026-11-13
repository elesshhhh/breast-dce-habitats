"""
Test di src/03_habitats.py su piccoli array sintetici (valori calcolabili a mano).

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python tests/test_03_habitats.py
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("habitats", ROOT / "src" / "03_habitats.py")
hab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hab)


def test_variabili_log_pe_e_late():
    X = hab.features([1.0, np.e], [-0.2, 0.3])
    assert X.shape == (2, 2)
    assert np.allclose(X[:, 0], [0.0, 1.0]) and np.allclose(X[:, 1], [-0.2, 0.3])


def test_variabili_rifiutano_valori_non_validi():
    for pe, late in (([0.0], [0.1]), ([np.nan], [0.1]), ([1.0], [np.nan])):
        try:
            hab.features(pe, late)
        except ValueError:
            continue
        raise AssertionError("doveva rifiutare PE <= 0, PE NaN o late NaN")


def test_campione_per_paziente():
    rng = np.random.default_rng(0)
    assert np.array_equal(hab.sample_indices(5, 10, rng), np.arange(5))   # tutti se pochi
    idx = hab.sample_indices(1000, 100, rng)
    assert idx.size == 100 and np.unique(idx).size == 100                  # senza ripetizioni
    assert idx.min() >= 0 and idx.max() < 1000
    a = hab.sample_indices(1000, 100, np.random.default_rng(1))
    b = hab.sample_indices(1000, 100, np.random.default_rng(1))
    assert np.array_equal(a, b)                                            # seme fisso


def test_trasformazione_taglio_e_z():
    X = np.column_stack([np.arange(1001.0), np.arange(1001.0) * 2])
    s = hab.fit_scaler(X, q=(1, 99))
    assert np.allclose(s["lo"], [10, 20]) and np.allclose(s["hi"], [990, 1980])
    Z = hab.apply_scaler(X, s)
    assert np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0), 1)
    # un valore fuori dall'intervallo del training viene portato al taglio
    far = hab.apply_scaler(np.array([[1e6, -1e6]]), s)
    edge = hab.apply_scaler(np.array([[990.0, 20.0]]), s)
    assert np.allclose(far, edge)


def test_trasformazione_usa_solo_i_dati_passati():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(500, 2))
    s1 = hab.fit_scaler(train)
    hab.apply_scaler(rng.normal(5, 3, size=(500, 2)), s1)    # applicare non modifica i parametri
    s2 = hab.fit_scaler(train)
    assert all(np.array_equal(s1[k], s2[k]) for k in s1)


def test_centroide_piu_vicino():
    centers = np.array([[0.0, 0.0], [10.0, 0.0]])
    Z = np.array([[1.0, 1.0], [9.0, -1.0], [4.9, 0.0]])
    assert list(hab.nearest_center(Z, centers)) == [0, 1, 0]


def test_ordine_fisso_degli_habitat():
    centers = np.array([[2.0, 0.0], [-1.0, 5.0], [-1.0, -3.0], [0.5, 0.0]])
    perm = hab.order_centers(centers)
    # per log(PE) crescente, a parita' per variazione tardiva crescente
    assert list(perm) == [2, 1, 3, 0]


def test_kmeans_ritrova_due_gruppi():
    rng = np.random.default_rng(0)
    a = rng.normal([-3, 0], 0.3, size=(300, 2))
    b = rng.normal([3, 0], 0.3, size=(300, 2))
    centers = hab.fit_final(np.vstack([a, b]), 2)
    assert centers[0, 0] < 0 < centers[1, 0]                 # habitat 1 = PE piu' basso
    lab = hab.nearest_center(np.vstack([a, b]), centers)
    assert (lab[:300] == 0).all() and (lab[300:] == 1).all()


def test_forma_della_curva():
    assert hab.curve_shape(-0.10) == "wash-out"
    assert hab.curve_shape(-0.05) == "plateau"               # estremi +-5% = plateau
    assert hab.curve_shape(0.05) == "plateau"
    assert hab.curve_shape(0.20) == "crescita continua"


def test_nomi_degli_habitat():
    n = hab.habitat_names([1.0, 3.0, 3.5], [0.2, -0.2, 0.0], pe_ref=1.66)
    assert n == ["perfusione bassa, crescita continua", "perfusione alta, wash-out",
                 "perfusione alta, plateau"]
    n = hab.habitat_names([2.0, 3.0], [-0.2, -0.3], pe_ref=1.66)
    assert n == ["perfusione alta, wash-out (1)", "perfusione alta, wash-out (2)"]


def test_mappa_e_frazioni():
    valid = np.zeros((1, 2, 3), dtype=bool)
    valid[0, 0, :] = True
    valid[0, 1, 0] = True
    h = hab.labels_to_volume(valid, [0, 1, 1, 2])            # etichette in ordine C (z, y, x)
    assert h.dtype == np.uint8
    assert h[0, 0].tolist() == [1, 2, 2] and h[0, 1].tolist() == [3, 0, 0]
    assert np.allclose(hab.fractions(h, 3), [0.25, 0.5, 0.25])
    assert np.allclose(hab.fractions(h, 4), [0.25, 0.5, 0.25, 0.0])


def test_erosione_nel_piano():
    valid = np.zeros((3, 5, 5), dtype=bool)
    valid[:, :, :] = True                                    # cubo che riempie il ritaglio
    er = hab.erode_inplane(valid)
    # tutte e 3 le fette restano; in ogni fetta si toglie solo il bordo (5x5 -> 3x3)
    assert er.sum() == 3 * 9
    assert er[:, 1:4, 1:4].all() and not er[:, 0, :].any()


def test_stabilita_su_gruppi_netti():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal([-3, 0], 0.3, (400, 2)), rng.normal([3, 0], 0.3, (400, 2))])
    groups = np.tile(np.repeat(np.arange(8), 50), 2)         # 8 "pazienti" con voxel di entrambi
    ari = hab.stability(Z, groups, 2, n_rep=3)
    assert np.allclose(ari, 1.0)


def test_solo_training_non_escluse():
    df = pd.DataFrame({"pid": list("abcde"), "split": ["train", "train", "val", "test", "train"],
                       "exclude": [False, True, False, False, False]})
    assert hab.training_pids(df) == ["a", "e"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test superati")
