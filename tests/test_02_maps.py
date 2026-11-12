"""
Test di src/02_maps.py su piccoli array sintetici (valori calcolabili a mano).

Uso (dalla cartella principale della repo, con l'ambiente attivo):
    python tests/test_02_maps.py
"""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("maps", ROOT / "src" / "02_maps.py")
maps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(maps)


def one_voxel(pre, early, late, in_mask=True):
    """Calcola le mappe su un volume 1x1x1."""
    a = lambda v: np.array([[[v]]], dtype=np.float32)
    return maps.compute_maps(a(pre), a(early), a(late), np.array([[[in_mask]]]))


def val(res, key):
    return res[key][0, 0, 0]


def test_voxel_plateau():
    r = one_voxel(100, 200, 200)
    assert np.isclose(val(r, "pe"), 1.0)
    assert np.isclose(val(r, "ser"), 1.0)
    assert np.isclose(val(r, "late"), 0.0)
    assert val(r, "valid") and val(r, "reason") == maps.REASON_VALID


def test_voxel_washout():
    r = one_voxel(100, 300, 250)          # PE 2, SER 200/150, late -50/300
    assert np.isclose(val(r, "pe"), 2.0)
    assert np.isclose(val(r, "ser"), 200 / 150)
    assert np.isclose(val(r, "late"), -50 / 300)


def test_pe_sotto_soglia_escluso_ma_calcolato():
    r = one_voxel(100, 150, 160)          # PE 0,5 < 0,70
    assert not val(r, "valid")
    assert val(r, "reason") == maps.REASON_PE_LOW
    assert np.isclose(val(r, "pe"), 0.5)  # la mappa resta calcolata (analisi di sensibilita')
    assert np.isclose(val(r, "ser"), 50 / 60)


def test_pe_esattamente_soglia_valido():
    r = one_voxel(100, 170, 170)
    assert val(r, "valid")


def test_ser_denominatore_nullo_o_negativo():
    for late in (100, 90):                # late = pre e late < pre
        r = one_voxel(100, 200, late)
        assert np.isnan(val(r, "ser"))
        assert val(r, "valid")            # il voxel resta valido per PE e variazione tardiva
        assert np.isclose(val(r, "pe"), 1.0)


def test_pre_non_positivo():
    r = one_voxel(0, 50, 60)
    assert np.isnan(val(r, "pe")) and np.isnan(val(r, "ser"))
    assert not val(r, "valid") and val(r, "reason") == maps.REASON_PRE_LE0


def test_fuori_maschera():
    r = one_voxel(100, 200, 200, in_mask=False)
    for k in ("pe", "ser", "late"):
        assert np.isnan(val(r, k))
    assert not val(r, "valid") and val(r, "reason") == maps.REASON_OUTSIDE


def test_bbox_e_ritaglio():
    mask = np.zeros((5, 6, 7), dtype=bool)
    mask[1:3, 2:5, 3] = True
    start, stop = maps.mask_bbox(mask)
    assert tuple(start) == (1, 2, 3) and tuple(stop) == (3, 5, 4)
    crop = maps.crop(mask, start, stop)
    assert crop.shape == (2, 3, 1) and crop.all()


def test_classi_curva():
    late = np.array([-0.10, -0.05, 0.0, 0.05, 0.10, np.nan])
    f = maps.curve_fractions(late, tol=0.05)
    # i valori esattamente a +-5% sono plateau; i NaN non contano
    assert np.isclose(f["washout"], 1 / 5)
    assert np.isclose(f["plateau"], 3 / 5)
    assert np.isclose(f["persistent"], 1 / 5)


def test_rapporti_del_fondo():
    # tumore al centro; fondo = tessuto fuori dal tumore con pre sopra la mediana
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[9:11, 9:11, 9:11] = True
    pre = np.ones(shape)
    pre[:, :10, :] = 100.0                # meta' volume "tessuto", meta' "aria"
    early, late = pre * 1.3, pre * 0.5    # late piu' scura in tutta l'immagine
    r_early, r_late = maps.background_ratios(pre, early, late, mask)
    assert np.isclose(r_early, 1.3) and np.isclose(r_late, 0.5)


def test_regola_di_esclusione():
    assert maps.exclusion_reasons(1.3, 1.35, 1.0) == []
    assert maps.exclusion_reasons(0.9, 2.5, 0.5) == []        # estremi inclusi = accettati
    assert len(maps.exclusion_reasons(1.3, 0.49, 1.0)) == 1   # late troppo scura
    assert len(maps.exclusion_reasons(0.67, 1.0, 1.0)) == 1   # early troppo scura
    assert len(maps.exclusion_reasons(1.3, 13.4, 1.0)) == 1   # late satura
    assert len(maps.exclusion_reasons(1.0, 1.1, 0.005)) == 1  # quasi nessun voxel valido
    assert len(maps.exclusion_reasons(0.5, 13.0, 0.1)) == 3


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} test superati")
