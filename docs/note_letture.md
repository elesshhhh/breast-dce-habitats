# Note sulle letture della Fase 0

## 1. Fridman et al. 2026 – BreastDCEDL (Scientific Data, doi:10.1038/s41597-026-06589-6)

- 2.070 pazienti: I-SPY1 172, I-SPY2 982, Duke 916. pCR disponibile per 1.452 (tutte le I-SPY, 298 Duke).
- Conversione DICOM → NIfTI (dcm2niix) senza normalizzazione di intensità o spaziale: valori originali conservati.
  → Le intensità NON sono confrontabili tra pazienti/scanner: servono misure relative (PE, SER).
- Nomi file: `<id>_<dataset>_acq<N>.nii.gz`; nel CSV le colonne `pre`, `post_early`, `post_late` dicono quale acquisizione è quale fase.
- Risoluzione temporale ~120 s nelle I-SPY (~60 s in Duke).
- MinCrop: 3 fasi per paziente, crop 256×256×Z attorno al tumore.
- **Maschere I-SPY**: derivate dalle maschere di analisi FTV del trial, codificate a bit (1 = VOI manuale, 32 = sopra soglia; 33 = dentro il VOI e sopra soglia), convertite in binario.
  Nel trial la soglia è PE ≥ 70% (nominale, adattata per sito), con soglia sul fondo pre-contrasto e regioni "OMIT" escluse a mano.
  → **La maschera contiene solo tessuto che si illumina**: le zone necrotiche/poco perfuse (PE < 70%) sono già escluse.
- Colonne CSV utili: `mask_start/mask_end` (slice z con tumore), `sraw/eraw`, `scol/ecol` (bounding box), `tum_vol` (cm³, verificato sulle maschere demo: n voxel × slice_thick × xy_spacing² / 1000), `menopause`, `HR`, `HER2`, `HR_HER2_STATUS`, `dataset`, `test` (split).
- Split train/val/test 1.532/268/270, stratificati su pCR (e HER2); tassi di pCR ~29-30% in tutti e tre.
- ViT: immagini RGB con pre/early/late nei tre canali, slice con tumore; AUC 0,94 solo nel sottogruppo HR+/HER2−.

## 2. Hylton et al. 2012 – ACRIN 6657/I-SPY 1 (Radiology, doi:10.1148/radiol.12110748)

- Protocollo: 1,5 T, T1 3D con soppressione del grasso, ~1 mm nel piano, fette ≤ 2,5 mm.
- Tempi: fase early ~2,5 min dopo l'iniezione, fase late ~7,5 min.
- Formule (S0 = pre, S1 = early, S2 = late):
  - PE = (S1 − S0) / S0 × 100%
  - SER = (S1 − S0) / (S2 − S0)
- FTV = voxel con PE sopra soglia (70% nominale, adattata per sito).
- Peak SER = media SER più alta su una piccola regione connessa nel tumore.
- AUC per pCR (univariata): volume early 0,70; peak SER early 0,57; più alte a metà/fine terapia.
  → Pre-trattamento la sola cinetica predice poco: coerente con la nostra domanda "aggiunge qualcosa al sottotipo?".
- Limite dichiarato: risoluzione temporale bassa → niente modelli farmacocinetici; SER meno interpretabile fisiologicamente.

## 3. Kuhl et al. 1999 – curve tempo-intensità (Radiology, doi:10.1148/radiology.211.1.r99ap38101)

- Tre forme di curva dopo il picco iniziale:
  - Tipo I: crescita continua (tipica dei benigni)
  - Tipo II: plateau (intermedio)
  - Tipo III: wash-out, calo del segnale (forte indicatore di malignità)
- Nei tumori maligni: ~9% tipo I, ~34% tipo II, ~57% tipo III.
- La forma della curva discrimina molto meglio della sola velocità di enhancement (specificità 83% contro 37%).
- In pratica la soglia tra le classi è una piccola tolleranza sulla variazione late-vs-early (in letteratura es. ±5%).

## Conseguenze per il progetto

1. Usare solo misure relative (PE, SER, variazione tardiva): le intensità grezze variano per scanner.
2. Le maschere I-SPY includono solo voxel con PE ≥ ~70%: gli habitat descriveranno l'eterogeneità **del tessuto che capta**, non la necrosi. Da dichiarare come limite. Verificato in Fase 1 sui 5 pazienti demo I-SPY con fasi DCE: PE minimo nella maschera = 0,700 esatto, nessun voxel sotto 0,70.
3. In termini di SER: > 1 ≈ wash-out (tipo III), ≈ 1 plateau (tipo II), < 1 crescita continua (tipo I).
4. Riferimenti di confronto pre-trattamento: AUC modeste (~0,6-0,7) nella letteratura I-SPY.
