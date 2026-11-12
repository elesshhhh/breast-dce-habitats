# dce-habitat-causal

**Eterogeneità tumorale e Causal AI da DCE-MRI per la predizione della risposta alla terapia neoadiuvante nel tumore al seno**

Il progetto studia se l'eterogeneità intratumorale, misurata sulla DCE-MRI pre-trattamento, aiuta a predire la risposta patologica completa (pCR) alla chemioterapia neoadiuvante, oltre a quanto già spiegano i fattori clinici (sottotipo recettoriale HR/HER2, età).

Dati: coorti I-SPY1 e I-SPY2 del dataset pubblico BreastDCEDL (versione MinCrop: fasi pre-contrasto, post-contrasto precoce e tardivo, segmentazioni 3D del tumore).

## Pipeline

1. Mappe di enhancement voxel per voxel: Peak Enhancement, Signal Enhancement Ratio (SER), variazione tardiva.
2. Clustering dei voxel tumorali in habitat cinetici (K-means / GMM a livello di popolazione).
3. Biomarcatori di eterogeneità: frazioni degli habitat, entropia, frammentazione, adiacenza spaziale.
4. Modelli predittivi (clinico, clinico + habitat, solo habitat) con AUC, accuratezza, sensibilità e specificità; analisi per sottotipo e tra coorti.
5. Analisi causale: DAG esplicito, effetto aggiustato e mediazione (sottotipo → habitat → pCR) con DoWhy, test di confutazione ed E-value.

## Struttura

```
dce-habitat-causal/
├── data/            # dati scaricati (NON versionati, vedi .gitignore)
├── outputs/         # mappe, feature, figure, report (NON versionati)
├── scripts/         # script di shell (download dati)
├── src/             # codice Python della pipeline
├── tests/           # test su dati sintetici (python tests/test_02_maps.py)
├── docs/            # note di lavoro
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh          # metadati + dati demo
python src/00_check_setup.py           # verifica ambiente e dati
```

Per scaricare le coorti complete (estratte in streaming, senza salvare l'archivio):

```bash
bash scripts/download_data.sh ispy1    # ~1,3 GB
bash scripts/download_data.sh ispy2    # ~13,6 GB, solo quando la pipeline funziona su I-SPY1
```

## Dataset

Fridman N. et al. *BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients*. Scientific Data, 2026. doi:10.1038/s41597-026-06589-6 — Zenodo record 18114231 (licenza CC BY-NC 4.0).
