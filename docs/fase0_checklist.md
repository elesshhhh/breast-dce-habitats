# Fase 0 – Setup e studio (giorni 1-3)

## Setup (da Terminale, nella cartella `dce-habitat-causal`)

- [ ] `python3 --version` → serve Python 3.10 o superiore
- [ ] `python3 -m venv .venv && source .venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] `bash scripts/download_data.sh` (metadati + demo, ~55 MB)
- [ ] `python src/00_check_setup.py` → genera `outputs/check_setup_report.txt`
- [ ] `git init && git add . && git commit -m "Fase 0: struttura progetto"`
- [ ] (facoltativo) creare la repo `dce-habitat-causal` su GitHub e fare `git push`

## Letture

| Paper | Identificativo | Cosa cercare |
|---|---|---|
| Fridman et al. 2026, Sci Data (BreastDCEDL) | doi:10.1038/s41597-026-06589-6 | Come sono costruite le coorti, cosa contiene MinCrop, come sono definiti gli split, Tabella 1 |
| Kuhl et al. 1999, Radiology | doi:10.1148/radiology.211.1.r99ap38101 | Le tre forme di curva (tipo I, II, III) e cosa significano |
| Hylton et al. 2012, Radiology | doi:10.1148/radiol.12110748 | Definizione di PE e SER, soglie usate in I-SPY, FTV |

Mentre leggi, annota in `docs/note.md`: definizioni (pCR, NAC, HR, HER2, sottotipi), formule e soglie di PE/SER, numeri della coorte.

## Prima di passare alla Fase 1

- [ ] Il report di `00_check_setup.py` non segnala librerie mancanti
- [ ] I conteggi di pazienti per coorte tornano con il paper (I-SPY1 172, I-SPY2 982)
- [ ] Hai aperto almeno un paziente demo e capito quali file sono le fasi e quale la maschera
- [ ] Hai deciso quando scaricare I-SPY1 (`bash scripts/download_data.sh ispy1`)
