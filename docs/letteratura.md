# Letteratura – verifica di originalità e lavori di confronto

Ricerca fatta il 12/11/2026 (3 ricerche web mirate, NON una revisione sistematica).
Da completare con PubMed / Google Scholar: "mediation" + "radiomics"/"habitat" + "breast" + "pCR".

## Esito in breve

- **Mediazione causale sottotipo → habitat → pCR: nessuno studio trovato.** È la parte più originale della tesi.
- **Habitat DCE-MRI per la pCR: esistono**, ma su coorti diverse da I-SPY, senza confronto esplicito con un modello solo-sottotipo e senza analisi causale.
- **Valore aggiunto rispetto al sottotipo: evidenza contrastante** (alcuni sì, altri no), quindi M0 vs M1 su >1.100 pazienti con test tra coorti è una domanda aperta.

## Lavori da citare

### Più vicino al nostro (letto l'abstract)
- **Kantha & Loew 2026**, *Pretreatment DCE-MRI Resolves Response Quality Within Pathologic Endpoints in Neoadjuvant Breast Cancer*, arXiv:2608.22097 (preprint, non revisionato). https://arxiv.org/abs/2608.22097
  - Dati: I-SPY1, I-SPY2, Duke e altri (~1.200 pazienti).
  - Feature: **entropia** dell'enhancement pre-trattamento con una soglia (stato "favorevole"/"avverso"), **niente habitat né clustering**.
  - Outcome: **recidiva** tra le pazienti che hanno risposto (pCR/RCB come riferimento), **non predizione della pCR**.
  - Nessuna analisi causale/mediazione; analisi soprattutto nel sottogruppo HER2+.
  - → **Non si sovrappone** al nostro lavoro; utile come conferma che l'eterogeneità pre-trattamento sui dati I-SPY ha valore informativo.

### Habitat per la pCR
- Habitat da immagini PE con supervoxel + K-means, entropia di volume e di intensità; multicentrico, non I-SPY. https://pubmed.ncbi.nlm.nih.gov/41965129/
- Habitat intratumorali e peritumorali per la pCR, multicentrico. https://pubmed.ncbi.nlm.nih.gov/41456995/

### Eterogeneità oltre HR/HER2
- **Wu et al. 2018**, eterogeneità spaziale della perfusione su I-SPY1: aggiunge informazione oltre ER/PR/HER2, ma per la **sopravvivenza libera da recidiva** e non per la pCR. https://ncbi.nlm.nih.gov/pmc/articles/PMC6029132
- I-SPY2, indice di eterogeneità per i responder all'inibitore di AKT (multi-omica). https://pmc.ncbi.nlm.nih.gov/articles/PMC12874905/

### Radiomica contro sottotipo (evidenza contrastante)
- A favore: radiomica + sottotipo meglio del solo sottotipo. https://link.springer.com/article/10.1186/s13058-020-01291-w
- Contro: nessun valore aggiunto rispetto al modello clinico (multicentrico). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8157878/
- Radiomica per classificare il sottotipo (gli habitat dipendono dal sottotipo: freccia sottotipo → habitat nel DAG). https://pmc.ncbi.nlm.nih.gov/articles/PMC12634091/

## Per il DAG
- Un preprint recente tratta il sottotipo come **confondente** tra radiomica e pCR (influenza sia l'aspetto in risonanza sia la risposta). È coerente con il nostro DAG: il sottotipo sta a monte, gli habitat in mezzo (mediatore). Riferimento esatto da recuperare.

## Riferimenti da recuperare

Ogni scelta di metodo presa finora che ha bisogno di una fonte nella tesi. I riferimenti proposti vengono **dalla memoria e non sono ancora verificati**: prima di citarli, controlla autori, anno e rivista su PubMed o Google Scholar.

| Fase | Scelta da giustificare | Riferimento proposto (da verificare) |
|---|---|---|
| 0-1 | Dati I-SPY1 su TCIA e trial ACRIN 6657 | Hylton et al. 2012 (già letto); Hylton et al. 2016, *Radiology* (FTV e sopravvivenza libera da recidiva, I-SPY 1); Clark et al. 2013, *J Digit Imaging* (TCIA) |
| 2 | Soglia PE ≥ 70% per i voxel validi (FTV) | Hylton et al. 2012 (già letto) |
| 2-3 | Tolleranza ±5% per wash-out, plateau e crescita continua | **Fonte da trovare.** Attenzione: il lessico ACR BI-RADS per la RM usa probabilmente ±10% per la fase tardiva. Serve una fonte per ±5%, oppure va detto che la soglia è solo descrittiva (frazioni e nomi degli habitat) e non entra nel clustering |
| 2 | Forme della curva (tipo I, II, III) | Kuhl et al. 1999 (già letto) |
| 3 | Concetto di "habitat" tumorali | Gatenby, Grove e Gillies 2013, *Radiology* ("Quantitative imaging in cancer evolution and ecology"); Napel et al. 2018, *Cancer* (radiomica e habitat) |
| 3 | Habitat DCE-MRI per la pCR, clustering dei voxel tra pazienti | Wu et al. 2016, *J Magn Reson Imaging* (partizione intratumorale della DCE-MRI per la risposta alla NAC); Wu et al. 2018 (già in elenco); i due studi sugli habitat per la pCR già in elenco |
| 3 | K-means e inizializzazione k-means++ | Lloyd 1982 o MacQueen 1967; Arthur e Vassilvitskii 2007 |
| 3 | GMM e BIC | Schwarz 1978 (BIC) |
| 3 | Silhouette | Rousseeuw 1987, *J Comput Appl Math* |
| 3 | ARI | Hubert e Arabie 1985, *J Classification* |
| 3 | Scelta di k tramite la stabilità su sotto-campioni | Lange et al. 2004, *Neural Computation*; Ben-Hur et al. 2002 |
| 3 | scikit-learn | Pedregosa et al. 2011, *JMLR* |
| 3-5 | Adattare tutto solo sul training (data leakage) | Kapoor e Narayanan 2023, *Patterns* |
| 6 | Preprint che tratta il sottotipo come confondente | Riferimento esatto da recuperare (vedi "Per il DAG") |
| 6 | DoWhy | Sharma e Kiciman 2020 (arXiv) |
| — | Esclusione di pazienti per problemi di acquisizione (rapporto del fondo) | Nessuna fonte specifica: è una regola nostra, motivata dai dati (roadmap, Fase 2) |
