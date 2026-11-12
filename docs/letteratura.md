# Letteratura – verifica di originalità e lavori di confronto

Ricerca fatta il 12/11/2026 (3 ricerche web mirate, NON una revisione sistematica).
Da completare in Fase 7 con PubMed / Google Scholar: "mediation" + "radiomics"/"habitat" + "breast" + "pCR".

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
- Un preprint recente tratta il sottotipo come **confondente** tra radiomica e pCR (influenza sia l'aspetto in risonanza sia la risposta). È coerente con il nostro DAG: il sottotipo sta a monte, gli habitat in mezzo (mediatore). Riferimento esatto da recuperare in Fase 7.
