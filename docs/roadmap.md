# Roadmap di sviluppo – dce-habitat-causal

Pianificazione ipotizzata: **6 settimane di lavoro più 1 settimana di margine**. Il piano segue la proposta aggiornata: habitat, valore incrementale oltre il sottotipo e mediazione causale.

## Panoramica

| Fase | Periodo | Cosa produce |
|---|---|---|
| 0. Setup e studio | Giorni 1-3 | Ambiente pronto, dati scaricati, prime letture |
| 1. Esplorazione dei dati | Settimana 1 | Coorte definita, tabella descrittiva |
| 2. Mappe di enhancement | Settimane 1-2 | Mappe PE, SER e wash-out per ogni paziente |
| 3. Clustering degli habitat | Settimana 2-3 | Mappe di habitat |
| 4. Biomarcatori | Settimana 3 | Tabella di feature, una riga per paziente |
| 5. Modelli predittivi | Settimana 4 | AUC con intervalli di confidenza e confronti |
| 6. Analisi causale | Settimane 4-5 | DAG, stime di effetto e mediazione, test di robustezza |
| 7. Scrittura e revisione | Settimane 5-6 (+ margine) | Tesi completa |

La scrittura non va lasciata alla fine: il capitolo sui metodi si scrive durante le fasi 2-6, mentre lavori.

---

## Fase 0: Setup e studio (giorni 1-3)

**Cosa fare**
- Leggi tre paper: Fridman (BreastDCEDL), Kuhl 1999 e Hylton 2012.
- Prepara l'ambiente Python con `nibabel`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `statsmodels`, `dowhy`.
- Scarica `BreastDCEDL_metadata_min_crop.csv` e `BreastDCEDL_demo_data.tar.gz` (55 MB).
- Lancia il notebook demo per vedere com'è fatto un paziente.

**Come.** Scarica prima la demo e verifica di saper aprire un NIfTI e la sua maschera. Solo dopo scarica gli archivi grandi: `ISPY1_min_crop` (1,3 GB) e `ISPY2_min_crop` (13,6 GB).

**Attenzione allo spazio su disco.** Servono circa 25-40 GB liberi. Scelta adottata: estrazione in streaming con `scripts/download_data.sh` (circa 34 GB liberi disponibili). Alternative:
- liberare spazio sul Mac;
- usare un disco esterno;
- lavorare su Google Colab con i dati su Google Drive.

Decidilo subito, perché se la scelta arriva tardi blocca tutto il resto.

**Perché.** I problemi tecnici (formati, spazio, librerie) vanno risolti prima di iniziare. Le tre letture ti danno il vocabolario minimo per capire cosa stai calcolando.

---

## Fase 1: Esplorazione dei dati e definizione della coorte (settimana 1)

**Cosa fare**
- Filtra il CSV tenendo solo I-SPY1 e I-SPY2, con pCR disponibile e maschera presente.
- Costruisci la "Tabella 1": numero di pazienti, età, HR, HER2, sottotipo e tasso di pCR, sia in totale sia per sottotipo e per coorte.
- Controlla la colonna `test`, cioè lo split train/val/test.

**Come**
- Usa `pandas`. Confronta i tuoi conteggi con quelli pubblicati nel paper BreastDCEDL: se tornano, stai leggendo i dati nel modo giusto.
- Guarda come varia la pCR tra sottotipi. Questa variazione è proprio il confondimento che dovrai gestire nell'analisi causale.

**Perché.** Devi sapere esattamente su quali pazienti lavori e perché escludi Duke (niente maschere voxel e pCR solo per un terzo dei casi). La Tabella 1 serve anche nella tesi.

**Checkpoint con il relatore.** Mostragli la coorte e una prima bozza del DAG, e fagli approvare la riformulazione della parte causale (§3.6 della proposta).

---

## Fase 2: Preprocessing e mappe di enhancement (settimane 1-2)

**Cosa fare.** Per ogni paziente:
1. Carica le 3 fasi (pre, early, late) e la maschera.
2. Calcola voxel per voxel dentro la maschera:
   - **PE = (early − pre) / pre**: quanto e quanto in fretta il tumore si illumina;
   - **SER = (early − pre) / (late − pre)**: cosa succede dopo, con SER > 1 per il wash-out, ≈ 1 per il plateau e < 1 per la crescita continua;
   - **variazione tardiva = (late − early) / early**.
3. Salva le mappe in file `.npy`.

**Come**
- Gestisci le divisioni per zero e i valori negativi, per esempio escludendo i voxel con enhancement sotto una soglia minima, come nel trial I-SPY.
- Fai un controllo visivo su circa 10 pazienti: sovrapponi le mappe all'immagine e verifica che abbiano senso.
- Scrivi una sola funzione `process_patient(pid)` e lanciala in ciclo su tutti.

**Perché.** Queste mappe sono il "segnale biologico" (vascolarizzazione e permeabilità) su cui si costruisce tutto il resto. Usare misure relative come PE e SER, invece di intensità grezze, rende confrontabili pazienti esaminati con scanner diversi.

---

## Fase 3: Clustering degli habitat (settimane 2-3)

**Cosa fare.** Raggruppa i voxel in 2-5 habitat in base al vettore (PE, SER, variazione tardiva).

**Come**
- **Addestra il clustering a livello di popolazione, non paziente per paziente.** Campiona voxel dai pazienti del solo training set, standardizzali e addestra K-means (e GMM come confronto). Poi applica lo stesso modello a tutti i pazienti.
- Scegli k con il silhouette score (per K-means) e il BIC (per GMM), calcolati solo sul training set.
- Dai un nome agli habitat in base alle medie, per esempio "alta perfusione con wash-out" o "bassa perfusione".

**Perché.** Con un clustering diverso per ogni paziente, "habitat 1" significherebbe una cosa diversa da paziente a paziente e i confronti tra pazienti non avrebbero senso. Usare solo il training set evita che informazioni del test set entrino nel modello (data leakage), cosa che renderebbe i risultati falsamente ottimistici.

---

## Fase 4: Estrazione dei biomarcatori (settimana 3)

**Cosa fare.** Per ogni paziente calcola:
- la frazione di volume di ciascun habitat;
- l'**entropia** della distribuzione degli habitat;
- il numero di regioni connesse per ogni habitat, cioè quanto sono frammentate;
- un indice di adiacenza tra habitat, versione semplificata dell'approccio di Wu 2018;
- il volume del tumore (`tum_vol`, già nel CSV).

**Come.** Usa `scipy.ndimage.label` per le regioni connesse e conta i voxel confinanti tra habitat diversi. Poi elimina le feature troppo correlate tra loro (|r| > 0,9).

**Perché.** Queste sono le misure quantitative previste dalla proposta (§3.4). Pochi indicatori interpretabili sono più adatti all'analisi causale di centinaia di feature radiomiche.

---

## Fase 5: Modelli predittivi (settimana 4)

**Cosa fare.** Confronta tre modelli per la pCR:
- **M0**, solo clinico: HR, HER2, età;
- **M1**, clinico più habitat;
- **M2**, solo habitat.

**Come**
- Usa la regressione logistica come modello principale, perché è interpretabile; aggiungi una random forest come controllo.
- Addestra sul training set, regola gli iperparametri sul validation set e valuta sul test set una sola volta.
- Riporta AUC, accuratezza, sensibilità e specificità, con intervalli di confidenza al 95% ottenuti con il bootstrap. Confronta M0 e M1 con il test di DeLong.
- Ripeti l'analisi per ciascun sottotipo.
- Fai una validazione incrociata tra coorti: addestri su I-SPY2 e testi su I-SPY1.
- Confronta i risultati con l'AUC del ViT pubblicata da Fridman.

**Perché.** La domanda centrale è se l'eterogeneità aggiunge qualcosa al sottotipo. Solo il confronto M0 contro M1 risponde. Il test tra coorti misura quanto il modello regge su dati diversi, che è il punto debole emerso dalla MAMA-MIA Challenge.

**Checkpoint con il relatore.** Mostragli i risultati prima di passare all'analisi causale.

---

## Fase 6: Analisi causale (settimane 4-5)

**Cosa fare**
1. **DAG.** Disegna il grafo: sottotipo (HR, HER2) → habitat → pCR; sottotipo → pCR; età come confondente.
2. **Effetto aggiustato.** Con DoWhy stima l'effetto dell'eterogeneità sulla pCR, aggiustando per sottotipo ed età.
3. **Mediazione.** Stima l'effetto naturale diretto e indiretto: quanta parte dell'effetto del sottotipo sulla pCR passa attraverso gli habitat. Puoi usare il metodo di regressione, oppure `statsmodels.stats.mediation`.
4. **Robustezza**: test di confutazione di DoWhy (placebo treatment, random common cause, data subset), E-value e intervalli di confidenza con il bootstrap.

**Come**
- Definisci il DAG come stringa `networkx`/`gml` e passalo a `dowhy.CausalModel`. Segui i quattro passi della libreria: model, identify, estimate, refute.
- Lavora su tutta la coorte I-SPY. Qui non stai predicendo, stai stimando effetti, quindi non serve lo split.

**Perché.** È la parte "Causal AI" della proposta, riformulata in modo difendibile. Mette per iscritto le ipotesi (il DAG) e verifica quanto i risultati reggono se quelle ipotesi sono sbagliate. Nella mia ricerca è anche l'aspetto più nuovo del tuo lavoro.

---

## Fase 7: Scrittura e revisione (settimane 5-6, più il margine)

**Struttura consigliata**
1. Introduzione: contesto clinico e obiettivo.
2. Background: tumore al seno, NAC e pCR, DCE-MRI, radiomica ed eterogeneità, inferenza causale.
3. Stato dell'arte: i lavori di confronto individuati nella ricerca bibliografica.
4. Dati e metodi: Fasi 1-6.
5. Risultati.
6. Discussione e limiti: solo 3 fasi temporali, niente Duke, ipotesi del DAG.
7. Conclusioni e sviluppi futuri: Causal Forest sui bracci di I-SPY2.

**Come**
- Scrivi i metodi via via che lavori: alla fine di ogni fase, una o due pagine.
- Nelle ultime due settimane scrivi risultati, discussione e introduzione, in quest'ordine.
- Lascia la settimana di margine per le correzioni del relatore.

**Perché.** L'introduzione si scrive meglio quando conosci già i risultati. Il margine assorbe i ritardi tipici: download lenti, bug nel preprocessing, richieste di modifica.

---

## Rischi principali

- **Spazio su disco.** È il blocco più probabile: risolvilo nella Fase 0.
- **Ritardo nelle fasi 2-3.** Inizia da I-SPY1, che conta 172 pazienti e pesa 1,3 GB. Metti a punto tutta la pipeline lì, poi estendila a I-SPY2.
- **La parte causale dà risultati deboli.** Va bene così. Un effetto indiretto piccolo, con analisi di robustezza fatta bene, è un risultato legittimo e coerente con la letteratura.

