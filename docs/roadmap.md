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

**Decisioni prese** (script `scripts/download_data.sh` e `src/00_check_setup.py`, riepilogo in `outputs/check_setup_report.txt`)
- **Dataset**: BreastDCEDL, versione **MinCrop**, Zenodo record **18114231**, licenza CC BY-NC 4.0 (Fridman et al. 2026). Metadati in `BreastDCEDL_metadata_min_crop.csv`: 2.070 righe e 29 colonne (I-SPY1 172, I-SPY2 982, Duke 916), conteggi coincidenti con il paper.
- **Scaricamento**: metadati e demo l'11/11/2026, I-SPY1 il 12/11/2026, con estrazione in streaming (l'archivio non viene salvato). I-SPY2 non è ancora scaricata.
- **Software** (da citare nei metodi): Python 3.14.0, numpy 2.5.4, scipy 1.18.1, pandas 3.1.1, nibabel 5.5.0, scikit-learn 1.9.1, statsmodels 0.15.0, dowhy 0.8, matplotlib 3.11.2. `requirements.txt` non fissa le versioni, quindi vanno ricontrollate se l'ambiente viene ricreato.
- **Riproducibilità**: tutte le estrazioni casuali usano il seme fisso 0 (Fase 3). Gli output in `outputs/` non sono versionati, ma si rigenerano lanciando gli script in ordine.

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

**Decisioni prese** (script `src/01_cohort.py`, riepilogo in `outputs/cohort_report.txt`)
- **Criteri di inclusione**: I-SPY1 + I-SPY2 (1.154) → pCR disponibile (1.154) → HER2 disponibile (1.149; esclusi ISPY1_1142, 1144, 1159, 1173, 1192) → fasi early e late distinte (**1.148**; escluso ISPY1_1139, che ha early = late = acquisizione 2, quindi SER e variazione tardiva non sarebbero calcolabili). Coorte finale: 166 I-SPY1, 982 I-SPY2.
- **Split** (colonna `test`): 0 = train, 1 = test, 2 = validation → 884 / 132 / 132. La mappatura è confermata dai conteggi del paper sul dataset completo (1.532 / 268 / 270, tutti coincidenti).
- **Età**: mancante per 3 pazienti I-SPY2 (ISPY2-111004, ISPY2-149654, ISPY2-282082, tutte nel train). La colonna `age` resta originale, `age_imputed` usa la mediana del solo training set (49 anni) e `age_was_imputed` segnala i casi. La Tabella 1 riporta l'età senza imputazione.
- **Menopausa**: solo descrittiva (Tabella 1, con i mancanti). Non entra nei modelli né nel DAG: manca per tutta I-SPY1 e per 38 pazienti I-SPY2. Codifica 0 = pre, 1 = post, verificata con l'età (mediane 43 e 57 anni).
- **Sottotipo**: 3 categorie come sottotipo principale (HR+/HER2−, HER2+, triplo negativo, come `HR_HER2_STATUS`), più il dettaglio a 4 categorie (HER2+ diviso in HR+/HER2+ e HR−/HER2+).
- **Tabella 1**: mediana [IQR] per le variabili continue, n (%) per le categoriche, mancanti come riga esplicita, nessun p-value. File completi in `outputs/table1.md` e `outputs/pcr_by_subtype.csv`. Numeri principali (totale, poi I-SPY1 / I-SPY2):
  - età 49,0 [41,0–56,0] anni (48,7 / 49,0);
  - HR+ 54,9%, HER2+ 25,8% (32,5% / 24,6%);
  - sottotipo HR+/HER2− 39,2%, HER2+ 25,8%, triplo negativo 35,0% (in I-SPY1: 41,6%, 32,5%, 25,9%);
  - volume tumorale 14,7 [6,6–31,3] cm³;
  - **pCR 31,5%** (27,7% / 32,2%; train 31,2%, val 32,6%, test 32,6%);
  - **pCR per sottotipo** (totale): HR+/HER2− 15,8%, HER2+ 45,9%, triplo negativo 38,6%. Nel dettaglio a 4 categorie: HR+/HER2+ 35,6%, HR−/HER2+ 62,1%. In I-SPY1: HR+/HER2− 10,1%, HER2+ 46,3%, triplo negativo 32,6%.
- **Verifiche sui dati**: `tum_vol` è in **cm³**. Sui pazienti demo I-SPY il PE minimo nella maschera è esattamente 0,70, quindi le maschere contengono solo tessuto che capta contrasto.
- **File I-SPY1** (scaricati il 12/11/2026 in `data/ispy1/BreastDCEDL_ISPY1_min_crop/`, 172 maschere e 516 fasi): tutte le 166 pazienti della coorte hanno la maschera e le 3 fasi indicate dal CSV, con shape coerenti. I-SPY2 non è ancora verificata.
- **Tumore tagliato dal ritaglio MinCrop**: nuova colonna `mask_frac_in_crop` in `cohort.csv`, pari al volume della maschera nel ritaglio / `tum_vol` (4 decimali); < 1 = flag. In I-SPY1 sono segnalati 6 pazienti: ISPY1_1120 (0,59), 1205 (0,72), 1025 (0,84), 1035 (0,92), tagliati in z (la maschera tocca la prima e l'ultima fetta del ritaglio); ISPY1_1064 (0,999), tagliato in z ma in modo trascurabile; ISPY1_1103 (0,9996), che non tocca i bordi, con una differenza di circa 35 voxel di origine non chiara. **Restano nella coorte**; il volume si prende sempre da `tum_vol`; nelle Fasi 4-5 le analisi principali si ripetono escludendoli (analisi di sensibilità, soglia da decidere).
- **Bozza del DAG** (`src/dag.py` → `outputs/dag.gml` per DoWhy e `outputs/dag_bozza.png` per il relatore). 7 nodi, 17 frecce:
  - esposizione: sottotipo (HR, HER2); esito: pCR;
  - mediatori in sequenza: **volume tumorale → habitat** (sottotipo → volume → habitat → pCR; volume → pCR). Così si può vedere se gli habitat aggiungono qualcosa oltre alla dimensione del tumore;
  - confondenti da aggiustare: **età** (→ sottotipo, volume, habitat, pCR) e **coorte** I-SPY1/I-SPY2 (→ sottotipo per i criteri di arruolamento, → volume per la dimensione minima richiesta, → habitat per scanner e protocollo, → trattamento, → pCR), più un'analisi di sensibilità sulla sola I-SPY2;
  - **trattamento come nodo non osservato** (sottotipo → trattamento → pCR; coorte → trattamento). Non è un genitore degli habitat, che sono misurati prima della terapia. Limite: l'effetto "diretto" del sottotipo comprende anche quello della terapia.
  - *Punti aperti per la Fase 6*: (a) il volume dipende dal sottotipo e influenza sia gli habitat sia la pCR, quindi la mediazione va impostata con mediatori multipli in sequenza (effetto via volume ed effetto via habitat oltre il volume) e non con un singolo mediatore; (b) come codificare l'esposizione, visto che il sottotipo ha 3 categorie (confronti a coppie o HR e HER2 separati).

**Checkpoint con il relatore.** Mostragli la coorte e una prima bozza del DAG, e fagli approvare la riformulazione della parte causale (§3.6 della proposta). *Non fatto, per scelta dell'utente (13/11/2026).*

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

**Decisioni prese** (script `src/02_maps.py`, test `tests/test_02_maps.py`, riepilogo in `outputs/maps_report.txt` e `outputs/maps_qc.csv`)
- **Voxel validi = maschera e PE ≥ 0,70** (definizione FTV del trial, Hylton 2012), per analizzare lo stesso tipo di tessuto in tutte le pazienti. Le tre mappe sono comunque calcolate su tutta la maschera e salvate, così un'analisi di sensibilità "tutta la maschera" non richiede di ricalcolarle.
- **Rapporti non calcolabili = NaN**, mai 0 o un valore di comodo: PE e variazione tardiva se pre ≤ 0, SER se late − pre ≤ 0. Un voxel con SER NaN resta valido per le altre mappe.
- **Nessun taglio dei valori estremi in Fase 2**: si decide in Fase 3, insieme al clustering.
- **Esclusione per problemi di acquisizione: regola oggettiva**, decisa dopo il controllo visivo e senza guardare la pCR. Una paziente è esclusa (colonna `exclude` in `maps_qc.csv`) in due casi:
  - il rapporto early/pre o late/pre dell'**intensità mediana del tessuto fuori dal tumore** esce da **[0,9; 2,5]**. Il "fondo" è il tessuto a più di 5 voxel dalla maschera, con pre sopra la mediana del volume (per togliere l'aria). Se cambia la luminosità di un'intera fase, il problema è il guadagno o la serie acquisita, non la biologia. Nella coorte questi rapporti hanno l'1°–99° percentile in 0,93–1,97;
  - meno del **50% dei voxel** della maschera è valido.

  Le mappe delle escluse restano salvate, per l'analisi di sensibilità. Le soglie vengono dalla distribuzione di I-SPY1 e **vanno ricontrollate su I-SPY2**.
  Il flag informativo (> 10% della maschera con PE < 0,70 o con late ≤ pre) resta solo come segnalazione.
- **Formato**: un `.npz` compresso per paziente in `outputs/maps/`, ritagliato sul riquadro della maschera, in float32. Contiene `pe`, `ser`, `late`, `valid`, `reason`, la posizione del ritaglio, la shape originale e lo spacing. Per I-SPY1 occupa 47 MB in tutto.
- **Tolleranza ±5%** sulla variazione tardiva: solo per le frazioni descrittive (wash-out, plateau, crescita continua).
- **Verifiche sui file**: gli assi sono **(z, y, x)**. L'header NIfTI dichiara uno spacing di 1×1×1 mm, che è falso: lo spacing va preso dal CSV (`slice_thick`, `xy_spacing`).

**Risultati su I-SPY1** (166 pazienti, 4.082.057 voxel nelle maschere)
- Voxel validi: 3.958.905 (97,0%). Le 123.152 esclusioni per PE < 0,70 riguardano 17 pazienti; 149 pazienti hanno il 100% dei voxel validi. Quindi la verifica "PE minimo = 0,70" fatta in Fase 1 sui pazienti demo non vale per tutta la coorte.
- SER non calcolabile nello 0,19% dei voxel validi (7.647 voxel, 7 pazienti).
- Sui voxel validi: PE con mediana 1,80 (p1–p99: 0,74–5,19, massimo 61); SER con mediana 1,01 (p1–p99: 0,46–3,20, massimo 657). Forma della curva: 40,5% wash-out, 20,1% plateau, 39,4% crescita continua.
- **Problemi di acquisizione** (controllo visivo e rapporto tra le fasi dell'intensità mediana del tessuto fuori dal tumore, che nella coorte ha mediana 1,31 per early/pre e 1,35 per late/pre):
  - ISPY1_1156: la fase early non mostra enhancement (è uguale alla pre), che compare solo nella late. Restano validi 380 voxel su 83.854;
  - ISPY1_1148: late più scura in tutta l'immagine (fondo late/pre = 0,49);
  - ISPY1_1193: late satura (fondo late/pre = 13,4). Non segnalata dalla regola sui soli voxel tumorali;
  - ISPY1_1042: fondo late/pre = 0,80 (late/early = 0,54);
  - ISPY1_1069 e ISPY1_1077: early più scura della pre nel fondo (0,67 e 0,85), quindi PE sottostimato;
  - ISPY1_1228: nessuna anomalia del fondo. I voxel con PE < 0,70 stanno su pelle e vasi inclusi nella maschera, e la soglia li esclude.
- **Esito della regola**: escluse **6 pazienti** (1042, 1069, 1077, 1148, 1156, 1193), le stesse individuate a vista; ISPY1_1228 resta inclusa. **Coorte I-SPY1 per gli habitat: 160 pazienti**. Nelle incluse:
  - voxel validi: 99,05%;
  - SER mediano per paziente tra 0,65 e 1,39 (prima dell'esclusione arrivava a 11,1);
  - variazione tardiva mediana per paziente tra −0,20 e +0,26;
  - forma della curva: 40,5% wash-out, 20,3% plateau, 39,2% crescita continua.
- **Limite del controllo sul fondo**: in circa 15 pazienti le intensità del fondo sono interi piccoli (mediana 4–16), quindi il rapporto procede a scatti (per esempio 4→5 = 1,25). Le due esclusioni ai margini (1069 e 1077) sono confermate anche dalla media (0,74 e 0,82). La media però non è stata scelta perché sensibile agli estremi: segnalerebbe ISPY1_1187, che al controllo visivo è normale (un tumore grande ed eterogeneo).
- *Punti aperti per la Fase 3*:
  - (a) in alcune pazienti normali (per esempio ISPY1_1107) si vede un bordo di falso wash-out, probabilmente dovuto a un leggero movimento tra early e late (le fasi non sono registrate tra loro). Da valutare, per esempio erodendo la maschera di un voxel;
  - (b) **le tre mappe contengono solo due informazioni indipendenti**. Vale infatti variazione tardiva = PE/(1+PE) · (1/SER − 1), perché tre immagini, una volta normalizzate rispetto alla pre, lasciano due gradi di libertà. Da decidere quali variabili dare al clustering (per esempio PE e SER, oppure tutte e tre).
- **Per la scrittura (Fase 7), cosa sono le mappe**: sono misure **semi-quantitative**, non grandezze fisiche assolute. Sono rapporti di intensità di segnale, adimensionali e dipendenti dal protocollo (tempi delle fasi, sequenza). Non misurano la concentrazione del mezzo di contrasto, né parametri farmacocinetici come Ktrans: con circa 2 minuti tra le fasi non sono stimabili (Hylton 2012). Sono però indicatori indiretti della fisiologia:
  - **PE** riflette quanto contrasto arriva in fretta, cioè perfusione, densità dei vasi e permeabilità;
  - **SER** e **variazione tardiva** riflettono cosa succede dopo: il wash-out, tipico dei vasi tumorali immaturi e permeabili, contro l'accumulo continuo, tipico di tessuto più stromale o fibrotico;
  - essendo rapporti rispetto alla pre, si annulla la scala di intensità propria di ogni esame, e questo li rende confrontabili tra pazienti e scanner.

  Servono come "coordinate" di ogni voxel per il clustering degli habitat (Fase 3).

---

## Fase 3: Clustering degli habitat (settimane 2-3)

**Cosa fare.** Raggruppa i voxel in 2-5 habitat in base al vettore (PE, SER, variazione tardiva).

**Come**
- **Addestra il clustering a livello di popolazione, non paziente per paziente.** Campiona voxel dai pazienti del solo training set, standardizzali e addestra K-means (e GMM come confronto). Poi applica lo stesso modello a tutti i pazienti.
- Scegli k con il silhouette score (per K-means) e il BIC (per GMM), calcolati solo sul training set.
- Dai un nome agli habitat in base alle medie, per esempio "alta perfusione con wash-out" o "bassa perfusione".

**Perché.** Con un clustering diverso per ogni paziente, "habitat 1" significherebbe una cosa diversa da paziente a paziente e i confronti tra pazienti non avrebbero senso. Usare solo il training set evita che informazioni del test set entrino nel modello (data leakage), cosa che renderebbe i risultati falsamente ottimistici.

**Decisioni prese** (script `src/03_habitats.py`, test `tests/test_03_habitats.py`; passo 1 in `outputs/habitat_k_selection.txt/.csv`, passo 2 in `outputs/habitats_report_k3.txt` e `outputs/habitats_report_k2.txt`)
- **Pazienti e voxel**: le 160 pazienti I-SPY1 con `exclude == False` (95 train, 32 val, 33 test), solo i voxel `valid` della Fase 2. Il modello si adatta **solo sul training**; val e test (e, solo per la sensibilità, le 6 escluse) ricevono gli habitat dal modello congelato. La pCR non viene mai letta.
- **Variabili: log(PE) e variazione tardiva.** Il SER è scartato perché, sul training, ha una correlazione di Spearman di −0,996 con la variazione tardiva (stessa informazione), ma esplode quando late ≈ pre (fino a 401, asimmetria 107) ed è NaN nello 0,19% dei voxel validi, cioè proprio in quelli con il wash-out più forte (late ≤ pre, variazione tardiva mediana −0,78). La variazione tardiva è sempre definita sui voxel validi e resta tra −1 e 2. Così il problema dei SER NaN non si pone: quei voxel restano nel clustering.
- **Estremi e scala**: logaritmo di PE (asimmetria da 1,6 a 0,3), poi taglio di ogni variabile ai **percentili 0,5 e 99,5** e **standardizzazione z** (media e DS). Tutti i parametri sono stimati sul campione del training e salvati nel modello: PE tagliato in [0,71; 5,28], variazione tardiva in [−0,47; 0,61].
- **Nessuna erosione della maschera nell'analisi principale.** Un'erosione di 1 voxel nel piano toglie il 38% dei voxel (44 pazienti su 160 ne perdono più della metà), una 3D il 58%, per via delle fette spesse 2-4 mm. Inoltre il bordo ha **meno** wash-out dell'interno (mediana per paziente 24% contro 44%; il bordo supera l'interno solo nel 5% delle pazienti), quindi il falso wash-out da movimento non è sistematico. La maschera erosa nel piano è salvata (`eroded2d`) per l'**analisi di sensibilità in Fase 4**. Con k = 3 l'erosione sposta le frazioni mediane di H2 −0,07 e H3 +0,06: il bordo è più spesso "crescita continua".
- **Campionamento**: **2.000 voxel per paziente** (tutti se ne ha meno; la più piccola ne ha 361), con seme fisso 0: 182.846 voxel. Ogni tumore pesa quasi allo stesso modo; con un campione globale la paziente più grande farebbe il 10% e le 5 più grandi il 29%.
- **Algoritmo**: **K-means** (n_init = 10) come principale, **GMM** a covarianza piena come confronto. Ogni voxel va al centroide più vicino.
- **Scelta di k (k = 2…6, solo training)**: si sono calcolati silhouette (3 sotto-campioni da 10.000 voxel), inerzia, stabilità (ARI tra K-means adattati su metà disgiunte delle pazienti, 10 ripetizioni) e BIC della GMM. La regola indicativa fissata prima era "il k più piccolo con silhouette alta e ARI ≥ 0,8".

  | k | silhouette | ARI medio (min) | BIC GMM | ARI K-means/GMM |
  |---|---|---|---|---|
  | 2 | 0,40 | 0,84 (0,66) | 997.839 | 0,85 |
  | 3 | 0,35 | 0,76 (0,60) | 986.487 | 0,72 |
  | 4 | 0,35 | 0,45 (0,35) | 980.866 | 0,38 |
  | 5 | 0,34 | 0,67 (0,45) | 981.275 | 0,71 |
  | 6 | 0,33 | 0,67 (0,55) | 981.146 | 0,79 |

  La silhouette scende in modo regolare e il gomito non è netto: lo spazio (PE, variazione tardiva) è **un continuo, senza gruppi naturali**, e gli habitat sono una suddivisione di questo continuo. Per questo pesano di più stabilità e interpretabilità. **Scelto k = 3 come analisi principale e k = 2 come analisi di sensibilità.** Ci si scosta dalla regola indicativa, che porterebbe a k = 2, per tre motivi (da dichiarare nella tesi):
  - k = 3 ritrova le tre forme di curva di Kuhl;
  - la sua stabilità (0,76) è appena sotto la soglia e molto sopra k ≥ 4;
  - con k = 2 ogni tumore sarebbe descritto da un solo numero, e l'entropia della Fase 4 ne sarebbe una funzione.
- **Alternative considerate e scartate** (per la discussione):
  - *PE + SER*, la coppia classica di Hylton, e *tutte e tre le mappe*: vedi sopra. Le tre insieme sarebbero ridondanti, perché la forma della curva conterebbe due volte rispetto a PE;
  - *solo taglio e z-score, senza logaritmo*: PE resta asimmetrico e la sua parte alta pesa di più nelle distanze;
  - *trasformazione sui ranghi*: rende uniformi le distribuzioni, i cluster tendono a diventare fasce di quantili e le distanze perdono significato;
  - *standardizzazione robusta (mediana/IQR) senza taglio*: gli estremi restano estremi e con K-means tirano i centroidi;
  - *campione casuale globale*: le pazienti conterebbero in proporzione al volume del tumore. *Tutti i voxel pesati 1/n per paziente*: equivalente al campione bilanciato, ma molto più lento;
  - *GMM come principale*: con circa 180 mila voxel che non seguono davvero una miscela di gaussiane, il BIC tende a premiare sempre più componenti (qui è piatto tra 4 e 6), quindi come criterio per k è poco affidabile. K-means assegna ogni voxel a un solo habitat ed è il metodo più usato negli studi sugli habitat;
  - *gomito*: non c'è un punto netto. *Silhouette massima*: premia k = 2;
  - *k = 4*: è il minimo del BIC e corrisponde ai quadranti perfusione alta/bassa × wash-out/crescita, ma ha la stabilità più bassa (ARI 0,45), perché esistono due divisioni quasi equivalenti tra cui il modello oscilla;
  - *nomi scelti a mano*: meno riproducibili se il clustering viene rifatto, per esempio con I-SPY2.
- **Ordine e nomi degli habitat**: numerati per PE crescente del centroide (a parità, per variazione tardiva). Il nome è dato da una regola:
  - perfusione "alta" o "bassa" rispetto al PE mediano del campione del training (1,52);
  - forma della curva dalla variazione tardiva mediana dell'habitat, con tolleranza ±5% (Kuhl).

  Il valore 0 nelle mappe indica i voxel non validi. I nomi prodotti dalla regola per k = 3 e k = 2 sono stati controllati e confermati.
- **Formato**: per ogni k ci sono:
  - `outputs/habitats_k<k>/<pid>.npz` (`habitat` uint8, `eroded2d`, stesso ritaglio e `spacing_zyx` della Fase 2);
  - `outputs/habitat_model_k<k>.npz` (parametri della trasformazione, centroidi, nomi, pazienti del training);
  - `outputs/habitat_fractions_k<k>.csv` (frazioni per paziente, intere ed erose);
  - le figure in `outputs/figures/habitats/k<k>/`.

**Risultati su I-SPY1, k = 3** (unità originali, campione del training)

| Habitat | Nome | Quota del campione | PE mediano [IQR] | Var. tardiva mediana [IQR] |
|---|---|---|---|---|
| H1 | perfusione bassa, plateau | 40% | 1,21 [1,00–1,46] | −0,01 [−0,08; 0,05] |
| H2 | perfusione bassa, crescita continua | 25% | 1,21 [0,95–1,62] | +0,21 [0,15; 0,29] |
| H3 | perfusione alta, wash-out | 35% | 2,46 [2,06–3,05] | −0,12 [−0,20; −0,03] |

- **Confronto con la GMM** (ARI 0,72): H1 e H3 coincidono al 97%. H2 viene diviso dalla GMM (64% / 31%), quindi il confine instabile è quello tra le due classi a bassa perfusione.
- **Frazioni per paziente** (160 incluse, mediana per split): sono simili tra train, val e test.

  | Habitat | Train | Val | Test |
  |---|---|---|---|
  | H1 | 0,33 | 0,21 | 0,24 |
  | H2 | 0,22 | 0,22 | 0,25 |
  | H3 | 0,27 | 0,29 | 0,26 |

  Gli habitat con almeno il 5% del tumore sono 3 in 94 pazienti, 2 in 65 e 1 in 1. H3 supera il 5% in 114 pazienti su 160: un quarto dei tumori ha poco o nessun tessuto "alta perfusione con wash-out".
- **Sensibilità k = 2**: H1 "perfusione bassa, crescita continua" (56%, PE 1,15, variazione tardiva +0,09) e H2 "perfusione alta, wash-out" (44%, PE 2,26, variazione tardiva −0,11). L'ARI con la GMM è 0,85.
- **Controllo visivo** su 6 pazienti di dimensioni diverse, compresa ISPY1_1107: gli habitat seguono le mappe di PE e variazione tardiva. Il bordo di falso wash-out di ISPY1_1107 cade in H3, che comunque copre la maggior parte di quel tumore.
- *Punti aperti per la Fase 4*:
  - ripetere le feature con la maschera erosa (`eroded2d`) e con k = 2 come sensibilità;
  - le frazioni sommano a 1, quindi una delle tre è ridondante nei modelli;
  - con I-SPY2 decidere se riadattare il modello sul training I-SPY1 + I-SPY2 e ricontrollare i parametri della trasformazione e la stabilità di k.

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

