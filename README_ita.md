# Auto Insurance Fall
L'obiettivo di questo progetto di Machine Learning è stato quello di
costruire un modello predittivo che ci aiutasse ad identificare i clienti
con maggior rischio assicurativo. 

- Se un cliente ha presentato almeno un sinistro (TARGET_FLAG = 1)
- Se un cliente non ha mai avuto sinistri (TARGET_FLAG = 0)

Il progetto è stato suddiviso quanto segue:

1) `a_ingestion.py`: In questa fase i dataset vengono caricati ed esplorati.


2) `b_processing.py` : In questa fase i dataset vengono processati (i valori nulli vengono rimpiazzati in modo coerente, applicate tecniche di Label e One-Hot encoding, analisi della correlazione e importanza, dropping di features inutili, SMOTE).


3) `c_training.py` : In questa fase viene addestrato il modello sui dati di train e poi viene scelto il migliore a fronte di vari confronti tra varie tipologie di modelli e ottimizzazione degli iperparametri.


4) `d_evaluation.py` : In questa fase il modello viene valutato, stampati report, plot matrice di confusione e Curva ROC e tratte conclusioni.


5) `e_inference.py` : In questa fase vengono fatte le predizioni sul test set (che il modello non ha mai visto prima d'ora), viene generato un csv con le varie predizioni e stampato un report su di esse.


6) `main.py` : Questo file è utile per lanciare lo script che avvia tutti i processi descritti sopra, il comando da dare da terminale è : ' python main.py '


7) `requirements.txt` : Fondamentale per avviare corretamente lo script e singoli file py, installazione di tutte le dipendenze py con comando : ' pip install -r requirements.txt'

L'applicazione è stata anche distribuita online attraverso Streamlit, ho costruito una semplice applicazione front-end che mostra metriche e risultati interattivi. 

L'applicazione è raggiungibile al seguente link : https://autoinsurancefall-fortim.streamlit.app/
