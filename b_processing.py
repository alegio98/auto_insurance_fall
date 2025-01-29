import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
#oop
from a_ingestion import train_data, test_data


#function for the main script
def preprocess_data(train_data, test_data, correlation_threshold=0.02):
    """
    - riempimento dati mancanti con forward fill
    - conversione colonne monetarie in numeri
    - label encoding e one-hot encoding
    - drop di features inutili dopo analisi

    Input:
        train_data = dataset training
        test_data = dataset test.

    Output:
        X_train = features processate, senza il target
        y_train = features target
        X_test = feartures processate di test
        correlation_matrix, target_correlation
    """

    # Riempire i valori mancanti
    train_data = train_data.ffill()
    #test_data = test_data.drop(['TARGET_FLAG', 'TARGET_AMT'], axis=1)
    test_data = test_data.ffill()

    # Identificare le colonne monetarie
    monetary_columns = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM']
    for column in monetary_columns:
        train_data[column] = train_data[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)
        test_data[column] = test_data[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)

    # Label Encoding per variabili con ordine naturale
    naturalOrder_value = ['PARENT1', 'MSTATUS', 'SEX', 'CAR_USE', 'RED_CAR', 'REVOKED', 'URBANICITY', 'EDUCATION']
    label_encoder = LabelEncoder()
    for value in naturalOrder_value:
        train_data[value] = label_encoder.fit_transform(train_data[value])
        test_data[value] = label_encoder.transform(test_data[value])

    # One-Hot Encoding per variabili senza ordine naturale
    new_categorical_columns = train_data.select_dtypes(include=['object']).columns
    train_data = pd.get_dummies(train_data, columns=new_categorical_columns)
    test_data = pd.get_dummies(test_data, columns=new_categorical_columns)

    # Allinea colonne tra train e test
    train_data, test_data = train_data.align(test_data, join='inner', axis=1)

    # Calcolo della correlazione

    correlation_matrix = train_data.corr()
    relevant_features = correlation_matrix['TARGET_FLAG'][abs(correlation_matrix['TARGET_FLAG']) >= correlation_threshold].index

    target_correlation = correlation_matrix['TARGET_FLAG'].sort_values(ascending=False)

    # Rimuovi TARGET_AMT (se presente) e filtra le feature
    relevant_features = [col for col in relevant_features if col != 'TARGET_AMT']
    train_data = train_data[relevant_features]


    # Separare feature e target
    X_train = train_data.drop('TARGET_FLAG', axis=1)
    y_train = train_data['TARGET_FLAG']

    X_test = test_data[relevant_features]
    #Bilanciamento delle features , troppi piu 0 che 1, analizzato in precedenza
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)



    return X_train_balanced, y_train_balanced, X_test, correlation_matrix, target_correlation

#Variabili globali
X_t, y_t, X_test, correlation_matrix, target_correlation = preprocess_data(train_data, test_data)

if __name__ == '__main__':
    print(train_data.isnull().sum())
    print(test_data.isnull().sum())

    print(train_data.duplicated().sum())  # 0
    print(test_data.duplicated().sum())  # 0

    print(train_data.shape)  # 8161 rows, 26 columns

    # Dataset Sbilanciato
    print("Distribuzione della variabile target:")
    target_counts = train_data['TARGET_FLAG'].value_counts(normalize=True) * 100
    print(f"0: {train_data['TARGET_FLAG'].value_counts()[0]} ({target_counts[0]:.2f}%)")
    print(f"1: {train_data['TARGET_FLAG'].value_counts()[1]} ({target_counts[1]:.2f}%)")

    train_data = train_data.ffill()
    test_data = test_data.ffill()

    train_object = train_data.select_dtypes(include=['object'])
    test_object = test_data.select_dtypes(include=['object'])

    print(train_object.info())
    print(train_object.head(2))


    unique = (train_object.nunique(axis=0))
    variability = pd.DataFrame(unique).sort_values(by=0, ascending=False)
    print(variability)  # controllo gli obj con piu varianza infatti sono degli interi camuffati in obj che sono INCOME, HOME_VAL, BLUEBOOK, OLDCLAIM

    monetary_columns = ['INCOME', 'HOME_VAL', 'BLUEBOOK', 'OLDCLAIM']
    for column in monetary_columns:
        test_data[column] = test_data[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)
        train_data[column] = train_data[column].replace('[\$,]', '', regex=True).replace('', '0').astype(float)

    label_encoder = LabelEncoder()

    naturalOrder_value = ['PARENT1', 'MSTATUS', 'SEX', 'CAR_USE', 'RED_CAR', 'REVOKED', 'URBANICITY', 'EDUCATION']

    for value in naturalOrder_value:
        train_data[value] = label_encoder.fit_transform(train_data[value])
        test_data[value] = label_encoder.fit_transform(test_data[value])

    new_categorical_columns = train_data.select_dtypes(
        include=['object']).columns  # a questo punto sono rimaste come colonne 'object' solamente CAR_TYPE e JOB

    train_data = pd.get_dummies(train_data, columns=new_categorical_columns)
    test_data = pd.get_dummies(test_data, columns=new_categorical_columns)

    print(train_data.shape)  # 8161 rows, 38 columns

    # train_data, test_data = train_data.align(test_data, join='inner', axis=1) #stesso numero di colonne in ordine corretto

    # analisi delle metriche graficamente per capire cosa e non droppare

    # Calcolo della correlazione tra feature numeriche e TARGET_FLAG

    correlation_matrix = train_data.corr()
    target_correlation = correlation_matrix['TARGET_FLAG'].sort_values(ascending=False)
    print("Correlazione con TARGET_FLAG:")
    print(target_correlation)

    """import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Heatmap delle Correlazioni")
    plt.show()"""

    # Droppa TARGET_AMT e seleziona feature con |correlazione| >= 0.02
    correlation_threshold = 0.02

    # Filtra le feature con correlazione significativa
    relevant_features = correlation_matrix['TARGET_FLAG'][
        abs(correlation_matrix['TARGET_FLAG']) >= correlation_threshold].index

    # Rimuovi a mano TARGET_AMT
    relevant_features = [col for col in relevant_features if col != 'TARGET_AMT']

    # Aggiorna il dataset X_t con le sole feature rilevanti
    X_train_filtered = train_data[relevant_features]

    test_data = test_data[relevant_features]

    print(f"Feature selezionate: {list(X_train_filtered.columns)}")
    print(f"Numero totale di feature selezionate: {len(X_train_filtered.columns)}")

    X_t = X_train_filtered.drop('TARGET_FLAG', axis=1)
    y_t = X_train_filtered['TARGET_FLAG']

