import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from b_processing import X_t, y_t
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


#Variabili globali
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

#function for th emain script
def train_model(X, y, output_model_path='model.pkl', test_size=0.2, random_state=42):
    """
    dopo aver confrontato le prestazioni dei vari modelli ne segue :
    addestramento del modello con  random forest e ottimizzazione degli iperparametri con GridSearchCV,
    valuta le performance su un validation set e salva il modello migliore.

    Input:
        X = feature di training.
        y = feature target di training.
        modello
        test_size= % dati utilizzati per la validazione
    Output:
        dizionario: contiene il modello migliore, l'importanza delle feature, X_val , y_val che userò per l'evaluation
    """
    try:
        # split dei dati in training e validation set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Definito modello Random Forest
        model = RandomForestClassifier(random_state=random_state, class_weight='balanced_subsample') #senza class_wight balanced -> ROC-AUC Score: 0.8112445919426574 , solo per 1 -> f1-score': 0.4911242603550296

        # Iperparametri da ottimizzare con GridSearchCV
        param_grid = {
            'n_estimators': [ 100,200,300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='f1')
        print("Addestramento del modello in corso...")
        grid_search.fit(X_train, y_train)

        # Miglior modello
        best_model = grid_search.best_estimator_

        # Salvare il modello
        joblib.dump(best_model, output_model_path)
        print(f"Model saved to {output_model_path}")

        feature_importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': best_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        print(feature_importances)

        return {
            'best_model': best_model,
            'feature_importances': feature_importances,
            'X_val':X_val,
            'y_val':y_val
        }

    except Exception as e:
        print(f"Error during model training and evaluation: {e}")
        raise


if __name__ == "__main__":

    # Splitting dei dati
    X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

    # Definizione degli algoritmi da confrontare
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
        "GradientBoosting": GradientBoostingClassifier(random_state=42), #risultati simili a random forest, ma piu tempo di processamento
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)  #risultati inferiori, minor tempo di processamento
    }

    # Parametri di ricerca per ogni modello
    param_grids = {
        "RandomForest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "GradientBoosting": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        },
        "LogisticRegression": {
            'C': [0.1, 1, 10],
            'solver': ['liblinear']
        }
    }

    # Iterazione sui modelli
    best_models = {}
    for model_name, model in models.items():
        print(f"\nAddestramento del modello: {model_name}")

        grid_search = GridSearchCV(estimator=model, param_grid=param_grids[model_name], cv=3, scoring='f1')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model

        # Salvataggio dei vari modelli
        model_filename = f"model_{model_name}.pkl"
        joblib.dump(best_model, model_filename)
        print(f"Model saved to {model_filename}")

        #Stampa delle metriche di valutazione
        y_val_pred = best_model.predict(X_val)
        print(f"\nPerformance del modello {model_name}:\n")
        print(classification_report(y_val, y_val_pred))

        feature_importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': best_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
        print(f"\nFeature Importances per {model_name}:\n")
        print(feature_importances)
