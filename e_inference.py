import pandas as pd
import joblib
from b_processing import X_test
import matplotlib.pyplot as plt
import seaborn as sns

def predict(model_path, test_data, output_path='predictions.csv'):
    """
    caricamente del modello addestrato e genera predizioni sul dataset di test.

    Input:
        model_path = percorso del modello salvato
        test_data = features preprocessate del test set.
        output_path = percorso del file CSV in cui salvare le predizioni.

    Output:
        risultati e csv dove sono presenti le predizioni
    """
    try:
        model = joblib.load(model_path)

        # Allineamento delle feature tra il modello e il dataset di test
        model_features = model.feature_names_in_  # Feature viste durante il training
        test_data_aligned = test_data.reindex(columns=model_features, fill_value=0)

        # Genera predizioni
        predictions = model.predict(test_data_aligned)
        predictions_prob = model.predict_proba(test_data_aligned)[:, 1]

        # Crea un DataFrame con i risultati
        results = pd.DataFrame({
            'Prediction': predictions,
            'Probability': predictions_prob
        })

        high_confidence = results[(results["Probability"] > 0.80) | (results["Probability"] < 0.20)]
        low_confidence = results[(results["Probability"] >= 0.48) & (results["Probability"] <= 0.53)]

        print(f" 1) Predizioni ad alta confidenza: {len(high_confidence)} su {len(results)} ({100 * len(high_confidence) / len(results):.2f}%)")
        print(f" 2) Predizioni incerte (tra 48% e 53%): {len(low_confidence)} su {len(results)} ({100 * len(low_confidence) / len(results):.2f}%)")

        # Salva le predizioni in un file CSV
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        return results

    except Exception as e:
        print(f"Error during inference: {e}")
        raise



if __name__ == "__main__":

    model_path = "model.pkl"
    output_path = "predictions.csv"
    results = predict(model_path, X_test, output_path)

    #Distribuzione delle probabilità delle predizioni
    plt.figure(figsize=(8, 5))
    sns.histplot(results["Probability"], bins=20, kde=True)
    plt.title("Distribuzione delle Probabilità delle Predizioni")
    plt.xlabel("Probabilità")
    plt.ylabel("Frequenza")
    plt.show()

    high_confidence = results[(results["Probability"] > 0.80) | (results["Probability"] < 0.20)]
    low_confidence = results[(results["Probability"] >= 0.48) & (results["Probability"] <= 0.53)]

    print(f" 1) Predizioni ad alta confidenza: {len(high_confidence)} su {len(results)} ({100 * len(high_confidence) / len(results):.2f}%)")
    print(f" 2) Predizioni incerte (tra 45% e 55%): {len(low_confidence)} su {len(results)} ({100 * len(low_confidence) / len(results):.2f}%)")

    print("Predictions:")
    print(results.head())
