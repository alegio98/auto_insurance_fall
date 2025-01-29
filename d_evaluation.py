from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from c_training import X_val, y_val
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def evaluate_model(model_path, X_test, y_test):
    """
    valutazione del modello addestrato utilizzando il test set. Genera il report di classificazione,
    la matrice di confusione e la curva ROC-AUC.

    Input:
        model_path = modello salvato
        X_test = feature di test.
        y_test = target di test.

    Output:
        Risultati della valutazione
    """
    try:
        # Carico il modello addestrato
        model = joblib.load(model_path)

        # Predizioni
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        # Report di classificazione
        classification_report_output = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # ROC-AUC Score
        auc_score = roc_auc_score(y_test, y_pred_prob)
        print(f"ROC-AUC Score: {auc_score}")

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()

        # Risultati
        return {
            'classification_report': classification_report_output,
            'roc_auc_score': auc_score,
            'confusion_matrix': cm
        }

        print("Classification Report:", results['classification_report'])

    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise


# --- codice esterno per provare la funzione ---
if __name__ == "__main__":
    # Percorso del modello salvato
    model_path = "model.pkl"
    results = evaluate_model(model_path, X_val, y_val)

    print("Classification Report:", results['classification_report'])
    print("ROC-AUC Score:", results['roc_auc_score'])
