import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from a_ingestion import train_data
from b_processing import X_t, y_t, correlation_matrix, target_correlation, preprocess_data


# Plot confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

# Plot ROC curve
def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

st.title("Auto Insurance Classification")


st.subheader("Exploratory Data Analysis (EDA)")

st.subheader("Feature Types and Overview")
try:
    feature_overview = pd.DataFrame({
        "Data Type": train_data.dtypes,
        "Non-Null Count": train_data.notnull().sum(),
        "Unique Values": train_data.nunique(),
        "Most Frequent Value": train_data.mode().iloc[0].values,
    })
    st.write(feature_overview)

    num_rows, num_columns = train_data.shape
    st.markdown(f"**Train set Overview:** {num_rows} rows, {num_columns} columns")

except Exception as e:
    st.error(f"Error generating feature types table: {e}")


# Matrice di Correlazione
st.subheader("Correlation Matrix")
try:
    plt.figure(figsize=(30, 18))
    sns.heatmap(correlation_matrix, annot=True, fmt=".1F", cmap="plasma", alpha = 0.8, annot_kws={"size":12}, linewidths = 2.5)
    plt.title("Correlation Matrix")
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error generating correlation matrix: {e}")


st.subheader("Feature Correlation with TARGET_FLAG")
try:
    # Converti la correlazione in un DataFrame
    correlation_table = pd.DataFrame({
        "Feature Name": target_correlation.index,
        "Correlation with TARGET_FLAG": target_correlation.values
    }).reset_index(drop=True)

    # Mostra la tabella in Streamlit
    st.write(correlation_table)

except Exception as e:
    st.error(f"Error generating correlation table: {e}")


st.subheader("Descriptive Statistics")
try:
    descriptive_stats = train_data.describe().transpose()
    st.write(descriptive_stats)

except Exception as e:
    st.error(f"Error generating descriptive statistics: {e}")


st.subheader("Feature Distribution")
try:
    feature_to_plot = st.selectbox("Select a feature to visualize:", train_data.columns)
    plt.figure(figsize=(8, 6))
    sns.histplot(train_data[feature_to_plot].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {feature_to_plot}")
    plt.xlabel(feature_to_plot)
    plt.ylabel("Frequency")
    st.pyplot(plt)
except Exception as e:
    st.error(f"Error generating feature distribution plot: {e}")


try:
    # Carica il modello addestrato
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.subheader("Data Processed Preview")
try:
    # Mostra un'anteprima dei dati
    st.write("Features Preview:")
    st.write(X_t.head())

    st.write("Target Preview:")
    st.write(y_t.head())

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.stop()


st.subheader("Feature Importances")
try:
    # Usa le feature importances dal modello addestrato
    feature_importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    st.pyplot(plt)

except Exception as e:
    st.error(f"Error generating feature importance plot: {e}")

st.markdown("---")
st.subheader("Predictions and Metrics")
try:
    # Predizioni sul validation set
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    # Mostra le predizioni
    st.write("Predictions:")
    predictions_df = pd.DataFrame({"Prediction": y_val_pred, "Probability": y_val_prob})
    st.write(predictions_df.head())

    # Classification Report
    st.write("Classification Report:")
    report = classification_report(y_val, y_val_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)
    st.write("Confusion Matrix:")
    plot_confusion_matrix(cm)

    # ROC-AUC
    auc_score = roc_auc_score(y_val, y_val_prob)
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    st.write(f"ROC-AUC Score: {auc_score}")
    plot_roc_curve(fpr, tpr, auc_score)


except Exception as e:
    st.error(f"Error during predictions or metrics calculation: {e}")



st.subheader("Make a Prediction on a New Sample")

# Form per inserire i parametri
with st.form("prediction_form"):
    st.write("### Insert Feature Values")

    user_input = {}
    columns = st.columns(2)  # Organizzazione in due colonne

    # Creazione dell'input basato sui train_data
    for i, col in enumerate(train_data.columns):
        if col == "TARGET_FLAG" or col == "TARGET_AMT":
            continue  # Escludiamo il target e valori non utili per la predizione

        if train_data[col].dtype == 'object':  # Se categorico
            user_input[col] = columns[i % 2].selectbox(
                f"{col}:",
                options=train_data[col].dropna().unique(),  # Opzioni dai dati reali
                index=0  # Default: primo valore
            )
        else:  # Se numerico
            user_input[col] = columns[i % 2].number_input(
                f"{col}:",
                value=float(train_data[col].median()),  # Default: valore mediano
                step=0.01
            )

    # Bottone per avviare la predizione
    submit_button = st.form_submit_button("Predict")

# Se l'utente ha premuto "Predict"
if submit_button:
    try:
        # Convertiamo il dizionario in un DataFrame
        input_df = pd.DataFrame([user_input])

        input_df["TARGET_FLAG"] = 0  # Valore fittizio, sarà ignorato dopo il preprocessing,

        # Preprocessiamo il nuovo campione
        _, _, processed_sample, _, _ = preprocess_data(train_data, input_df)

        processed_sample = processed_sample.drop(columns=["TARGET_FLAG"], errors='ignore')

        # Allineamento delle colonne
        missing_cols = set(X_t.columns) - set(processed_sample.columns)
        extra_cols = set(processed_sample.columns) - set(X_t.columns)

        # **Aggiungiamo le feature mancanti con valore 0**
        for col in missing_cols:
            processed_sample[col] = 0

        # Rimuoviamo eventuali feature extra
        processed_sample = processed_sample[X_t.columns]


        # Predizione sul campione processato
        prediction = model.predict(processed_sample)[0]

        # Mostriamo il risultato in modo chiaro
        st.markdown("### Prediction Result:")
        st.success(f"Prediction: **{'Approved' if prediction == 1 else 'Denied'}**")

    except Exception as e:
        st.error(f"Error in prediction: {e}")

