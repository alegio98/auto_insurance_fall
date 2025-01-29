from a_ingestion import load_data
from b_processing import preprocess_data
from c_training import train_model
from d_evaluation import evaluate_model
from e_inference import predict

def main():
    train_file_path = 'train_auto.csv'
    test_file_path = 'test_auto.csv'
    model_output_path = 'model.pkl'
    predictions_output_path = 'predictions.csv'

    try:
        print("Step 1: Ingestion")
        train_data, test_data = load_data(train_file_path, test_file_path)

        print("\nStep 2: Processing")
        X_train, y_train, X_test, correlation_matrix, target_correlation = preprocess_data(train_data, test_data)

        print("\nStep 3: Training")
        training_results = train_model(X_train, y_train, output_model_path=model_output_path)
        X_val = training_results['X_val']
        y_val = training_results['y_val']

        print("\nStep 4: Evaluation on learning set")
        results = evaluate_model(model_output_path, X_val, y_val)
        print(results['classification_report'])

        print("\nStep 5: Inference")
        predict(model_output_path, X_test, output_path=predictions_output_path)

        print("Pipeline executed successfully.")
    except Exception as e:
        print(f"Error in pipeline execution: {e}")

if __name__ == "__main__":
    main()
