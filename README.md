# Auto Insurance Fall

The goal of this Machine Learning project is to build a predictive model to help identify customers with a higher insurance risk.

- If a customer has filed at least one insurance claim (TARGET_FLAG = 1)
- If a customer has never filed an insurance claim (TARGET_FLAG = 0)

The project is divided into the following steps:

1) `a_ingestion.py`: _In this phase, the datasets are loaded and explored._
2) `b_processing.py` : _In this phase, the datasets are processed (missing values are replaced consistently, Label Encoding and One-Hot Encoding are applied, correlation and feature importance are analyzed, not revelant features are dropped, SMOTE is used)._ 
3) `c_training.py` : _In this phase, the model is trained on the training data, and the best model is selected based on comparisons between different models and hyperparameter optimization._
4) `d_evaluation.py` : _In this phase, the model is evaluated, reports are generated, the Confusion Matrix and ROC Curve are plotted, and conclusions are drawn._
5) `e_inference.py` : _In this phase, predictions are made on the test set (which the model has never seen before), a CSV file with the predictions is generated, and a report is printed._
6) `main.py` : _This file is used to run the script that executes all the processes described above. The command to run it from the terminal is: ' python main.py '_
7) `requirements.txt` : _Essential to correctly run the script and individual Python files. Install all dependencies using the command: ' pip install -r requirements.txt'_

The application has also been deployed online using Streamlit. A simple front-end application has been built to display metrics and interactive results.

The application is available at the following link: https://autoinsurancefall-fortim.streamlit.app/

