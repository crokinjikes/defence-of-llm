# defence-of-llm Using shap value for feature filtering

Data Loading & Preprocessing:
Load JSONL files and process them into feature matrices and labels using custom functions.

Initial Model Training:
Train a neural network model with all extracted features using BCELoss and the Adam optimizer for 30 epochs.

Initial Evaluation:
Evaluate the trained model on the test set to gauge baseline performance.

SHAP Analysis:
Use SHAP to analyze feature importance, identifying the contribution of each feature to model predictions.

Feature Selection:
Apply a voting mechanism on the SHAP values to select only the consistently important features.

Data Filtering:
Filter the original dataset to retain just the important features and reprocess the data into new DataLoaders.

Retraining:
Retrain a new model using the refined, filtered feature set.

Final Evaluation & Saving:
Evaluate the retrained model on test data and save the improved model for future use .

Model Evaluation Results Before Flitering
Accuracy: 0.9673
Precision: 0.7513
Recall: 0.8094
F1 Score: 0.7793
ROC-AUC: 0.8944


Model Evaluation Results After Flitering
Accuracy: 0.9709
Precision: 0.7923
Recall: 0.8011
F1 Score: 0.7967
ROC-AUC: 0.8925
