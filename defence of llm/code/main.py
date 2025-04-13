import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from data_process.feature_extraction import load_data
from data_process.data_processing import process_dataframe, create_dataloaders
from data_process.data_processing import get_features_name
from train_model.model_definition import initialize_model
from train_model.model_training import train_model
from train_model.model_inference import evaluate_model, predict
from shap_analyze.shap_analysis import analyze_shap
from shap_analyze.shap_analysis import feature_selection


def main():
    # Step 1: Load and process data
    # Load data from a JSONL file and preprocess it for training and testing
    jsonl_file_train = r'../data/train_metrics_summary.jsonl'
    jsonl_file_test = r'../data/test_metrics_summary.jsonl'

    df_train = load_data(jsonl_file_train)
    df_test = load_data(jsonl_file_test)
    print("Data loaded successfully")
    # Split the data into training and testing sets, scale features, and extract target labels
    X_train, y_train = process_dataframe(df_train, target_col="toxicity")
    X_test, y_test = process_dataframe(df_test, target_col="toxicity")
    # Create DataLoader objects for batch processing during training and evaluation
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test)
    print("Data processing completed")

    # Step 2: Initialize the model
    # Define the input dimensions based on the training data and initialize the model
    input_dim = X_train.shape[1]
    model = initialize_model(input_dim)

    # Step 3: Train the model
    # Train the model using Binary Cross-Entropy loss and the Adam optimizer
    print('Training NeuralNetTorch model...')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_loader, criterion, optimizer, epochs=30)


    # Step 4: Save the trained model to file
    # Save the model to a specified path for future use
    model_path = "../model/NeuralNetTorch.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    # Step 5: Load the trained model for evaluation
    # Load the previously trained model from file and set it to evaluation mode
    model = torch.load("../model/NeuralNetTorch.pth")
    model.eval()

    # Step 6: Evaluate the model
    # Evaluate the model's performance on the test dataset
    evaluate_model(model, test_loader)

    # Step 7: Perform inference using the trained model
    # Use the trained model to make predictions on a sample of the test data
    test_sample = X_test[:2]  # Use the first two samples from the test data
    predictions = predict(model, test_sample)
    print("Predictions:", predictions)


    # Step 8: Analyze feature importance using SHAP
    # Analyze feature importance using SHAP values to understand model interpretability
    features = get_features_name(df_test,target_col="toxicity")
    importance_df = analyze_shap(model, X_train, features)
    print('All Feature Importance :', importance_df)
    important_features = feature_selection(importance_df)
    print(f"Important Features: \n{important_features}")

    # Step 9: Filter the data based on important features
    # Filter the original DataFrame to keep only the important features identified by SHAP
    filtered_df_train = df_train.loc[:, df_train.columns.isin(important_features)]
    filtered_df_test = df_test.loc[:, df_test.columns.isin(important_features)]

    # Add the target label column back to the filtered DataFrame
    filtered_df_train = pd.concat([filtered_df_train, df_train['toxicity']], axis=1)
    print(f"Filtered Train Features: \n{filtered_df_train}")
    filtered_df_test = pd.concat([filtered_df_test, df_test['toxicity']], axis=1)
    print(f"Filtered Test Features: \n{filtered_df_test}")
    # Step 10: Reprocess the filtered data
    # Process the filtered data and create new DataLoader objects for training and testing

    X_train, y_train = process_dataframe(filtered_df_train, target_col="toxicity")
    X_test, y_test = process_dataframe(filtered_df_test, target_col="toxicity")
    train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test)
    print("Data processing completed")

    # Step 11: Initialize and retrain the model using filtered features
    # Initialize a new model using the filtered feature set and train it again
    input_dim = X_train.shape[1]
    model = initialize_model(input_dim)
    print('Training NeuralNetTorchFiltered model...')
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, train_loader, criterion, optimizer, epochs=30)

    # Step 12: Evaluate the retrained model
    # Evaluate the model's performance on the test dataset after retraining
    evaluate_model(model, test_loader)

    # Step 13: Save the retrained model
    # Save the retrained model to a specified path for future use
    model_path = "../model/NeuralNetTorchFiltered.pth"
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
