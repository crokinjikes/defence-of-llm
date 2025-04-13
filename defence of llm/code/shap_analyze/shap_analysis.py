import shap
import torch
import numpy as np
import pandas as pd

def analyze_shap(model, X_train, features):
    """
    Perform SHAP value analysis on the trained model.
    Args:
        model: Trained PyTorch model
        X_train: Training data used for calculating SHAP values
        features: List of feature names used in the model
    """
    model.eval()

    # Use PyTorch's Deep Explainer for SHAP value analysis
    background = torch.tensor(X_train[:100], dtype=torch.float32)  # Use a subset of data as background data
    explainer = shap.DeepExplainer(model, background)

    # Convert the sample to a tensor
    sample = torch.tensor(X_train[:10], dtype=torch.float32)

    # Calculate SHAP values
    shap_values = explainer.shap_values(sample)

    shap_values_2d = shap_values[:, :, 0]
    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap_values = np.mean(np.abs(shap_values_2d), axis=0)  # Result shape is [n_features]

    # Create a DataFrame containing the mean absolute SHAP values for each feature
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'MeanAbsSHAP': mean_abs_shap_values
    })
    # Sort by MeanAbsSHAP values in descending order
    feature_importance_df = feature_importance_df.sort_values(by='MeanAbsSHAP', ascending=False).reset_index(drop=True)
    return feature_importance_df


def feature_selection(feature_importance_df):
    """
    Select features based on SHAP value threshold.
    Args:
        feature_importance_df: DataFrame containing feature importance values
    Returns:
        Series containing the selected features based on the threshold
    """
    # Set the threshold
    threshold = 0.001

    # Filter features with MeanAbsSHAP values greater than the threshold
    filtered_df = feature_importance_df[feature_importance_df['MeanAbsSHAP'] > threshold]
    shap_column = filtered_df.iloc[:, 0]

    return shap_column
