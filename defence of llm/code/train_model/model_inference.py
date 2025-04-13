import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, test_loader):
    """
    Evaluate the model performance and calculate multiple evaluation metrics.
    """
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features).squeeze()
            predictions = (outputs > 0.5).int()  # Threshold of 0.5, predictions greater than 0.5 are classified as positive

            # Store predictions and true labels in lists
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    # Convert to NumPy arrays
    import numpy as np
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=1)
    recall = recall_score(all_labels, all_predictions, zero_division=1)
    f1 = f1_score(all_labels, all_predictions, zero_division=1)
    auc = roc_auc_score(all_labels, all_predictions)

    # Print evaluation results
    print('Model Evaluation Results')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    return accuracy, precision, recall, f1, auc


def predict(model, features):
    """
    Use the trained model to make predictions.
    """
    model.eval()
    with torch.no_grad():
        # Ensure input is a tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)

        outputs = model(features).squeeze()
        predictions = (outputs > 0.5).int()  # Probability > 0.5 is classified as 1
        return predictions

