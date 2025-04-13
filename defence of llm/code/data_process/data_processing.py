import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class ToxicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Data processing function
def process_dataframe(df, target_col):
    """
    Process the DataFrame, extract features and target variable, and standardize the features without splitting.
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
    Returns:
        X_train: Standardized feature matrix
        y_train: Target variable array
        scaler: Fitted StandardScaler object
        feature_name: List of feature names
    """
    # 提取特征和标签
    features = df.drop(columns=[target_col]).values
    labels = df[target_col].values

    # 标准化特征
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    y = labels

    return x, y


def get_features_name(df, target_col):

    feature_name = df.drop(columns=[target_col]).columns

    return feature_name
# Function to create DataLoaders


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    """
    Create training and testing DataLoader objects.
    Args:
        X_train, X_test: Features for training and testing
        y_train, y_test: Labels for training and testing
        batch_size: Number of samples per batch
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    train_dataset = ToxicDataset(X_train, y_train)
    test_dataset = ToxicDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
