import torch

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    """
    Train the model.
    Args:
        model: PyTorch model to be trained
        train_loader: DataLoader for the training data
        criterion: Loss function
        optimizer: Optimization algorithm
        epochs: Number of training epochs
    """
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in train_loader:
            # Ensure inputs are tensors
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.float32)

            # Forward pass
            outputs = model(features)  # Do not use squeeze to change output dimensions
            # Make outputs and labels have the same shape
            outputs = outputs.view(-1)
            labels = labels.view(-1)

            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
