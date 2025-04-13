import torch.nn as nn

class NeuralNetTorch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetTorch, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Ensure a single input x
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


# Define a model initialization method
def initialize_model(input_dim, hidden_dim=64, output_dim=1):
    """
    Initialize the model.
    """
    model = NeuralNetTorch(input_dim, hidden_dim, output_dim)
    return model
