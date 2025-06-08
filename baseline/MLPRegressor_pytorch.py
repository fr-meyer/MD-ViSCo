# Standard library imports

# Third-party imports
import torch.nn as nn

class MLPRegressor(nn.Module):
    """PyTorch implementation of MLPRegressor"""
    def __init__(self, input_size, hidden_size=100, activation='relu', alpha=0.0001):
        super(MLPRegressor, self).__init__()
        
        # Define activation function
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
        # Define network architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            self.activation,
            nn.Linear(hidden_size, 1)
        )
        
        # L2 regularization strength
        self.alpha = alpha
        
    def forward(self, x):
        return self.layers(x)