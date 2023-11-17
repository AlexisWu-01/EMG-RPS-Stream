import torch
import torch.nn as nn

class EMGClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128,32], num_classes=3, dropout_rate=0.2):
        super(EMGClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[1], num_classes)

        )
        
    def forward(self, x):
        return self.layers(x)
