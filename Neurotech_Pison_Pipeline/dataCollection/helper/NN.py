import torch
import torch.nn as nn
import torch.optim as optim

class EMGClassifier(nn.Module):
    def __init__(self, input_size=48, hidden_sizes=[128,32], num_classes=3, dropout_rate=0.2):
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

input_size = 48
output_size = 3
model = EMGClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Function to perform live training update
def live_training_update(x_new, y_true, model=model, criterion=criterion, optimizer=optimizer):
    # Convert new input data and the correct label to tensors
    x_new = torch.tensor([x_new], dtype=torch.float)
    y_true = torch.tensor([y_true], dtype=torch.long)

    # Set model to training mode
    model.train()

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(x_new)

    # Compute loss
    loss = criterion(outputs, y_true.unsqueeze(0))

    # Backward pass
    loss.backward()

    # Update the weights
    optimizer.step()
    print(f"loss: {loss.item()}")

    return loss.item()


