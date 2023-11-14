import joblib
import numpy as np
from helper.filter import filter_channels
from helper.feature_extraction import feature_extraction
from helper.onset import get_onset_data
import torch
import torch.nn as nn
class EMGClassifier(nn.Module):
    def __init__(self, input_size=44, hidden_sizes=[128,64,32], num_classes=3, dropout_rate=0.2):
        super(EMGClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            # nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], num_classes)
        )

    def forward(self, x):
        return self.layers(x)
class RunPythonModel:
    def __init__(self, modelPath):
        if modelPath.endswith('pth'):
            self.model = EMGClassifier()
            self.model.load_state_dict(torch.load(modelPath))
            self.scaler = joblib.load('models/scaler.pkl')
        else:
            self.model = joblib.load(modelPath)
            self.scaler = None



    def get_rps(self, data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """
        data = np.array(data).reshape(4,1400)
        filtered_data = filter_channels(data, 1000)
        onset_data = get_onset_data(np.array(filtered_data))
        features = feature_extraction(onset_data)
        if self.scaler:
            features = features.reshape(1,-1)
            features = self.scaler.transform(features)
            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.from_numpy(features).float())
                pred = torch.argmax(output).item()
        else:
            indices = [33, 22, 42, 0, 11, 39, 31,20,34,17,21,36,38,23,25,3,41,12,32,14]
            features = features[indices]
            features = features.reshape(1,-1)
            pred = self.model.predict(features)[0]
        return int(pred+1)