import joblib
import numpy as np
from helper.filter import filter_channels
from helper.feature_extraction import feature_extraction
from helper.onset import get_onset_data
class RunPythonModel:
    def __init__(self, modelPath, scalerPath=None):
        self.model = joblib.load(modelPath)
        if scalerPath:
            self.scaler = joblib.load(scalerPath)
        else:
            self.scaler = None
    



    def get_rps(self, data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """
        filtered_data = filter_channels(data, 1000)
        onset_data = get_onset_data(np.array(filtered_data))
        features = feature_extraction(onset_data)
        if self.scaler:
            features = self.scaler.transform(features)
        pred = self.model.predict(features)
        return pred+1