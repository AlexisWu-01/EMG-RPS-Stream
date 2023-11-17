import joblib
import numpy as np
import pandas as pd
from helper.filter import filter_channels
from helper.feature_extraction import feature_extraction
from helper.onset import get_onset_data
from helper.NN import EMGClassifier
import matplotlib.pyplot as plt
import torch

class RunPythonModel:
    def __init__(self, modelPath):
        filename = modelPath.split('/')[-1].split('.')[0]
        if modelPath.endswith('pth'):
            self.model = EMGClassifier(input_size=48)
            self.model.load_state_dict(torch.load(modelPath))
            self.scaler = joblib.load(f'models/{filename}_scaler.pkl')
            self.NN = True
        else:
            self.model = joblib.load(modelPath)
            self.scaler = joblib.load(f'models/{filename}_scaler.pkl')
            self.NN = False

        if filename.startswith('selected'):
            self.select = True
        else:
            self.select = False

    def select_features(self,data, num_features=25):
        ranked_features = pd.read_csv('models/RF_feature_importance.csv',index_col=0)
        feature_set = ranked_features[:num_features].index.to_numpy()
        columns = [
            'ch1_wl', 'ch1_mav', 'ch1_ssc', 'ch1_zc', 'ch1_var', 'ch1_rms', 'ch1_mf','ch1_pf','ch1_activity','ch1_mobility','ch1_complexity','ch1_bp',
            'ch2_wl', 'ch2_mav', 'ch2_ssc', 'ch2_zc', 'ch2_var', 'ch2_rms', 'ch2_mf', 'ch2_pf','ch2_activity','ch2_mobility','ch2_complexity', 'ch2_bp',
            'ch3_wl', 'ch3_mav', 'ch3_ssc', 'ch3_zc', 'ch3_var', 'ch3_rms', 'ch3_mf', 'ch3_pf','ch3_activity','ch3_mobility','ch3_complexity', 'ch3_bp',
            'ch4_wl', 'ch4_mav', 'ch4_ssc', 'ch4_zc', 'ch4_var', 'ch4_rms', 'ch4_mf', 'ch4_pf','ch4_activity','ch4_mobility','ch4_complexity'   , 'ch4_bp'
        ]
        feature_indices = [columns.index(x) for x in feature_set if x in columns]
        data = data[feature_indices]
        return data


    def get_rps(self, data):
        """
        Function to take in data and return the rps. You can
        place this function wherever you want, but add code here that
        takes in the data and returns rock, paper, or scissors
        after putting the data through your model.
        """
    
        data = np.array(data).T[1:-1]
        data = np.array(data).reshape(4,1400)
        filtered_data = filter_channels(data, 1000)
        onset_data = get_onset_data(np.array(filtered_data))
        # fig, ax = plt.subplots(3)
        # ax[0].plot(data[0])
        # ax[1].plot(filtered_data[0])
        # ax[2].plot(onset_data[0], color='r')
        # plt.show()
        features = feature_extraction(onset_data)
        if self.select:
            features = self.select_features(features)

        features = features.reshape(1,-1)
        features = self.scaler.transform(features)



        if self.NN:
            self.model.eval()
            with torch.no_grad():
                output = self.model(torch.from_numpy(features).float())
                pred = torch.argmax(output).item()
                print(f"guess score: {output}")
        else:
            pred = int(self.model.predict(features)[0])
        
        print(f"prediction: {['rock', 'paper', 'scissors'][pred]}")
        return int(pred+1)
