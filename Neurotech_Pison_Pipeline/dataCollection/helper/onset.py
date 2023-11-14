import numpy as np

def get_rms(data, frame_length, step):
    rms = []
    for i in range(0, len(data)-frame_length+1, step):
        rms.append(np.sqrt(np.mean(data[i:i+frame_length]**2)))
    return np.array(rms).squeeze()

def find_onset_index(rms):
    threshold = 0.4
    for i in range(len(rms)):
        if rms[i] > threshold:
            return i
        
def find_onset_time(data, threshold, frame_length, step):
    rms = get_rms(data, frame_length, step)
    index = find_onset_index(rms)
    return index * step

def moving_average(rms):
    w = 1
    return np.convolve(rms, np.ones(w), 'valid') / w

def normalize(rms):
    #min max scaler
    return np.array([n/(max(rms)-min(rms)) for n in rms])

def get_onset(signal, frame_size, step, threshold=0.4):
    rms = get_rms(signal, frame_size, step)
    rms = moving_average(rms)
    rms = normalize(rms)
    index = np.where(rms >threshold)[0][0]
    onset = index * step
    return onset

def average_onset(signal, frame_size, step, threshold=0.5):
    onsets = 0
    for i in range(4):
        onsets += get_onset(signal[i], frame_size, step)
    return int(onsets/4)

def get_onset_data(data):
    onset = average_onset(data, 300, 100)
    onset_data = []
    if onset > 1400-600:
        onset = 800
    return data[:, onset:onset+600]
