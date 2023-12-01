""" 
Project: Neurotech Pison Rock Paper Scissors Gesture Recognition
Author: Alexis (Xinyi) Wu

This file contains functions to find the onset of the signal.
We use average moving window to smooth the signal, and then find the onset.
We use the onset to extract the data from the signal.
We find an onset for each channel, and then average the onsets.
Therefore, our model will only process the 0.6 seconds of the signal that contains the gesture after the onset starts.
"""
import numpy as np

def get_rms(data, frame_length, step):
    """
    Calculate the root mean square of the signal.
    """
    rms = []
    for i in range(0, len(data)-frame_length+1, step):
        rms.append(np.sqrt(np.mean(data[i:i+frame_length]**2)))
    return np.array(rms).squeeze()

def find_onset_index(rms):
    """
    Find the onset index of the signal.
    """
    threshold = 0.4
    for i in range(len(rms)):
        if rms[i] > threshold:
            return i
        
def find_onset_time(data, threshold, frame_length, step):
    """
    Transform the onset index to onset time.
    """
    rms = get_rms(data, frame_length, step)
    index = find_onset_index(rms)
    return index * step

def moving_average(rms):
    """
    Smooth the signal with a moving average.
    """
    w = 1
    return np.convolve(rms, np.ones(w), 'valid') / w

def normalize(rms):
    """
    Normalize the signal using min-max normalization.
    """
    return np.array([n/(max(rms)-min(rms)) for n in rms])

def get_onset(signal, frame_size, step, threshold=0.4):
    """
    Get the onset of the signal.
    When the signal is above the threshold, we find the onset.
    """
    rms = get_rms(signal, frame_size, step)
    rms = moving_average(rms)
    rms = normalize(rms)
    index = np.where(rms >threshold)[0][0]
    onset = index * step
    return onset

def average_onset(signal, frame_size, step, threshold=0.5):
    """
    Average the onset times of the signal from each channel so that we can get the onset of the gesture.
    """
    onsets = 0
    for i in range(4):
        onsets += get_onset(signal[i], frame_size, step)
    return int(onsets/4)

def get_onset_data(data):
    """
    Slice the data beginning from the onset for 0.6 seconds.
    If the onset is too late, we take the last 0.6 seconds of the signal.
    """
    onset = average_onset(data, 300, 100)
    if onset > 1400-600:
        onset = 800
    return data[:, onset:onset+600]
