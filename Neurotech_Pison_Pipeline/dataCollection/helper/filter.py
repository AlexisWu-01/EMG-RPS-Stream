from scipy.signal import butter, filtfilt
import numpy as np

def highpass_filter(data, cutoff, fs):
    b, a = butter(5, cutoff / (0.5 * fs), btype='highpass')
    return filtfilt(b, a, data)

def bandstop_filter(data, lowcut,highcut, fs=1000):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='bandstop')
    return filtfilt(b, a, data)

def filter_channels(data, fs):
    ret = []
    for d in data:
        d = highpass_filter(d, 20, fs)
        d = bandstop_filter(d, 58,62, fs)
        d = bandstop_filter(d, 118,122, fs)
        d = bandstop_filter(d, 178,182, fs)
        ret.append(d)
    
    return np.array(ret)
