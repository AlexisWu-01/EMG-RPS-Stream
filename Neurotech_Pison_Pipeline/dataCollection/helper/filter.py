from scipy.signal import butter, filtfilt
import numpy as np

def highpass_filter(data, cutoff, fs):
    """
    Highpass filter the data with a cutoff frequency of cutoff Hz. (default: 20 Hz)
    """
    b, a = butter(5, cutoff / (0.5 * fs), btype='highpass')
    return filtfilt(b, a, data)

def bandstop_filter(data, lowcut,highcut, fs=1000):
    """
    Bandstop filter the data within a certain frequency range. We use it to remove electrical noises.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='bandstop')
    return filtfilt(b, a, data)

def filter_channels(data, fs):
    """
    Filter the data with a highpass filter and a bandstop filter.
    """
    ret = []
    for d in data:
        d = highpass_filter(d, 20, fs)
        d = bandstop_filter(d, 58,62, fs)
        d = bandstop_filter(d, 118,122, fs)
        d = bandstop_filter(d, 178,182, fs)
        ret.append(d)
    
    return np.array(ret)
