import numpy as np
from scipy.signal import welch

def WL(data):
    wl = np.sum(np.abs(np.diff(data)))
    return wl / len(data)

def MAV(data):
    return np.sum(np.abs(data))/len(data)

def SSC(data,threshold):
    res = 0
    for i in range(1, len(data)-1):
        curr = (data[i]-data[i-1]) * (data[i+1]-data[i])
        if curr >= threshold:
            res += 1
    return res

def ZC(data):
    """
    Counts how many times the signal crosses zero. 
    """
    res = 0
    for i in range(1, len(data)):
        curr = data[i] * data[i-1]
        if curr < 0:
            res += 1
    return res

def VAR(data):
    """
    Measures the spread of signal values around the mean. It reflects the level of muscle activation.
    """
    return np.var(data)

def RMS(data):
    """
    Measures the amplitude of the signal.
    """
    return np.sqrt(np.mean(data**2))

def MF(data, sr):
    f, Pxx = welch(data, sr, nperseg=1024)
    cumulative_power = np.cumsum(Pxx)
    total_power = np.sum(Pxx)
    median_freq = np.interp(total_power/2, cumulative_power, f)
    return median_freq

def PF(data, fs):
    freqs, psd = welch(data, fs)
    peak_freq = freqs[np.argmax(psd)]

    return peak_freq

def calculate_hjorth_parameters(data):
    # Activity is the signal variance
    activity = np.var(data)

    # Mobility is the square root of the variance of the first derivative of the signal
    # divided by the activity
    mobility = np.sqrt(np.var(np.diff(data)) / activity)

    # Complexity is the mobility of the first derivative of the signal divided by the mobility
    complexity = np.sqrt(np.var(np.diff(data, n=2)) / np.var(np.diff(data)))

    return activity, mobility, complexity

def BP(data, fs, band=(20,450)):
    """Calculates the band power """
    freqs, psd = welch(data, fs, window='hann', nperseg=1024, scaling='density')
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    bp = np.trapz(psd[freq_mask], freqs[freq_mask])
    return bp


def feature_extraction(data):
    features = []
    for i in range(4):
        wl = WL(data[i])
        mav = MAV(data[i])
        ssc = SSC(data[i], 0.001)
        zc = ZC(data[i])
        var = VAR(data[i])
        rms = RMS(data[i])
        mf = MF(data[i], 1000)
        pf = PF(data[i], 1000)
        activity, mobility, complexity  = calculate_hjorth_parameters(data[i])
        bp = BP(data[i], 1000)
        features.extend([wl, mav, ssc, zc, var, rms, mf, pf, activity, mobility, complexity, bp])
    
    return np.array(features)