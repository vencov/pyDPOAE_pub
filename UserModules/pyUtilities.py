from scipy.signal import butter, filtfilt
import numpy as np

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def rfft(x):
    """calculates half of signal spectrum (assumed that signal is real)"""
    if x.ndim==1:  # check whether x is 1D
        N = int(np.ceil(len(x)/2)) # half of the spectrum
        xc = np.fft.fft(x)
        X = 2*xc[0:N]/len(x)  # take first half of the spectrum and normalize
    else:
        print('Input must be a 1D array')
        X = None
    return X