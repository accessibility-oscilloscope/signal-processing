import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test_signal = [4, 5, 6, 7, 6, 5]


def normalize_signal(signal):
    sig_maxval = np.max(signal)
    sig_minval = np.min(signal)
    sig_norm = np.zeros(len(signal))
    for i in range(len(signal)):
        sig_norm[i] = (signal[i] - sig_minval) / (sig_maxval - sig_minval)
    print(sig_norm)

def remove_extrema(signal):



normalize_signal(test_signal)
