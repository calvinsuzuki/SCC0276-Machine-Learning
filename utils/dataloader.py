#usr/bin/python3
import numpy as np
import mne
from sympy import false

def calculate_neg_pos(y):
    unique, counts = np.unique(y, return_counts=True)
    dic = dict(zip(unique, counts))
    neg = dic[0]
    pos = dic[1]
    return neg, pos

def read_edf_to_raw(file_path):
    mne.set_log_level(verbose='ERROR')
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.set_eeg_reference(verbose=False)
    raw.filter(l_freq=0.5, h_freq=45, verbose=False)
    return raw