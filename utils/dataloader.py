#usr/bin/python3
import numpy as np
import mne

def calculate_neg_pos(y):
    unique, counts = np.unique(y, return_counts=True)
    dic = dict(zip(unique, counts))
    neg = dic[0]
    pos = dic[1]
    return neg, pos

def read_edf_to_raw(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='WARNING')
    raw.set_eeg_reference()
    raw.filter(l_freq=0.5, h_freq=45)
    return raw