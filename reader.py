import os
import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import pyeeg
from scipy.stats import kurtosis, skew
from scipy.signal import argrelextrema, welch
from scipy.integrate import cumtrapz
import statistics 
import time

def eeg_features(data):
    data = np.asarray(data)
    res  = np.zeros([22])
    Kmax = 5
    # M    = 10
    # R    = 0.3
    Band = [1,5,10,15,20,25]
    Fs   = 256
    power, power_ratio = pyeeg.bin_power(data, Band, Fs)
    f, P = welch(data, fs=Fs, window='hanning', noverlap=0, nfft=int(256.))       # Signal power spectrum
    area_freq = cumtrapz(P, f, initial=0)
    res[0] = np.sqrt(np.sum(np.power(data, 2)) / data.shape[0])                   # amplitude RMS
    res[1] = statistics.stdev(data)**2                                            # variance
    res[2] = kurtosis(data)                                                       # kurtosis
    res[3] = skew(data)                                                           # skewness
    res[4] = max(data)                                                            # max amplitude
    res[5] = min(data)                                                            # min amplitude
    res[6] = len(argrelextrema(data, np.greater)[0])                              # number of local extrema or peaks
    res[7] = ((data[:-1] * data[1:]) < 0).sum()                                   # number of zero crossings
    res[8] = pyeeg.hfd(data, Kmax)                                                # Higuchi Fractal Dimension
    res[9] = pyeeg.pfd(data)                                                      # Petrosian Fractal Dimension
    res[10] = pyeeg.hurst(data)                                                   # Hurst exponent
    res[11] = pyeeg.spectral_entropy(data, Band, Fs, Power_Ratio=power_ratio)     # spectral entropy (1.21s)
    res[12] = area_freq[-1]                                                       # total power
    res[13] = f[np.where(area_freq >= res[12] / 2)[0][0]]                         # median frequency
    res[14] = f[np.argmax(P)]                                                     # peak frequency
    res[15], res[16] = pyeeg.hjorth(data)                                         # Hjorth mobility and complexity
    res[17] = power_ratio[0]
    res[18] = power_ratio[1]
    res[19] = power_ratio[2]
    res[20] = power_ratio[3]
    res[21] = power_ratio[4]
    # res[22] = pyeeg.samp_entropy(data, M, R)             # sample entropy
    # res[23] = pyeeg.ap_entropy(data, M, R)             # approximate entropy (1.14s)
    return (res)


def eeg_preprocessing(file, seizures, epoch_length = 10, step_size = 1, start_time = 0):
    start = time.time()

    # reading in data 
    raw = mne.io.read_raw_edf(file)  

    # apply filterbank
    raw = raw.load_data().filter(l_freq=0.25, h_freq=25)    
    channels = raw.ch_names                                  # column names

    # Divide into epochs
    res = []
    while start_time <= max(raw.times) + 0.01 - epoch_length:  # max(raw.times) = 3600
        features = []
        start, stop = raw.time_as_index([start_time, start_time + epoch_length])
        temp = raw[:, start:stop][0]

        # start time as ID
        features.append(start_time)

        # features
        for i in range(23):
            features.extend(eeg_features(temp[i]).tolist())

        # seizure flag for y
        if filename in seizures:  # if file has seizure
            for seizure in seizures[filename]:
                if start_time > seizure[0] and start_time < seizure[1]:
                    features.append(1)
                elif start_time + epoch_length > seizure[0] and start_time + epoch_length < seizure[1]:
                    features.append(1)
                else:
                    features.append(0)
        else:    
            features.append(0)

        res.append(features)        
        start_time += step_size
        print("Section ", str(len(res)), "; start: ", start, " ; stop: ", stop)

    # formatting
    feature_names = ["rms", "variance", "kurtosis", "skewness", "max_amp", "min_amp", "n_peaks", "n_crossings", 
        "hfd", "pfd", "hurst_exp", "spectral_entropy", "total_power", "median_freq", "peak_freq", 
        "hjorth_mobility", "hjorth_complexity", "power_1hz", "power_5hz", "power_10hz", "power_15hz", "power_20hz"]

    column_names = ["start_time"]
    for channel in channels:
        for name in feature_names:
            column_names.append(channel + "_" + name)
    column_names.append("seizure")

    res = pd.DataFrame(res, columns=column_names)

    end = time.time()
    print("Finished preprocessing ", file, f" took {(end - start) / 60} minutes")
    return res


def eeg_visualize(raw, start_time, end_time):
    n = 2

    # MNE-Python's interactive data browser to get a better visualization
    raw.plot()

    # select a time frame
    start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
    temp, times = raw[:, start:stop]
    fig, axs = plt.subplots(n)
    fig.suptitle('Patient EEG')
    plt.xlabel('time (s)')
    plt.ylabel('MEG data (T)')
    for i in range(n):
        axs[i].plot(times, temp[i].T)
    plt.show()



####### Main
# file path here
folder = "data\seizure"
files = [file for file in os.listdir(folder) if file.endswith(".edf")]
print(files)

seizures =	{
    "chb02_16": [[130, 212]],
    "chb05_06": [[417, 532]], 
    "chb05_13": [[1086, 1196]],
    "chb05_16": [[2317, 2413]], 
    "chb05_17": [[2451, 2571]],
    "chb05_22": [[2348, 2465]],
    "chb08_02": [[2670, 2841]], 
    "chb08_05": [[2856, 3046]],
    "chb08_11": [[2988, 3211]], 
    "chb08_13": [[2417, 2577]],
    "chb08_21": [[2083, 2347]]
}

for filename in files:
    file = os.path.join(folder, filename)
    filename = os.path.splitext(filename)[0]
    res = eeg_preprocessing(file, seizures)
    res.to_csv(os.path.join("data", filename + '.csv'), index=False) 

print("done")



    

