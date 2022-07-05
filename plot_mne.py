import mne
import matplotlib.pyplot as plt
import os

def read_edf_to_raw(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.set_eeg_reference()
    raw.filter(l_freq=0.5, h_freq=45)
    return raw

################ INPUT YOUR DATASET PATH HERE
DATASET_PATH = os.path.join(os.getcwd(), 'data') #ONLY FOR TESTING

################ CHOOSE THE FILES THAT YOU WANT HERE
selected_files = {'chb01': ['03']}
files = [os.path.join(DATASET_PATH, folder, f"{folder}_{fn}.edf") for folder, fn_list in selected_files.items() for fn in fn_list]

for file_path in files:
    print(file_path)


# parameters for epochs generation
curr_time = 0
epoch_time = 10
overlap = 5

# Divide into epochs
dataset = []
for file_path in files:
    raw = read_edf_to_raw(file_path)
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_time, preload=True, overlap=overlap)

raw.plot(n_channels=30)
plt.show()