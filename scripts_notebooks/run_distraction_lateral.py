"""This script splits the encoding trials into left and right high and low distraction trials."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# import custom functions
from timescales_memory.settings import EVENT_DICT, DERIV_DIR, RAW_CLEANED, keys
from timescales_memory.analyses import timescales_acf_single_trials

# set system variables
sub = sys.argv[1]

# load cleaned data for subject
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False)
events[:,2] = events[:, 2] - 64512


print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# define keys and events 
keys = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target', 
        'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']

event_dist = {k: EVENT_DICT[k] for k in keys}

# construct & demean epochs
epochs = mne.Epochs(reconst_raw, events=events, event_id=event_dist, tmin = 1.2, tmax=2.1,baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# interpolate bad channels
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False)

# subselect epochs
epochs_left = epochs_interpolated['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Baseline Left']
epochs_right = epochs_interpolated['Encoding Stimulus Onset Distraction Right Target', 'Encoding Stimulus Onset Baseline Right']
epochs_high_dist_right = epochs_interpolated['Encoding Stimulus Onset Distraction Right Target']
epochs_high_dist_left = epochs_interpolated['Encoding Stimulus Onset Distraction Left Target']
epochs_low_dist_right = epochs_interpolated['Encoding Stimulus Onset Baseline Right']
epochs_low_dist_left = epochs_interpolated['Encoding Stimulus Onset Baseline Left']


## Lateralization Split
epochs_list = [epochs_right, epochs_left, epochs_high_dist_right, epochs_high_dist_left, epochs_low_dist_right, epochs_low_dist_left]
epochs_names = ["right", "left", "high_right", "high_left", "low_right", "low_left"]

# loop over epoch objects and names
for name, epoch in zip(epochs_names, epochs_list):
        
        acf_trials, rsq_trials = timescales_acf_single_trials(sub=sub, epochs = epoch, osc=False)

        # reshape 
        acf_reshaped = acf_trials.reshape(acf_trials.shape[0] * acf_trials.shape[1], acf_trials.shape[2])
        epochs_idx = np.repeat(np.arange(acf_trials.shape[0]), acf_trials.shape[1])
        
        chs_idx = np.tile(epochs.ch_names, acf_trials.shape[0])

        # convert to pd.DataFrame
        df = pd.DataFrame(acf_reshaped, columns = [f"tau_{name}", f"height_{name}", f"offset_{name}"])

        df["epoch"] = epochs_idx
        df["chs"] = chs_idx
        df["sub"] = sub

        # reshape rsq
        rsq_reshaped = rsq_trials.reshape(rsq_trials.shape[0] * rsq_trials.shape[1], rsq_trials.shape[2])
        df[f'rsq_{name}'] = rsq_reshaped


        # save as csv file
        df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'lateralization', f'sub-{sub}_acf_params_{name}.csv'), sep = ',', header = True, index = False)


