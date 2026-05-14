"""This script performs timescale analysis on the Evoked objects  """

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED
from timescales_memory.analyses import timescales_acf_evoked

# set system variables
sub = sys.argv[1]

# load cleaned data for subject
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
events[:,2] = events[:, 2] - 64512

# define keys and events 
keys = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target', 
        'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


event_dist = {k: EVENT_DICT[k] for k in keys}


# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = event_dist, tmin = 1.2, tmax=2.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# interpolate bad channels
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False)


# subselect epochs
epochs_high = epochs_interpolated['Encoding Stimulus Onset Distraction Right Target', 'Encoding Stimulus Onset Distraction Left Target']
epochs_low = epochs_interpolated['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']



# construct Evoked objects
evoked_high = epochs_high.average()
evoked_low = epochs_low.average()



# FIT TIMESCALES ON EVOKED OBJECTS (NO OSC)

# High vs. Low Distraction Split
evoked_list = [evoked_high, evoked_low]
evoked_names = ['high', 'low']

# loop over epoch objects and names
for name, evoked in zip(evoked_names, evoked_list):
        
        acf_trials, rsq_trials = timescales_acf_evoked(sub=sub, evoked=evoked, osc=False)

        # print their shape
        print(acf_trials.shape)
        print(rsq_trials.shape)

        # set channel index to match channel order of info object
        chs_idx = evoked.info['ch_names']

        # convert to pd.DataFrame
        df = pd.DataFrame(acf_trials, columns = [f"tau_{name}", f"height_{name}", f"offset_{name}"])

        df["chs"] = chs_idx
        df["sub"] = sub
        df[f'rsq_{name}'] = rsq_trials


        # save as csv file
        df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'evoked', 'distraction', f'sub-{sub}_acf_params_evoked_{name}.csv'), sep = ',', header = True, index = False)


