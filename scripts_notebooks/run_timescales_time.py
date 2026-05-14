"""This script performs timescale analysis in the time domain for encoding & retrieval trials."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 

# import custom functions
from timescales_memory.settings import DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import timescales_acf_single_trials

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

# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = 0.2, tmax=1.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# interpolate bad channels, reset_bads: If True, remove the bads from info.
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False) 

# subset epochs based on task phase
epochs_enc = epochs_interpolated['Fixation Onset Enc']
epochs_ret = epochs_interpolated['Fixation Onset Ret']
epochs_all = epochs_interpolated[['Fixation Onset Enc','Fixation Onset Ret']]


## Split into: Encoding, Retrieval, All fixation periods
epochs_list = [epochs_enc, epochs_ret, epochs_all]
epochs_names = ['encoding', 'retrieval', 'fix_all']

# loop over epoch objects and names
for name, epoch in zip(epochs_names, epochs_list):
        
        acf_trials, rsq_trials = timescales_acf_single_trials(sub=sub, epochs=epoch, osc=False)

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
        df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', f'{name}', f'sub-{sub}_acf_params_{name}.csv'), sep = ',', header = True, index = False)

