"""This script performs basic timescale analysis."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

# import useful 
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

# import aABC functions


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest

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

epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = None, reject_by_annotation=True, picks = 'eeg', on_missing="ignore")

epochs_fix_enc = epochs['Fixation Onset Enc']
epochs_fix_ret = epochs['Fixation Onset Ret']


# compute evoked
evoked_fix_enc = epochs_fix_enc.average()
evoked_fix_ret = epochs_fix_ret.average()


# for aABC tau we need the data in the format of a numpy array (numTrials x time-points)
# so we should call the get_data method on the evoked objects

evoked_enc_data = evoked_fix_enc.get_data()
print('EVOKED AFTER CALLING GET_DATA', evoked_enc_data)

print('SHAPE', evoked_enc_data.shape) # shape (32, 3073); but maybe this is still not the correct format...
