"""This script performs timescale analysis in the frequency domain."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 

# import functions from timescales package
# from statsmodels.tsa.stattools import acf as compute_acf
# from neurodsp.spectral import compute_spectrum
# from neurodsp.spectral import compute_spectrum, trim_spectrum

# # make imports from timescales methods
# from timescales.conversions import knee_to_tau

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw, run_epochs, compute_psd, timescales_acf_evoked_np, timescales_psd_evoked_np, timescales_acf_single_trials

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
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# crop epochs 
epochs_crop = epochs.copy().crop(tmin=0.2, tmax=1.1)


# check bad channels
print("INFO OBJECT OF EPOCHS CROPPED INFO:", epochs_crop.info["bads"])

# exclude bad channels
chs_cleaned = [ch for ch in epochs_crop.ch_names if ch not in epochs_crop.info['bads']] 

epochs_crop = epochs_crop.copy().pick_channels(chs_cleaned)
print(epochs_crop.info)

# TODO: compute drop log & save that information
epochs_fix_enc = epochs_crop['Fixation Onset Enc']
epochs_fix_ret = epochs_crop['Fixation Onset Ret']
epochs_fix_all = epochs_crop[['Fixation Onset Enc','Fixation Onset Ret']]


evoked_fix_enc = epochs_fix_enc.average()
evoked_fix_ret = epochs_fix_ret.average()


# FIT TIMESCALES ON EVOKED OBJECTS (NO OSC)
acf_fitted, rsq_fitted = timescales_acf_evoked_np(sub, evoked_fix_enc, osc = False)

# convert to pd.DataFrame
df_wo = pd.DataFrame(acf_fitted, columns = ["tau", "height", "offset"])

# add channel names
# df_wo['ch_names'] = ch_names

# add explained variance per channel
df_wo['rsq'] = rsq_fitted
print(df_wo.head(5))

# save as csv file
df_wo.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc_without_osc.csv'), sep = ',', header = True, index = False)

# FIT TIMESCALES ON EVOKED OBJECTs WITH OSCILLATIONS
print('NOW FITTING WITH OSCILLATIONS\n')

# fit timescales on evoked objects 
acf_osc, rsq_osc = timescales_acf_evoked_np(sub, evoked_fix_enc, osc = True)

# convert to pd.DataFrame
df_osc = pd.DataFrame(acf_osc, columns = ["exp_tau", "osc_tau", "osc_gamma", "osc_freq", "amp_ratio", "height", "offset"])

# add channel names
# # df_osc['ch_names'] = ch_names

# add explained variance per channel
df_osc['rsq'] = rsq_fitted
print(df_osc.head(5))

# save as csv file
df_osc.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc.csv'), sep = ',', header = True, index = False)


