"""This script performs timescale analysis."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import seaborn
import fooof 

# import functions from timescales package
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw, run_epochs, compute_psd, timescales_enc_ret, timescales_acf_evoked, timescales_acf_evoked_np, timescales_psd_evoked_np

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

# demean - entire epoch!
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# crop epochs 
epochs_crop = epochs.copy().crop(tmin=0.2, tmax=1.1)

print("INFO OBJECT OF EPOCHS CROPPED INFO:", epochs_crop.info)

# TODO: compute drop log & save that information
epochs_fix_enc = epochs_crop['Fixation Onset Enc']
epochs_fix_ret = epochs_crop['Fixation Onset Ret']
epochs_fix_both = epochs_crop[['Fixation Onset Ret', 'Fixation Onset Enc']]

print(len(epochs_fix_both))

evoked_fix_enc = epochs_fix_enc.average()
evoked_fix_ret = epochs_fix_ret.average()
evoked_fix_all = epochs_fix_both.average()

# set ch_names
ch_names = epochs.info['ch_names']

# Fit timescales on evoked objects 
acf_fitted_wo, rsq_fitted_wo = timescales_acf_evoked_np(sub, evoked_fix_all, osc = False)  

# convert to pd.DataFrame
df_wo = pd.DataFrame(acf_fitted_wo, columns = ["tau", "height", "offset"])

# FIXME: this doesn't account for bad channels (but I guess it would make sense to compute timescales for all channels and then only later specify the bad ones)
df_wo['chs'] = ch_names

# add explained variance per ch
df_wo['rsq_all'] = rsq_fitted_wo
print(df_wo.head(5))

# add a column with tau values converted to milliseconds
df_wo['tau_ms_all'] = df_wo['tau'] * 1000

# add column with subject ID
df_wo['sub'] = sub

# compute grand average timescale
df_wo['grand_avg_all'] = df_wo['tau_ms_all'].mean()

# add columns with channel clusters (based on Wynn paper) - I guess having a parietal, frontal and occipital cluster makes sense
# first cluster: F1, Fz,  F2, FC1, FCz, FC2 (frontocentral cluster, based on Cellier; they used hierarchical clustering to identify them)
cluster_frontal = ['F1', 'Fz', 'FC1', 'FC2']

# second cluster: parietal-occipital
cluster_parietal = ['P3', 'P1', 'Pz', 'P2', 'P4', 'PO3', 'POz', 'PO4']


# compute average tau & avg rsq over frontal cluster
df_wo['avg_tau_frontal_all'] = (
    df_wo.loc[df_wo['chs'].isin(cluster_frontal), 'tau_ms_all']
    .mean()
)

df_wo['avg_rsq_frontal_all'] = (
    df_wo.loc[df_wo['chs'].isin(cluster_frontal), 'rsq_all']
    .mean()
)


# compute average tau & avg rsq over parietal-occipital cluster
df_wo['avg_tau_parietal_all'] = (
    df_wo.loc[df_wo['chs'].isin(cluster_parietal), 'tau_ms_all']
    .mean()
)

df_wo['avg_rsq_parietal_all'] = (
    df_wo.loc[df_wo['chs'].isin(cluster_parietal), 'rsq_all']
    .mean()
)

# re-order columns to make it more readable
df_reordered = df_wo.loc[:, ['sub', 'chs', 'tau', 'tau_ms_all', 'rsq_all', 'grand_avg_all', 'avg_tau_frontal_all', 'avg_tau_parietal_all', 'avg_rsq_frontal_all', 'avg_rsq_parietal_all', 'height', 'offset']]


# save as csv file
df_reordered.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_all.csv'), sep = ',', header = True, index = False)

## Print statstical summary of timescales
print(df_reordered['tau'].describe()) 
print(df_reordered['tau_ms_all'].describe())
print(df_reordered['rsq_all'].describe())

