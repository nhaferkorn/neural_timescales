"""This script aims to replicate the EEG results from Wynn et al. 2022."""

# make imports
import mne
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from datetime import datetime

# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED

# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")

# set system variables
sub = sys.argv[1]

# read cleaned data
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
events[:,2] = events[:, 2] - 64512

# define keys & events of interest
keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'}
events_of_interest = {k: EVENT_DICT[k] for k in keys}


## Replication of Syanah's analysis
# 1. Stimulus-locked epochs (−2000 to 2000 ms) were extracted for encoding trials
# baseline window: [-1.25 -1]; but this doesn't really make sense..
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin=-2.0, tmax = 2.0,  preload = True)

# interpolate bad channels
# TODO: I don't actually want to reset_bads in info object, find different way
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=True)

# subselection of epochs
epochs_attend_left = epochs_interpolated[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]
epochs_attend_right = epochs_interpolated[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]
print(epochs_attend_left)

# crop epochs to time window of: −750 to −250 ms
epochs_attend_left_cropped = epochs_attend_left.copy().crop(tmin=-0.75, tmax=-0.25)
print(epochs_attend_left_cropped)

freqs = np.arange(2,30,2)

n_cycles = freqs / 2

# Compute TRF: −750 to −250 ms were chosen for analysis 
power_tf_left = epochs_attend_left.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_tf_right = epochs_attend_right.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)

# Sanity checks 
# of type.'mne.time_frequency.tfr.AverageTFR' (n_channels, n_frequencies, n_time_points)
print(type(power_tf_left))
print(power_tf_left.shape)
print(power_tf_right.shape)

# compute differences
pmi_numerator = power_tf_left - power_tf_right
pmi_denominator = power_tf_left + power_tf_right

print(type(pmi_denominator))
print(type(pmi_numerator))

# compute power modulation index PMI: pmi_numerator / pmi_denominator
# okay, so right now the interpolation doesn't seem to have worked
pmi = pmi_numerator.get_data() / pmi_denominator.get_data()

# numpy array 
print('PMI Shape', type(pmi))


# FIXME: doesn't work
# # convert back into AverageTFR object (not sure if this actually works though)
# average_tfr = mne.time_frequency.AverageTFRArray(info=epochs_interpolated.info, data=pmi, times=pmi_denominator.times, freqs=freqs)
# print(average_tfr.shape)


## Interpretation: For each hemisphere, positive PMI values indicate higher power when attending to the left hemifield as compared to the right,
#  whereas negative values indicate the opposite.

# plot topo of pmi
# fig, ax1 = plt.subplots(1)
# im1, _ = mne.viz.plot_topomap(data = pmi, pos = epochs_attend_left.info, cmap = 'viridis', axes = ax1)
# add subtitles
# ax1.set_title("Power Modulation Index")
# fig.colorbar(im1, ax=ax1, location = 'right', shrink=0.7, label='Power',  pad = 0.1)
# fig.tight_layout()
# fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', f'topo_time_freq_pmi_{date}.png'))








