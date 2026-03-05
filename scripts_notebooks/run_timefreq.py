"""This script aims to replicate the EEG results from Wynn et al. 2022."""

# make imports
import mne
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED

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

# run epochs 
keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
          ,'Fixation Onset Enc'}

events_of_interest = {k: EVENT_DICT[k] for k in keys}

epochs_cleaned = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = None, reject_by_annotation=True, picks = 'eeg', on_missing="ignore")

# Power estimates pooled over distraction
epochs_left_target = epochs_cleaned[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]
epochs_right_target = epochs_cleaned[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]

## Compute TF power: 
freqs = np.arange(2,30,2)
n_cycles = freqs / 2

power_left = epochs_left_target.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_right = epochs_right_target.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)

# plotting time-freq power
power_left.plot_joint(baseline=(-0.3, 0), mode="mean", tmin=-0.2, tmax=1, timefreqs=((0.1, 8), (.3, 8)))
power_left.plot_topo(baseline=(-0.3, 0), mode="mean", title=f"{sub} Average Power - Left Enc Stimuli")
power_right.plot_topo(baseline=(-0.3, 0), title=f"{sub} Average Power - Right Enc Stimuli") 



###################################################################################################################################################################################################


                                                                        ## Replication of Syanah's analysis


##################################################################################################################################################################################################

## 1. Identify the sensors in which alpha power was modulated strongly in the cue-target interval:
# calculate difference between alpha power for attending left and attending right trials & applying cluster-based permutation test

print("NOW RUNNING REPLICATION ANALYSIS")

# not sure whether I should apply baseline correction or not
epochs_cue_window = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin=-0.75, tmax = -0.25, baseline=None,  preload = True)

epochs_attend_left = epochs_cue_window[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]

epochs_attend_right = epochs_cue_window[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]

freqs = np.arange(2,30,2)

n_cycles = freqs / 2

# Compute PSD 
# psd_wynn_left = epochs_wynn_attend_left.compute_psd(method='multitaper', tmin = -0.75, tmax = 0.75)
# psd_wynn_right = epochs_wynn_attend_left.compute_psd(method = 'multitaper', tmin = -0.75, tmax = 0.75)


# Compute TRF
power_tf_left = epochs_attend_left.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_tf_right = epochs_attend_left.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)


# Compute TRF Difference (Left - Right)
power_diff = power_tf_left - power_tf_right


print(type(power_diff))

# plot power_diff
power_diff.plot_topo(baseline=(-0.3, 0), mode="mean", title=f"{sub} Average Power - Left Enc Stimuli")




# Brittas adivce: compute grand average or use get_data method and then divide on raw data
# pmi_numerator = (power_left_good - power_right_good) 
# pmi_denomenator = (power_left_good + power_right_good)
