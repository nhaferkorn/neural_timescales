"""This script aims to replicate the EEG results from Wynn et al. 2022."""

# make imports
import mne
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR

# set system variables
sub = sys.argv[1]


# read cleaned data



# Power estimates pooled over distraction
epochs_cleaned = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=None, preload=True)

epochs_left_after_rejection = epochs_cleaned[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]
epochs_right_after_rejection = epochs_cleaned[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]
epochs_cue_onset_after_rejection = epochs_cleaned['Cue Onset']


## Compute TF power: 
freqs = np.arange(2,30,2)

n_cycles = freqs / 2
power_left = epochs_left_after_rejection.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_right = epochs_right_after_rejection.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_cue = epochs_cue_onset_after_rejection.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)

power_left.plot_joint(baseline=(-0.3, 0), mode="mean", tmin=-0.2, tmax=1, timefreqs=((0.1, 8), (.3, 8)))
power_left.plot_joint(baseline=(-0.3, 0))


power_left.plot_topo(baseline=(-0.3, 0), mode="mean", title=f"{sub} Average Power - Left Enc Stimuli (after ICA)")
power_right.plot_topo(baseline=(-0.3, 0), title=f"{sub} Average Power - Right Enc Stimuli (after ICA)") # this looks extremely weird & cannot be right

power_cue.plot_topo(baseline=(-0.3, 0), title=f"{sub} Average Power - Cue Onset (after ICA)") # this looks extremely weird & cannot be right

## Plot Evokeds for all channels

evoked_left = epochs_left_after_rejection.average().plot(titles=f"{sub} Evoked - Left Attend", picks = 'eeg')
evoked_right = epochs_right_after_rejection.average().plot(titles=f"{sub} Evoked - Right Attend", picks = 'eeg')
evoked_cue = epochs_cue_onset_after_rejection.average().plot(titles=f"{sub} Evoked - Cue", picks = 'eeg')

times = np.linspace(0.05, 0.13, 5)
epochs_left_after_rejection.average().plot_topomap(times=times, colorbar=True)




## POWER MODULATION INDEX

## "To identify sensors and frequencies that reliably distinguished between the attentional conditions without any contribution from encoding-related processes >> they analyzed the cue window (-750 to -250ms relative to stimulus onset)"
## In the cue window, subjects were instructed to covertly direct their attention to the cued hemifield, but encoding related processes were expected to be minimal

## 1. Identify the sensors in which alpha power was modulated strongly in the cue-target interval:
# - calculate difference between alpha power for attending left and attending right trials & applying cluster-based permutation test

# Brittas adivce: compute grand average or use get_data method and then divide on raw data
# What is the difference between TRFAverage and Grand Average??

# pmi_numerator = (power_left_good - power_right_good) 
# pmi_denomenator = (power_left_good + power_right_good)



## REPLICATION WYNN on cleaned data

# keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
#          ,'Fixation Onset Enc', 'Cue Onset'}
# events_of_interest = {k: EVENT_DICT[k] for k in keys}

# # amplitude rejection criteria
# rejection_criteria = dict(eeg=100e-6,eog = 200e-6)


# # print reconst_raw_ref object and info
# print(reconst_raw_ref)


# # extract -2000ms - 2000ms epochs
# epochs_wynn = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin=-2, tmax = 2, baseline=None, reject = rejection_criteria, preload = True)

# epochs_wynn_attend_left = epochs_wynn[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]

# epochs_wynn_attend_right = epochs_wynn[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]


# freqs = np.arange(2,30,2)

# n_cycles = freqs / 2

# ## Compute PSD average
# psd_wynn_left = epochs_wynn_attend_left.compute_psd(method='multitaper', tmin = -0.75, tmax = 0.75)
# psd_wynn_right = epochs_wynn_attend_left.compute_psd(method = 'multitaper', tmin = -0.75, tmax = 0.75)


# ## Compute TRF
# power_tf_wynn_left = epochs_wynn_attend_left.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
# power_tf_wynn_right = epochs_wynn_attend_left.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)


# ## Plot PDS and TRF
# psd_wynn_left.plot()

# # I need to specify channels for this!!
# power_tf_wynn_left.plot()

# ## how can I pool an average trf object over channels - doesn't exist, but check shape
# ## What does it mean that AverageTRF object is not subscriptable; then how can I index the power of a single channel??


# ### Grand Averages - mne.grand_averages()
# ### make grand average of a list of Evoked, Average TRF or Spectrum data

# ## use for mean of all subjects or runs