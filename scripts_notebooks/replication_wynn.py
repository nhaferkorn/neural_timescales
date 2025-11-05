"""This script aims to replicate the EEG results from Wynn et al. 2022."""

import mne
import numpy as np
import os

import matplotlib.pyplot as plt

from src.settings import DATA_DIR, BIDS_ROOT, EVENT_DICT

# either load the data again or import a saved epoched objects
raw_file = os.path.join(DATA_DIR, "eeg", "103.bdf")

# change this  per subject
sub_id = "03"

# load the data
raw = mne.io.read_raw_bdf(raw_file, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS")

raw.set_montage("biosemi32", on_missing = "ignore") 


events = mne.find_events(raw, stim_channel="Status", initial_event=False)

# subtract marker offset value
events[:,2] = events[:, 2] - 64512
print(events)

# filtered the data between 0.5 and 30 Hz; apply to copy to not change raw object
raw_filtered = raw.copy().filter(l_freq = 0.5, h_freq=30)
events_eog = mne.preprocessing.find_eog_events(raw_filtered)

# set average reference (but only set after removing bad periods, because otherwise it will spread)
# raw_avg_ref = raw_filtered.set_eeg_reference(ref_channels = "average")

### Stimulus-locked epochs for encoding trials (-2000 to 2000 ms); why so long?

# subsetting a dictionary based on key values
keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
         ,'Fixation Onset Enc', 'Cue Onset'}

events_of_interest = {k: EVENT_DICT[k] for k in keys}

# Create epochs; check tmin and tmax parameters!!!
epochs = mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin=-1, tmax = 1, baseline=None)


fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest)
fig.suptitle(f"Events {sub_id}", fontsize=14)


################################################################################
## REJECTION BASED ON AMPLITUDE - COMPARE THE EFFECT OF MULTIPLE REJECTION PARAMETERS

# eog_values = [100e-6, 150e-6, 200e-6]

# rejection_criteria_list = []

# for value in eog_values:
#     reject_citeria = dict(eeg=100e-6,  
#     eog=value)

#     rejection_criteria_list.append(reject_citeria)

# print(rejection_criteria_list)

# flat_criteria = dict(eeg=1e-6) 

## Test how many epochs will be rejected based on these parameter combinations

# Is there way in MNE Python to create multiple epochs objects in one 

# epochs_list = list() # I don't even have to save it in a separate list 

# Note to myself: I am already using the band-pass filtered data here (raw_avg_ref)

# initialize empty list of epochs object

# epochs_after_rejection = list()

# for reject in rejection_criteria_list: 
#     epochs_bad_rejected =  mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin=-1, tmax = 1, baseline=None, reject_tmax=0,
#     reject=reject,
#     flat=flat_criteria,
#     reject_by_annotation=False,
#     preload=True,   
# )
#     epochs_bad_rejected.plot_drop_log()

#     epochs_after_rejection.append(epochs_bad_rejected)

# # not sure if this works, but I guess it does.
# print(epochs_after_rejection)


################################################################################
## REJECTION BASED ON ICA

raw_ica_filtered = raw.copy().filter(l_freq = 1.0, h_freq=40)
print('this is raw_ica_filtered', raw_ica_filtered)

# not sure which method is best here, and how many components; you can also directly pass a reject dictionary!!
ica = mne.preprocessing.ICA(n_components=15, method = "fastica", max_iter="auto", random_state=95)

# effect of including reject:  Signal periods exceeding the thresholds in reject or less than the thresholds in flat will be removed before fitting the ICA
# but how does it know how to set-up the epocj??
ica.fit(raw_ica_filtered)  # reject = rejection_criteria_ica
ica

# plot components
ica.plot_components(title=f"ICs for {sub_id}")


# find which ICs match the EOG pattern: this doesn't work!!
# eog_indices, eog_scores = ica.find_bads_eog(raw_filtered)

# select manually which component to exclude; not sure if this type of indexing actually works
ica.exclude = [1]

# # ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw_filtered.copy()
ica.apply(reconst_raw)

# plot the signal with blinks removed
reconst_raw.plot()
# use dedicated EOG channels as a "pattern" to check the ICs against, and automatically mark for exclusion any ICs that match EOG/ECG pattern

# this makes sense!!
raw_filtered.load_data()
ica.plot_sources(raw_filtered, show_scrollbars=False)

# after fitting ica - apply amplitude rejection criteria
reject_criteria = dict(eeg=100e-6,eog = 200e-6)


# set average reference: 
reconst_raw_ref = reconst_raw.copy().set_eeg_reference(ref_channels = 'average')

epochs_after_rejection = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin = -0.7, tmax=0.5, baseline=None, reject = reject_criteria)

# plot how many epochs were dropped 
epochs_after_rejection.drop_bad()

# plot epochs: 
# inspections shows that there is still quite a number of epochs that contain blinks
epochs_after_rejection.plot()


fig = epochs_after_rejection.plot_drop_log(show=False)
fig.suptitle(f"{sub_id} - ICA and Amplitude Reject (100, 200)", fontsize=14)
plt.show()


#################################################################################
## POWER ESTIMATES (after ICA & Amplitude Rejection) POOLED OVER DISTRACTION

# power_left.plot_joint(baseline=(-0.3, 0), mode="mean", title = 'sub-02 TF Power left, 100 EEG, 200 EOG')

# epochs_left_after_rejection.compute_psd(fmin=2.0, fmax=30).plot()
# epochs_right_after_rejection.compute_psd(fmin=2.0, fmax=30).plot()

# # butterfly plots
# epochs_left_after_rejection.average().plot(titles = 'Sub-03 Epochs left, 100 EEG, 200 EOG')
# epochs_right_after_rejection.average().plot(titles = 'Sub-03 Epochs right, 100 EEG, 200 EOG')

epochs_after_rejection = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=None, reject = reject_criteria, preload=True)

epochs_left_after_rejection = epochs_after_rejection[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]
epochs_right_after_rejection = epochs_after_rejection[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]
epochs_cue_onset_after_rejection = epochs_after_rejection['Cue Onset']

## This is interesting: I guess the small peak in voltage centered around -600ms pre-stimulus represent the (visual) processing of the cue stimulus.

## Compute TF power: 
freqs = np.arange(2,30,2)

n_cycles = freqs / 2
power_left = epochs_left_after_rejection.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_right = epochs_right_after_rejection.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_cue = epochs_cue_onset_after_rejection.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)

## and I am not sure of this makes sense - because the baseline correction that I am using still contains activity from the cue...
# %matplotlib qt
# Double check this
power_left.plot_joint(baseline=(-0.3, 0), mode="mean", tmin=-0.2, tmax=1, timefreqs=((0.1, 8), (.3, 8)))
power_left.plot_joint(baseline=(-0.3, 0))


power_left.plot_topo(baseline=(-0.3, 0), mode="mean", title=f"{sub_id} Average Power - Left Enc Stimuli (after ICA)")
power_right.plot_topo(baseline=(-0.3, 0), title=f"{sub_id} Average Power - Right Enc Stimuli (after ICA)") # this looks extremely weird & cannot be right

power_cue.plot_topo(baseline=(-0.3, 0), title=f"{sub_id} Average Power - Cue Onset (after ICA)") # this looks extremely weird & cannot be right


## Plot Evokeds for all channels
# %matplotlib inline
evoked_left = epochs_left_after_rejection.average().plot(titles=f"{sub_id} Evoked - Left Attend", picks = 'eeg')
evoked_right = epochs_right_after_rejection.average().plot(titles=f"{sub_id} Evoked - Right Attend", picks = 'eeg')
evoked_cue = epochs_cue_onset_after_rejection.average().plot(titles=f"{sub_id} Evoked - Cue", picks = 'eeg')



times = np.linspace(0.05, 0.13, 5)
epochs_left_after_rejection.average().plot_topomap(times=times, colorbar=True)
#############################################################################################################
## POWER MODULATION INDEX

## "To identify sensors and frequencies that reliably distinguished between the attentional conditions without any contribution from encoding-related processes >> they analyzed the cue window (-750 to -250ms relative to stimulus onset)"
## In the cue window, subjects were instructed to covertly direct their attention to the cued hemifield, but encoding related processes were expected to be minimal

## 1. Identify the sensors in which alpha power was modulated strongly in the cue-target interval:
# - calculate difference between alpha power for attending left and attending right trials & applying cluster-based permutation test

# Brittas adivce: compute grand average or use get_data method and then divide on raw data
# What is the difference between TRFAverage and Grand Average??

# pmi_numerator = (power_left_good - power_right_good) 
# pmi_denomenator = (power_left_good + power_right_good)


#####################################################################################################################
## LOOK AT 'Fixation Onset Enc' EPOCHS (DATA QUALITY)

# also spefify amplitude rejection criteria
rejection_criteria = dict(eeg=100e-6,eog = 200e-6)


# epoch such that I only extract the resting state periods
epochs = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin=-0.2, tmax = 1.2, baseline=None, reject = rejection_criteria, preload = True)

# subselect only fixation periods
epochs_resting = epochs['Fixation Onset Enc']

# plot those epochs
epochs_resting.plot()


# plot evoked objects
epochs_resting.average().plot(titles = "Sub-03 - Fixation Period Activity, 150, 200")


################################################################################################
## REPLICATION WYNN on cleaned data

keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
         ,'Fixation Onset Enc', 'Cue Onset'}
events_of_interest = {k: EVENT_DICT[k] for k in keys}

# amplitude rejection criteria
rejection_criteria = dict(eeg=100e-6,eog = 200e-6)


# print reconst_raw_ref object and info
print(reconst_raw_ref)


# extract -2000ms - 2000ms epochs
epochs_wynn = mne.Epochs(reconst_raw_ref, events = events, event_id = events_of_interest, tmin=-2, tmax = 2, baseline=None, reject = rejection_criteria, preload = True)



epochs_wynn_attend_left = epochs_wynn[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']]

epochs_wynn_attend_right = epochs_wynn[['Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Right Target']]


freqs = np.arange(2,30,2)

n_cycles = freqs / 2

## Compute PSD average
psd_wynn_left = epochs_wynn_attend_left.compute_psd(method='multitaper', tmin = -0.75, tmax = 0.75)
psd_wynn_right = epochs_wynn_attend_left.compute_psd(method = 'multitaper', tmin = -0.75, tmax = 0.75)


## Compute TRF
power_tf_wynn_left = epochs_wynn_attend_left.compute_tfr(method='multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)
power_tf_wynn_right = epochs_wynn_attend_left.compute_tfr(method = 'multitaper', freqs=freqs, n_cycles=n_cycles, average=True, decim=3)


## Plot PDS and TRF
psd_wynn_left.plot()

# I need to specify channels for this!!
power_tf_wynn_left.plot()

## how can I pool an average trf object over channels - doesn't exist, but check shape
## What does it mean that AverageTRF object is not subscriptable; then how can I index the power of a single channel??


### Grand Averages - mne.grand_averages()
### make grand average of a list of Evoked, Average TRF or Spectrum data

## use for mean of all subjects or runs