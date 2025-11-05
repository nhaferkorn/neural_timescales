"""This script takes BIDS-formatted EEG data and pre-processess them.
    
    It is basically doing the same as preprocessing.py but with BIDSified data

"""

import os
import numpy as np
import mne
import mne_bids
import glob
import re

from src.settings import DATA_DIR, BIDS_ROOT, EVENT_DICT

###################################################################################################
###################################################################################################

# def main():

########################################################
## SETUP
subjs_list = list()
subjs_dirs = list()

# list subjects directory
sub_ids = [item for item in os.listdir(BIDS_ROOT)
           if os.path.isdir(os.path.join(BIDS_ROOT, item)) and item.startswith('sub-')]

# this just yields a list with: ['sub-01', 'sub-02', 'sub-03']

for subject in sub_ids:
    subject_dir = BIDS_ROOT(BIDS_ROOT, subject, "eeg")
    # subjs_dirs.append(subject_dir)

print("This is the subjects directory", "\n", subjs_dirs) 

#########################################################
## RUN ACROSS SUBJECTS

# fix subj_label extraction 

for subject in subjs_dirs:

    # Read raw BIDS data should now work with just one line of code
    raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose=False)

    # raw = mne.io.read_raw_bdf(raw_file, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS")
    # raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    # okay, so  I have to set montage directly on raw object (and not a copy of it!) > this creates a new entry in the info object
    raw.set_montage("biosemi32", on_missing = "ignore")

    ## downsample data 
    # raw.resample(500)

    ## check info structure
    print(raw.info)
    print(raw.info['ch_names'])
    print(raw.info['dig'])



# read subject specific event.tsv file
events = mne.read_events()

fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=EVENT_DICT
)

#%% Set EEG reference
# set average reference (but only set after removing bad periods, because otherwise it will spread)
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels = "average")

#%% Filtering (High-pass at 0.1 Hz)
# raw_filtered = raw_avg_ref.copy().filter(l_freq = 1.0, h_freq=None)

raw_filtered = raw.fcopy().filter(l_freq = 0.1, h_freq=45)

# To be implemented: visualize effect of filtering on the data
# raw.plot()
# raw_filtered.plot()

#%% Detecting & Annotating breaks
# mark end of encoding block and start of retrieval block 
mapping_blocks = {13:"End Encoding" , 50:"Start Retrieval", 93:"End Retrieval"}
annot_from_events = mne.annotations_from_events(events, event_desc = mapping_blocks, sfreq = raw_filtered.info["sfreq"], orig_time=raw.info["meas_date"],)
raw_filtered.set_annotations(annot_from_events)

print(raw_filtered.annotations.onset)

#%% Annotate blocks
onsets_blocks = [raw_filtered.first_time, raw_filtered.annotations.onset[1]]
durations_blocks = [raw_filtered.annotations.onset[0], raw_filtered.annotations.onset[2]]
descriptions_blocks = ["encoding block", "retrieval block"]

# okay so this works now!
block_annots = mne.Annotations(
    onset=onsets_blocks,
    duration=durations_blocks,
    description=descriptions_blocks,
    orig_time=raw_filtered.info["meas_date"],
)

#%% Annotate break between encoding & retrieval phase
onset_break = raw_filtered.annotations.onset[0]
duration_break = raw_filtered.annotations.onset[1] - raw_filtered.annotations.onset[0]
description_break = "BAD Break"

break_annots = mne.Annotations(
    onset=onset_break,
    duration=duration_break,
    description=description_break,
    orig_time=raw_filtered.info["meas_date"],
)


raw_filtered.set_annotations(annot_from_events + break_annots)
# raw_filtered.plot()

#%% Annotate rest periods 
mapping_rests = {90:"Rest onset", 91:"Rest offset"}

annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw_filtered.info["sfreq"], orig_time=raw.info["meas_date"],)

# this should reset the annotations - yesss, it does!
raw_filtered.set_annotations(annot_from_events_rests)

print(raw_filtered.annotations.onset)

# okay - basically I have to loop through the list and always select the pairs: I guess I need to implement a counter of some sorts: but this doesnt really work
onsets_rests = []
durations_rests = []

# okay, this works; but might not be the most elegant solution
for i in range(0, len(raw_filtered.annotations.onset)-1, 2):
    # set onset of rest
    onsets_rest = raw_filtered.annotations.onset[i]
    onsets_rests.append(onsets_rest)

    # set duration of rest
    durations_rest = raw_filtered.annotations.onset[i+1] - raw_filtered.annotations.onset[i]
    durations_rests.append(durations_rest)


print(onsets_rests)
print(durations_rests)

descriptions_rest = ["BAD_Rest Period"] * len(onsets_rests)
orig_times = raw_filtered.info["meas_date"]


rest_annots = mne.Annotations(
    onset=onsets_rests, 
    duration=durations_rests,
    description=descriptions_rest,
    orig_time=orig_times)


raw_filtered.set_annotations(annot_from_events_rests + rest_annots + break_annots)

#%% Detection of eye blinks
# dataset inclused horizontal and vertical EOG channels:
# EXG1 - horizontal EOG left 
# EXG2 - horizontal EOG right 
# EXG3 -  vertical EOG top
# EXG4 - vertical EOG bottom

# rename channels in raw object (not strictly necessary)
# new_names = {"EXG1": "EOGhl", "EXG2": "EOGhr",  "EXG3" : "EOGvt", "EXG4": "EOGvb"}

# ch_idx_by_type = mne.channel_indices_by_type(raw.info)
# print(ch_idx_by_type.keys())
#%% Detection of Blinks - check whether it makes a difference to omit breaks
# only give horizontal EEG channels as input to detect blinks: default threshold: (max(eog) - min(eog)) / 4
eog_events = mne.preprocessing.find_eog_events(raw_filtered, reject_by_annotation=True, ch_name = "EXG4")
# raw_filtered.plot(eog_events)

# From MNE Forum: 
# "Does your data consist of a long recording that spans several experimental blocks, 
# with breaks in-between blocks? If yes, then the movement artifacts during breaks could throw
#  off the peak detector (there are ways to avoid this issue, though).""

# After changing the threshold of mne.preprocessing.create_eog_epochs to 90e-6 and the rejection rate to a value between 250e-6 and 500e-6 (to what best fit the data), it is solved.
# https://mne.discourse.group/t/problem-finding-and-removing-eog-artifacts/4881/7 
# %% Using an EOG channel to select ICA components
ica = mne.preprocessing.ICA(n_components=15, max_iter="auto")
ica.fit(raw_filtered)
ica

# plot components
ica.plot_components()

# examine the ICs to see what they captured
explained_var_ratio = ica.get_explained_variance_ratio(raw_filtered)

for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

# use dedicated EOG channels as a "pattern" to check the ICs against, and automatically mark for exclusion any ICs that match EOG/ECG pattern
eog_evoked = mne.preprocessing.create_eog_epochs(raw_filtered).average()
ica.exclude = []

# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw_filtered)
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw_filtered, picks = eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw_filtered, show_scrollbars=False)

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

# %% Power line noise
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)
# add some arrows at 50 Hz and its harmonics:
for ax in fig.axes[1:]:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in (50, 100, 150, 200):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(
            x=freqs[idx],
            y=psds[idx] + 18,
            dx=0,
            dy=-12,
            color="red",
            width=0.1,
            head_width=3,
            length_includes_head=True,
        )

# okay, we definitely see some line noise at 50Hz
# %% Try to adjust the threshold
eog_events = mne.preprocessing.find_eog_events(raw_filtered, event_id=998, thresh = 200e-6)

(
    raw
    .copy()
    .pick_types(eog=True)
    .plot(events=eog_events, event_id={'EOG': 998})
)

# eog_epochs = mne.preprocessing.create_eog_epochs(raw_filtered)
# eog_epochs.plot_image(combine="mean")
# eog_epochs.average().plot_joint()


# %% Rejecting Epochs based on peak-to-peak channel amplitude 

# # use values from MNE Python tutorial (https://mne.tools/stable/auto_tutorials/preprocessing/20_rejecting_bad_data.html)
reject_criteria = dict(
    # eeg=100e-6,  # 100 µV
    eog=200e-6,    # 200 µV
) 

flat_criteria = dict(eeg=1e-6)  # 1 µV

epochs = mne.Epochs(
    raw,
    events,
    tmin=-0.2,
    tmax=0.5,
    reject_tmax=0,
    reject=reject_criteria,
    flat=flat_criteria,
    reject_by_annotation=False,
    preload=True,
)
epochs.plot_drop_log()

# nope. this doesn"t work, because it only detects 2 eog events
# raw.load_data()
# eog_epochs = mne.preprocessing.create_eog_epochs(raw.copy().filter(1, None))
# eog_epochs.average().plot_joint()
#%% Look Evoked Data - check for visual components 

# first step would be to epoch my data

raw_lh_filtered = raw.filter(l_freq=0.1, h_freq = 45)
epochs = mne.Epochs(raw_lh_filtered, events = events, event_id = event_dict)

# visualizing epoched data - plot power spectrum of encoding stimulus onset baseline left
epochs["Encoding Stimulus Onset Baseline Left"].compute_psd().plot(picks="eeg")


# compute evoked response
evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average()
evoked_enc_base_left.plot()


## plot evoked response relative to Fixation Onset Encoding - we see a decay in power over time (which makes sense, but is also interesting)
evoked_enc_fix = epochs["Fixation Onset Enc"].average()
evoked_enc_fix.plot()

#%% Compare left vs. right presented stimuli in encoding phase 

evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average(picks)
evoked_enc_base_left.plot(titles="left")

# now only pick occipital channels; I guess I can also use pick_channels_regexp method!
# for now: pick occipital electrodes
picks = mne.pick_channels(raw.info["ch_names"], ['O1', 'Oz' ,'O2'])

## Interesting: so, we definitely see a positive deflection centered at around 0.1 Hz, and then the signal plateus afterwards
evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average(picks)
evoked_enc_base_right.plot();
#%% I guess it make sense to pool over targets and baseline stimuli and then check for left and right side separately 

evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average().plot()
evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average().plot()

# I don't understand why 
evoked_enc_dist_left = epochs['Encoding Stimulus Onset Distraction Left Target'].average().plot()

# now, we get a much flatter power spectrum
enc_pooled_left = epochs[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']].average().plot()

#%% Evoked estimates for rerieval phase

evoked_ret_dist_left = epochs['Retrieval Stimulus Onset Distraction Left Target'].average().plot()
evoked_ret_base_left = epochs['Retrieval Stimulus Onset Baseline Left'].average().plot()


