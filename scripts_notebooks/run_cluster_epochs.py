"""This script performs timescale analysis in the frequency domain."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys
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


keys_high_dist = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target']
keys_low_dist = ['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


event_high_dist = {k: EVENT_DICT[k] for k in keys_high_dist}
event_low_dist = {k: EVENT_DICT[k] for k in keys_low_dist}


# demean epochs
epochs_high_dist = mne.Epochs(reconst_raw, events = events, event_id = event_high_dist, tmin = 0, tmax=2.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)
epochs_low_dist = mne.Epochs(reconst_raw, events = events, event_id = event_low_dist, tmin = 0, tmax=2.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# print len of epochs
print('# of high dist epochs', len(epochs_high_dist))
print('# of low dist epochs', len(epochs_low_dist))

# plot the epochs & event markers on top of that (sanity check!)
# keys = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target',
#         'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right',
#          'Fixation Onset Enc', 'Fixation Onset Ret' ]

# events_of_interest = {k: EVENT_DICT[k] for k in keys}

# print(events_of_interest)
# epochs_high_dist.plot(events=events, event_id=event_high_dist, block=True)


# crop epochs (relative to inset of Stimuli, I think that should work)
epochs_high_dist_crop = epochs_high_dist.copy().crop(tmin=1, tmax=2.1)
epochs_low_dist_crop = epochs_low_dist.copy().crop(tmin=1, tmax=2.1)

# plot cropped epochs
# epochs_high_dist_crop.plot(events=events, block=True)
# epochs_low_dist_crop.plot(events=events, block=True)


# check bad channels
print("INFO OBJECT OF HIGH DIST EPOCHS CROPPED INFO:", epochs_high_dist_crop.info["bads"])


# # exclude bad channels
chs_cleaned = list(set(epochs_high_dist_crop.info['ch_names']) - set(epochs_high_dist_crop.info['bads']))

epochs_highdist = epochs_high_dist_crop.copy().pick_channels(chs_cleaned)
epochs_lowdist = epochs_low_dist_crop.copy().pick_channels(chs_cleaned)

print("INFO OBJECT epochs_highdist", epochs_highdist.info)


###########################################################################################
evoked_fix_highdist = epochs_highdist.average()
evoked_fix_lowdist = epochs_lowdist.average()

## FIT TIMESCALES ON EVOKED OBJECTS (NO OSC) - HIGH DISTRACTION TRIALS
acf_fitted_high, rsq_fitted_high = timescales_acf_evoked_np(sub, evoked_fix_highdist, osc = False)  

# convert to pd.DataFrame
df_high = pd.DataFrame(acf_fitted_high, columns = ["tau_high", "height_high", "offset_high"])

# add channel names
df_high['chs'] = evoked_fix_highdist.info['ch_names']

# add explained variance per channel
df_high['rsq_high'] = rsq_fitted_high
print(df_high.head(5))

# save as csv file
df_high.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'distraction', f'sub-{sub}_acf_params_evoked_highdist.csv'), sep = ',', header = True, index = False)


## FIT TIMESCALES ON EVOKED OBJECTS (NO OSC) - LOW DISTRACTION TRIALS
acf_fitted_low, rsq_fitted_low = timescales_acf_evoked_np(sub, evoked_fix_lowdist, osc = False)  

# convert to pd.DataFrame
df_low = pd.DataFrame(acf_fitted_low, columns = ["tau_low", "height_low", "offset_low"])

# add channel names
df_low['chs'] = evoked_fix_lowdist.info['ch_names']

# add explained variance per channel
df_low['rsq_low'] = rsq_fitted_low
print(df_low.head(5))

# save as csv file
df_low.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'distraction', f'sub-{sub}_acf_params_evoked_lowdist.csv'), sep = ',', header = True, index = False)


# maybe I can already combine the two dataframes at this stage and just save it as one big frame
df_combined = pd.merge(df_high, df_low, how = 'inner', on = "chs")

print(df_combined.head(5))

# add column with subject id
df_combined['sub'] = sub

# re-order columns to make it more readable
df_combined = df_combined.loc[:, ['sub', 'chs', 'tau_low', 'rsq_low', 'tau_high', 'rsq_high']]

# save the combined df as csv file
df_combined.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'distraction', f'sub-{sub}_acf_params_evoked_dist_combined.csv'), sep = ',', header = True, index = False)

