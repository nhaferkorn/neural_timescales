"""This script performs timescale estimation on high and low distraction trials."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, keys
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

# define keys
keys_high_dist = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target']
keys_low_dist = ['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']
key_high_dist_right = ['Encoding Stimulus Onset Distraction Right Target']
key_high_dist_left = ['Encoding Stimulus Onset Distraction Left Target']
key_low_dist_right = ['Encoding Stimulus Onset Baseline Right']
key_low_dist_left = ['Encoding Stimulus Onset Baseline Left']

# define events
event_high_dist = {k: EVENT_DICT[k] for k in keys_high_dist}
event_low_dist = {k: EVENT_DICT[k] for k in keys_low_dist}


# construct & demean epochs
epochs_high_dist = mne.Epochs(reconst_raw, events = events, event_id = event_high_dist, tmin = 0, tmax=2.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)
epochs_low_dist = mne.Epochs(reconst_raw, events = events, event_id = event_low_dist, tmin = 0, tmax=2.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# print len of epochs
print('# of high dist epochs', len(epochs_high_dist))
print('# of low dist epochs', len(epochs_low_dist))

# crop epochs (relative to stimulus onset)
epochs_high_dist_crop = epochs_high_dist.copy().crop(tmin=1, tmax=2.1)
epochs_low_dist_crop = epochs_low_dist.copy().crop(tmin=1, tmax=2.1)

# check bad channels (before interpolation)
print("INFO OBJECT OF HIGH DIST EPOCHS CROPPED INFO:", epochs_high_dist_crop.info["bads"])

# interpolate bad channels
epochs_highdist = epochs_high_dist_crop.copy().interpolate_bads(reset_bads=False) 
epochs_lowdist = epochs_low_dist_crop.copy().interpolate_bads(reset_bads=False)
print(epochs_highdist.info)
print(epochs_lowdist.info)

# save the epochs info object for that subject
epochs_highdist.info.save(fname = os.path.join(DERIV_DIR,'info', f'sub-{sub}_info.fif'), overwrite=True)


###########################################################################################
## FIT TIMESCALES ON SINGLE TRIALS (NO OSC) - HIGH DISTRACTION TRIALS
acf_trials_high, rsq_trials_high = timescales_acf_single_trials(sub=sub, epochs = epochs_highdist, osc=False)

# reshape output
acf_high_reshaped = acf_trials_high.reshape(acf_trials_high.shape[0] * acf_trials_high.shape[1], acf_trials_high.shape[2])
epochs_idx = np.repeat(np.arange(acf_trials_high.shape[0]), acf_trials_high.shape[1])
chs_idx = np.tile(epochs_highdist.ch_names, acf_trials_high.shape[0])

# convert to pd.DataFrame
df_high = pd.DataFrame(acf_high_reshaped, columns = ["tau_high", "height_high", "offset_high"])

df_high["epoch"] = epochs_idx
df_high["chs"] = chs_idx
df_high["sub"] = sub

# reshape rsq_trials_low
rsq_high_reshaped = rsq_trials_high.reshape(rsq_trials_high.shape[0] * rsq_trials_high.shape[1], rsq_trials_high.shape[2])
df_high['rsq_high'] = rsq_high_reshaped

# include the trial average
df_high['tau_high_trialavg'] = df_high.groupby('chs')['tau_high'].transform('mean')

# save as csv file
df_high.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'sub-{sub}_acf_params_singletrial_highdist.csv'), sep = ',', header = True, index = False)


####################################################################
## FIT TIMESCALES ON SINGLE TRIALS (NO OSC) - LOW DISTRACTION TRIALS
acf_trials_low, rsq_trials_low = timescales_acf_single_trials(sub=sub, epochs = epochs_lowdist, osc=False)

# reshape 
acf_low_reshaped = acf_trials_low.reshape(acf_trials_low.shape[0] * acf_trials_low.shape[1], acf_trials_low.shape[2])
epochs_idx = np.repeat(np.arange(acf_trials_low.shape[0]), acf_trials_low.shape[1])
chs_idx = np.tile(epochs_lowdist.ch_names, acf_trials_low.shape[0])

# convert to pd.DataFrame
df_low = pd.DataFrame(acf_low_reshaped, columns = ["tau_low", "height_low", "offset_low"])

df_low["epoch"] = epochs_idx
df_low["chs"] = chs_idx
df_low["sub"] = sub

# add explained variance per channel
# reshape rsq_trials_low
rsq_low_reshaped = rsq_trials_low.reshape(rsq_trials_low.shape[0] * rsq_trials_low.shape[1], rsq_trials_low.shape[2])
df_low['rsq_low'] = rsq_low_reshaped

# also add trial average
df_low['tau_low_trialavg'] = df_low.groupby('chs')['tau_low'].transform('mean')

# save as csv file
df_low.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'sub-{sub}_acf_params_singletrial_lowdist.csv'), sep = ',', header = True, index = False)


# # create combined data frame (I am equalizing the epochs count here, but not in an ideal / random way)
# df_combined = pd.merge(df_high, df_low, how = 'inner', on=["sub", "epoch", "chs"])

# # re-order columns to make it more readable
# df_combined = df_combined.loc[:, ['sub', 'epoch', 'chs', 'tau_low', 'tau_high', 'rsq_low', 'rsq_high', 'tau_high_trialavg', 'tau_low_trialavg']]

# # save the combined df as csv file
# df_combined.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'sub-{sub}_acf_params_singletrial_combined.csv'), sep = ',', header = True, index = False)


