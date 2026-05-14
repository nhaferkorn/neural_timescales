"""This script computes descriptive epoch stats."""

# make imports 
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import mne

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# # read in data for multiple subjects
# sub = sys.argv[1]

# # load cleaned data for subject
# print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
# reconst_fname = f'sub-{sub}-raw_cleaned.fif'
# RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)
# reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# # find events
# events = mne.find_events(reconst_raw, stim_channel="Status", initial_event=False,shortest_event=1)
# events[:,2] = events[:, 2] - 64512

# # encoding - retrieval split
# fix_enc_events = mne.pick_events(events, include=40)
# fix_ret_events = mne.pick_events(events, include=80)

# # subset of encoding trials: low (21, 22)vs. high (23, 24) distractor load
# fix_low_dist_events = mne.pick_events(events, include=[21, 22])
# fix_high_dist_events = mne.pick_events(events, include =[23, 24])

# # left - right split
# fix_left_events = mne.pick_events(events, include = [21, 23])
# fix_right_events = mne.pick_events(events, include=[22, 24])

# # split left/right and distraction load
# fix_low_dist_left_events = mne.pick_events(events, include = [21])
# fix_low_dist_right_events = mne.pick_events(events, include = [22])
# fix_high_dist_left_events = mne.pick_events(events, include = [23])
# fix_high_dist_right_events = mne.pick_events(events, include = [24])

# # compute event counts (before trial rejection)
# count_fix_enc_events = len(fix_enc_events)
# count_fix_ret_events = len(fix_ret_events)
# count_high_dist_events = len(fix_high_dist_events)
# count_low_dist_events = len(fix_low_dist_events)
# count_left_events = len(fix_left_events)
# count_right_events = len(fix_right_events)
# count_high_dist_right_events = len(fix_high_dist_right_events)
# count_high_dist_left_events = len(fix_high_dist_left_events)
# count_low_dist_right_events = len(fix_low_dist_right_events)
# count_low_dist_left_events = len(fix_low_dist_left_events)

# # check that high & low distraction trials add up to encoding event count
# # assert count_high_dist_events + count_low_dist_events == count_fix_enc_events, 'Distraction event counts do not add up'

# #I need to create two different epochs objects, i. for fixation periods and ii. for distraction split
# # create fixation epochs & demean them 
# epochs_fixation = mne.Epochs(reconst_raw, events = events, event_id = {'Fixation Onset Enc': 40, 'Fixation Onset Ret':80}, tmin = 0.2, tmax=1.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)
# epochs_distraction = mne.Epochs(reconst_raw, events = events, event_id = {'Encoding Stimulus Onset Baseline Left': 21, 'Encoding Stimulus Onset Baseline Right': 22, 'Encoding Stimulus Onset Distraction Left Target': 23, 'Encoding Stimulus Onset Distraction Right Target': 24}, tmin = 1.2, tmax=2.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# # interpolate bad channels, reset_bads: If True, remove the bads from info
# epochs_interpolated_fixation = epochs_fixation.copy().interpolate_bads(reset_bads=False) 
# epochs_interpolated_distraction = epochs_distraction.copy().interpolate_bads(reset_bads=False)

# # subset epochs
# epochs_fix_enc = epochs_interpolated_fixation['Fixation Onset Enc']
# epochs_fix_ret = epochs_interpolated_fixation['Fixation Onset Ret']
# epochs_fix_all = epochs_interpolated_fixation[['Fixation Onset Enc','Fixation Onset Ret']]
# epochs_high_dist = epochs_interpolated_distraction[['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target']]
# epochs_low_dist = epochs_interpolated_distraction[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']]
# epochs_left = epochs_interpolated_distraction[['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Baseline Left']]
# epochs_right = epochs_interpolated_distraction[['Encoding Stimulus Onset Distraction Right Target', 'Encoding Stimulus Onset Baseline Right']]
# epochs_high_dist_right = epochs_interpolated_distraction['Encoding Stimulus Onset Distraction Right Target']
# epochs_high_dist_left = epochs_interpolated_distraction['Encoding Stimulus Onset Distraction Left Target']
# epochs_low_dist_right = epochs_interpolated_distraction['Encoding Stimulus Onset Baseline Right']
# epochs_low_dist_left = epochs_interpolated_distraction['Encoding Stimulus Onset Baseline Left']

# # compute epoch drop stats
# count_enc_drop = count_fix_enc_events - len(epochs_fix_enc)
# count_ret_drop = count_fix_ret_events- len(epochs_fix_ret)
# count_high_dist_drop = count_high_dist_events - len(epochs_high_dist)
# count_low_dist_drop = count_low_dist_events - len(epochs_low_dist)
# count_left_drop = count_left_events - len(epochs_left)
# count_right_drop = count_right_events - len(epochs_right)


# # construct dataframe
# df = pd.DataFrame()
# df['sub'] = [sub]
# df['enc_count'] = [count_fix_enc_events]
# df['enc_drop'] = [count_enc_drop]
# df['enc_retained'] = [len(epochs_fix_enc)]
# df['enc_drop_pct'] = [round(count_enc_drop/count_fix_enc_events, 4)]
# df['ret_count'] = [count_fix_ret_events]
# df['ret_drop'] = [count_ret_drop]
# df['ret_retained'] = [len(epochs_fix_ret)]
# df['ret_drop_pct'] = [round(count_ret_drop/count_fix_ret_events, 4)]
# df['high_dist_count'] = [count_high_dist_events]
# df['high_dist_drop'] = [count_high_dist_drop]
# df['high_dist_drop_pct'] = [count_high_dist_drop/count_high_dist_events]
# df['high_dist_retained'] = [len(epochs_high_dist)]
# df['low_dist_count'] = [count_low_dist_events]
# df['low_dist_drop'] = [count_low_dist_drop]
# df['low_dist_drop_pct'] = [count_low_dist_drop/count_low_dist_events]
# df['low_dist_retained'] = [len(epochs_low_dist)]
# df['left_count'] = [count_left_events]
# df['left_drop'] = [count_left_drop]
# df['left_drop_pct'] = [round(count_left_drop/count_left_events, 4)]
# df['left_retained'] = [len(epochs_left)]
# df['right_count'] = [count_right_events]
# df['right_drop'] = [count_right_drop]
# df['right_drop_pct'] = [round(count_right_drop/count_right_events, 4)]
# df['right_retained'] = [len(epochs_right)]
# df['high_dist_left_count'] = [count_high_dist_left_events]
# df['high_dist_left_retained'] = [len(epochs_high_dist_left)]
# df['high_dist_right_count'] = [count_high_dist_right_events]
# df['high_dist_right_retained'] = [len(epochs_high_dist_right)]
# df['low_dist_left_count'] = [count_low_dist_left_events]
# df['low_dist_left_retained'] = [len(epochs_low_dist_left)]
# df['low_dist_right_count'] = [count_low_dist_right_events]
# df['low_dist_right_retained'] = [len(epochs_low_dist_right)]


# # save as csv file
# df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'epochs', f'sub-{sub}_epochs_stats.csv'), sep = ',', header = True, index = False)



# compute mean epochs stats
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]

epochs_stats = []

for sub in subjects:

    path = os.path.join(DERIV_DIR, 'epochs',f'{sub}_epochs_stats.csv')

    if os.path.exists(path):
        stats = pd.read_csv(path)

        epochs_stats.append(stats)

# join them 
df = pd.concat(epochs_stats)


df_mean = df.describe().round(4)


df_mean.to_csv(os.path.join(DERIV_DIR, 'epochs', f'summary_epochs_stats.csv'), sep = ',', header = True, index = True)



