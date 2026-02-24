"""This script examines the drop logs of the trials and computes the overall number of trials."""

# make imports 
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import mne

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys



# read in data for multiple subjects
sub = sys.argv[1]


# read raw data and construct epochs
# load cleaned data for subject
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False)
# events = mne.find_events(reconst_raw, stim_channel = "Status")
events[:,2] = events[:, 2] - 64512


fix_enc_events = mne.pick_events(events, include = 40)
fix_ret_events = mne.pick_events(events, include = 80)

# subset of encoding trials: high vs. low distraction
fix_high_dist_events = mne.pick_events(events, include = [21, 22])
fix_low_dist_events = mne.pick_events(events, include = [23, 24])

# compute event counts (before trial rejection)
count_fix_enc_events = len(fix_enc_events)
count_fix_ret_events = len(fix_ret_events)
count_high_dist_events = len(fix_high_dist_events)
count_low_dist_events = len(fix_low_dist_events)

# check that high & low distraction trials add up to encoding event count
assert count_high_dist_events + count_low_dist_events == count_fix_enc_events, 'Distraction event counts does not add up'


# # print the number
# print('Number of fix encoding events', len(fix_enc_events), '\n')
# print('Number of ret encoding events', len(fix_ret_events), '\n')
# print('Number of high dist fix events', len(fix_high_dist_events), '\n')
# print('Number of low dist fix events', len(fix_low_dist_events))


# demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# crop epochs 
epochs_crop = epochs.copy().crop(tmin=0.2, tmax=1.1)


# interpolate bad channels, reset_bads: If True, remove the bads from info
epochs_interpolated = epochs_crop.copy().interpolate_bads(reset_bads=False) 
print(epochs_interpolated.info)


# subset epochs
epochs_fix_enc = epochs_interpolated['Fixation Onset Enc']
epochs_fix_ret = epochs_interpolated['Fixation Onset Ret']
epochs_fix_all = epochs_interpolated[['Fixation Onset Enc','Fixation Onset Ret']]


# print("# fix enc epochs after rejection", len(epochs_fix_enc), '\n')
# # print drop stats that is computed by MNE-Python - very interesting, I think it operates on the whole epochs count!
# # but I would expect them to be different, however that is not the case
# print('fix enc drop log', len(epochs_fix_enc.drop_log))
# print('fix enc drop stats', epochs_fix_enc.drop_log_stats())
# print("# fix ret epochs after rejection", len(epochs_fix_ret), '\n')


# compute my own epoch drop stats
count_enc_drop = len(fix_enc_events) - len(epochs_fix_enc)
count_ret_drop = len(fix_ret_events) - len(epochs_fix_ret)


print(f'Of the original {len(fix_enc_events)} fixation encoding periods, {count_enc_drop} were dropped, in percent: {count_enc_drop/len(fix_enc_events): .2f} %')

print(f'Of the original {len(fix_ret_events)} fixation retention periods, {count_ret_drop} were dropped ({count_ret_drop/len(fix_ret_events): .2f} %)')



# construct dataframe
print(count_enc_drop)
print(count_fix_enc_events)
print(sub)

df = pd.DataFrame()
df['sub'] = [sub]
df['enc_count'] = [count_fix_enc_events]
df['enc_drop'] = [count_enc_drop]
df['enc_drop_pct'] = [count_enc_drop/len(fix_enc_events)]
df['ret_count'] = [count_fix_ret_events]




print(df)








# with open(os.path.join(DERIV_DIR, 'epochs',f"sub-{sub}_fix_stats_drop.txt"), "w") as f:
#     f.write(
#         f"Of the original {len(fix_enc_events)} fixation encoding periods, "
#         f"{count_enc_drop} were dropped, "
#         f"in percent: {count_enc_drop/len(fix_enc_events)} %"
#     )