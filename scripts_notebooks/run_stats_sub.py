"""This script computes statistics for the timescale estimates."""

# make imports 
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import mne

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# # start with statistical analysis for single subject
# sub = 103

# # read info object
# info_path =  os.path.join(DERIV_DIR,'info', f'sub-{sub}_info.fif')
# info = mne.io.read_info(info_path)

# file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'sub-{sub}_acf_params_singletrial_combined.csv')

# df = pd.read_csv(file_path)

# df_high = df[['sub', 'epoch', 'chs', 'tau_high']]
# df_low = df[['sub', 'epoch', 'chs', 'tau_low']]


# # Compute Wilcoxon Signed Rank Test (one mean tau value per subject)

# read in data for multiple subjects
subjects = ['sub-%d' % i for i in range(101, 152)]

# initialize lists
datasets = []
infos = []
tau_high = []
tau_low = []


for subject in subjects:
    # construct filepath
    file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_singletrial_combined.csv')
    info_path = os.path.join(DERIV_DIR, 'info', f'{subject}_info.fif')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)

    # check if file path exists 
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        datasets.append(df)
        infos.append(info)
        tau_high.append(df['tau_high'].mean())
        tau_low.append(df['tau_low'].mean())

# convert lists to numpy arrays
np_tau_high = np.array(tau_high)
np_tau_low = np.array(tau_low)

# Wilcoxon signed rank test (okay, so apparently this is significant) - which is nice
# what kind of post-hoc analyses do you perform after running a wilcoxon signed rank test? - if any?
result = sp.stats.wilcoxon(np_tau_high, np_tau_low, alternative = 'two-sided') # but this doesn't correct for multiple comparisons!
print(result)


# Try to turn each subjects dataframe into an MNE object (but I am really not sure if this works... and if it actually makes sense to do that
# not sure if this is a good idea or if I am just overcomplicating things 

# # epochs_array with 1 timepoint 
# for df in datasets[:2]:
#     print(df.head())

#     df_low = df['tau_low']

#     # convert into numpy array
#     np_low = df_low.to_numpy()

#     print(np_low)

#     # try to initialze epochs array
#     epochs_array = mne.EpochsArray()



# But this is generally correct
# really not sure what the aggfunc parameter does...
# df_high = df_high.pivot_table(index=['sub', 'epoch'], columns=['chs'], values='tau_high',  aggfunc='first')
# df_low = df_low.pivot_table(index=['sub', 'epoch'], columns=['chs'], values='tau_low',  aggfunc='first')

# print(df_high.head())

# np_high = df_high.to_numpy()
# np_low = df_low.to_numpy()

# np_diff = np_high - np_low

# print(np_high)
# print(np_high.shape) # now I actually get the shape that I want (n_epochs x n_channels)


## ALTERNATIVE APPROACH (I THINK THIS SHOULD BE CORRECT)
# FIXME: compute cluster threshold 
# set the p-value we want to conduct our test at
# p_value = 0.05

# tail = 0 
# # Now, let's compute the threshold for our t-value - we need to pass that to the stats function.
# # len refers to the first dimension of our data
# deg_of_free = len(np_high) - 1
# threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

# # # Automatic detection of an appropriate adjacency matrix template only works for MEG data at the moment. This means that the
# # #  adjacency matrix is always computed for EEG data and never loaded from a template file. If you want to load a template for 
# # # a given montage use read_ch_adjacency() directly.
# ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')
# print(type(ch_adjacency)) 


# cluster_stats = mne.stats.permutation_cluster_1samp_test(
#     np_diff,
#     threshold=threshold,
#     verbose=True, 
#     tail=tail,
#     adjacency=ch_adjacency,
#     n_permutations=5000, 
#     seed=42)

# T_obs, clusters, cluster_p_values, _ = cluster_stats

# # visualize the results
# print(cluster_p_values)







##########################################################################################################################################################################
## Statistical Analysis at the Group Level

subjects = ['sub-%d' % i for i in range(101, 152)]

# initialize numpy array 
datasets = []

for subject in subjects:
    # construct filepath
    file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_singletrial_combined.csv')

    # check if file path exists 
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        datasets.append(df)
    else:
        continue


dfs_high = []
dfs_low = []
for dataset in datasets: 
    df_high = dataset.pivot_table(index=['sub', 'epoch'], columns=['chs'], values='tau_high',  aggfunc='first')
    df_low = dataset.pivot_table(index=['sub', 'epoch'], columns=['chs'], values='tau_low',  aggfunc='first')

    # append them to list (to be combined later)
    dfs_high.append(df_high)
    dfs_low.append(df_low)


df_glob = pd.concat(datasets, axis = 0, join='outer')
print(df_glob.head())

# df_high_sub = pd.concat(dfs_high)
# print(df_high_sub.head())

# group by subject
df_grouped = df_glob.groupby('sub')
print(df_grouped)


# try to set sub column as index column (still not exactly what I want)
# df_glob_reindex = df_glob.set_index(['sub', 'chs'])

# print(df_glob_reindex.head())
# print(df_glob_reindex['epoch'].dtype)
# print(df_glob_reindex.shape)


# # creating Evoked objects requires: an info object & a data array of shape (n_channels, n_times)




