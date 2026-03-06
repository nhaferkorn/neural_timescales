"""This script computes statistics for the timescale estimates and plots them as topographies."""

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
subjects = ['sub-%d' % i for i in range(101, 152)]

# Statistical inference: Comparison of High vs. Low Distraction trials
infos = []
subj = []
tau_high = []
tau_low = []

for subject in subjects:
    # construct filepath
    file_path_high = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_singletrial_highdist.csv')
    file_path_low = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_singletrial_lowdist.csv')
    info_path = os.path.join(DERIV_DIR, 'info', f'{subject}_info.fif')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)

    # check if file path exists 
    if os.path.exists(file_path_high):
        df_high = pd.read_csv(file_path_high)
        df_low = pd.read_csv(file_path_low)

        # collect info objects
        infos.append(info)

        # update subjects
        subj.append(subject)

        # compute mean per channel (average over trials)
        df_tau_high_trialavg = df_high.groupby('chs')['tau_high'].mean()
        df_tau_low_trialavg = df_low.groupby('chs')['tau_low'].mean()

        # reindex to match channel order of info object
        df_tau_high_trialavg = df_tau_high_trialavg.reindex(info["ch_names"])
        df_tau_low_trialavg = df_tau_low_trialavg.reindex(info["ch_names"])

        tau_high.append(df_tau_high_trialavg)
        tau_low.append(df_tau_low_trialavg)



# Checked (06.03.26): it doesn't make a difference
tau_high_array_column = np.column_stack(tau_high)
tau_low_array_column = np.column_stack(tau_low)

# vertical stack
tau_high_array_row = np.stack(tau_high)
tau_low_array_row = np.stack(tau_low)

print(tau_high_array_column.shape) # (32, 41)
print(tau_high_array_row.shape) # (41, 32)

# average over subjects 
tau_high_grandavg_row = tau_high_array_row.mean(axis=0) 
tau_low_grandavg_row = tau_low_array_row.mean(axis=0)
print(tau_high_grandavg_row.shape)


# check if stacking method actually makes a difference
tau_high_grandavg_column = tau_high_array_column.mean(axis=1)
tau_low_grandavg_column = tau_low_array_column.mean(axis=1)
print(tau_high_grandavg_column.shape)

assert tau_high_grandavg_row.all() == tau_high_grandavg_column.all(), "shapes do not match"


# compute difference array
tau_diff_array = tau_high_grandavg_column - tau_low_grandavg_column
print(tau_diff_array)
print(tau_diff_array.shape)

## Visualize raw input arrays 
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_high_grandavg_column, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('High Dist (Channel Bug Fixed)') 
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_highdist_bugfix.png'))


fig, (ax1) = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_low_grandavg_column, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
fig.suptitle('Low Dist (Channel Bug Fixed)')
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_lowdist_bugfix.png'))


# Difference Plot
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_diff_array, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.suptitle('Difference High-Low Dist (Bug Fixed)')
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_high_low_diff_bugfix.png'))



# # Topoplots for individual subjects (maybe only 9 to start with)
# datasets = subj[:10]
# print(len(datasets))
# make this prettier, add title and spacing and labels & legend!!
# f, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
# for ax, subject, tau_high in zip(axes.ravel(), datasets, tau_high_array):
#     mne.viz.plot_topomap(tau_high, ch_type ='eeg', pos = infos[1],
#                                  axes=ax, cmap = 'viridis')
#     ax.set_title(subject)
# plt.tight_layout()
# plt.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_sub_high.png'))


# FIXME: Value Error!
# Dependent samples permutation test
p_value = 0.05

tail = 0 

# compute the threshold for our t-value 
# len refers to the first dimension of our data (here: number of sub)
deg_of_free = len(tau_diff_array) - 1
threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')


# implement Wilcoxon-signed rank as stat_func
def stat_fun_wilcox(X):
    result = sp.stats.wilcoxon(X)
    return result.statistic

cluster_stats = mne.stats.permutation_cluster_1samp_test(
    tau_diff_array,
    threshold=threshold,
    verbose=True, 
    tail=tail,
    adjacency=ch_adjacency,
    n_permutations=5000, 
    seed=13)

T_obs, clusters, cluster_p_values, _ = cluster_stats

# visualize the results
print(cluster_p_values)
print(T_obs)
print(T_obs.shape)



# # ## Encoding vs. Retrieval Timescales (these are still computed on different number of epochs)
# # ## So I might need to downsample them; randomly select them 
# # tau_enc = []
# # tau_ret = []

# # for subject in subjects:
# #     file_path_enc = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'encoding', f'{subject}_acf_params_single_trials_enc.csv')
# #     file_path_ret = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'retrieval', f'{subject}_acf_params_single_trials_ret.csv')

# #     # check if file path exists 
# #     if os.path.exists(file_path_enc):
# #         df_enc = pd.read_csv(file_path_enc)
# #         df_ret = pd.read_csv(file_path_ret)

# #         # average tau across trials
# #         df_tau_enc_trialavg = df_enc.groupby('channel')['tau_enc'].mean()
# #         df_tau_ret_trialavg = df_ret.groupby('channel')['tau_ret'].mean()

# #         tau_enc.append(df_tau_enc_trialavg)
# #         tau_ret.append(df_tau_ret_trialavg)

# # # convert lists into numpy arrays of 
# # tau_enc_array = np.stack(tau_enc)
# # tau_ret_array = np.stack(tau_ret)
# # tau_diff_enc_ret = tau_enc_array - tau_ret_array



# # ## Dependent samples permutation test
# # p_value = 0.05

# # tail = 0 

# # # compute the threshold for our t-value 
# # # len refers to the first dimension of our data (here: number of sub)
# # deg_of_free = len(tau_diff_enc_ret) - 1
# # threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

# # ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

# # cluster_stats = mne.stats.permutation_cluster_1samp_test(
# #     tau_diff_enc_ret,
# #     threshold=threshold,
# #     verbose=True, 
# #     tail=tail,
# #     adjacency=ch_adjacency,
# #     n_permutations=5000, 
# #     seed=13)

# # T_obs, clusters, cluster_p_values, _ = cluster_stats

# # # visualize the results
# # print(cluster_p_values)
# # print(T_obs)
# # print(T_obs.shape)


# # implement Wilcoxon-signed rank as stat_func
# def stat_fun_wilcox(X):
#     result = sp.stats.wilcoxon(X)
#     return result.statistic

# # I probably need to simulate distribution to extract threshold for wilcoxon signed rank (or maybe use TCFE?)

# # T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
# #     tau_diff_array,
# #     n_permutations=5000,
# #     threshold=thresh,
# #     tail=tail,
# #     adjacency=adjacency,
# #     stat_fun=stat_fun_wilcox
# # )