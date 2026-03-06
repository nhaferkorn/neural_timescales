"""This script computes statistics for the timescale estimates.
Age Contrast."""

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


# Old Adults vs. Young Adults
infos = []
sub_young = []
sub_old = []
tau_old = []
tau_young = []

for subject in subjects:
    # construct paths
    timescales_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'{subject}_acf_params_single_trials_all.csv')
    behavioral_path = os.path.join(DERIV_DIR, 'behavioral', 'summary_stats', f'{subject}_summary.csv')
    info_path = os.path.join(DERIV_DIR, 'info', f'{subject}_info.fif')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        
        # collect info objects
        infos.append(info)

    # check if file path exists 
    if os.path.exists(behavioral_path):
        df_behav = pd.read_csv(behavioral_path)
    

    # split subjects into OLD and YOUNG based on behavior
    if df_behav['age'].iloc[0] == 'young':
        sub_young.append(df_behav['sub'].iloc[0])
    
    else:
        sub_old.append(df_behav['sub'].iloc[0])

# roughly equal (23 and 28)
print(len(sub_young))
print(len(sub_old))





# for young adults
for sub in sub_young:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'sub-{sub}_acf_params_single_trials_all.csv')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
    
    if os.path.exists(timescales_path):
        df_young = pd.read_csv(timescales_path)

        # compute trial-average per channel 
        df_tau_young_trialavg = df_young.groupby('channel')['tau_all'].mean()

        # reindex channels to match order of info object 
        df_tau_young_trialavg = df_tau_young_trialavg.reindex(info["ch_names"])

        tau_young.append(df_tau_young_trialavg)


# for old adults
for sub in sub_old:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'sub-{sub}_acf_params_single_trials_all.csv')
    
    if os.path.exists(timescales_path):
        df_old = pd.read_csv(timescales_path)

        # compute trial-average per channel 
        df_tau_old_trialavg = df_old.groupby('channel')['tau_all'].mean()

        # reindex channels to match order of info object 
        df_tau_old_trialavg = df_tau_old_trialavg.reindex(info["ch_names"])

        tau_old.append(df_tau_old_trialavg)


print('LEN OLD', len(tau_old))
print('LEN YOUNG', len(tau_young))


# convert lists into numpy arrays
tau_young_array = np.stack(tau_young)
tau_old_array = np.stack(tau_old)

# check their dimensions
print(tau_young_array.shape)
print(tau_old_array.shape)

# average over subjects 
tau_young_grandavg = tau_young_array.mean(axis=0) 
tau_old_grandavg = tau_old_array.mean(axis=0)
tau_diff_grandavg = tau_old_grandavg - tau_young_grandavg
print(tau_young_grandavg.shape)




# visualize input arrays
# young adults
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_young_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('Young Adults GrandAvg (Bugfix)')
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_young_bugfix.png'))

# old adults
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_old_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('Old Adults GrandAvg (Bugfix)')
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_old_bugfix.png'))


## difference plot
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_diff_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('OLD - YOUNG Diff GrandAvg (Bugfix)')
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_old_young_diff_bugfix.png'))




# # for statistical test - now I am comparing two independent groups


