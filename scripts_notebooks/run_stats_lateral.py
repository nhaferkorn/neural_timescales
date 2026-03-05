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

# read in data for multiple subjects
subjects = ['sub-%d' % i for i in range(101, 152)]
contrasts = ['high_left', 'high_right', 'low_right', 'low_left']

# Statistical inference: Comparison of High vs. Low Distraction trials
infos = []
subj = []

# initialize empty lists 
tau_high_left = []
tau_high_right = []
tau_low_right = []
tau_low_left = []

# create dict - mapping contrasts to empty lists
tau_dict = {
    'high_left': tau_high_left,
    'high_right': tau_high_right,
    'low_right': tau_low_right,
    'low_left': tau_low_left
}

infos = []
subj_list = []

for sub in subjects:
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        infos.append(info)
        subj_list.append(sub)
    
    for contr in contrasts:
        file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'lateralization', f'{sub}_acf_params_{contr}.csv')
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # compute mean per channel
            df_tau = df.groupby('chs')[f'tau_{contr}'].mean()
            
            # append to the correct list
            tau_dict[contr].append(df_tau)

# print(tau_dict['high_left'])

# create arrays 

tau_arrays = {}  
for key in tau_dict:
    tau_arrays[key] = np.stack(tau_dict[key])


print(tau_arrays['high_left'].shape) # matches what I expect (41, 32)


# okay interesting, so now I end up with a dict where I store 4 numpy arrays for high_right, high_left, low_right & low_left

# subtract left - right side arrays
# but for that to work I first need to combine them (should I just take the mean??)
tau_array_left = (tau_arrays['high_left'] + tau_arrays['low_left']) / 2
tau_array_right = (tau_arrays['high_right'] + tau_arrays['low_right']) / 2

tau_array_side_diff = tau_array_left - tau_array_right
print(tau_array_side_diff.shape)

# compute grand average 
tau_left_grandavg = tau_array_left.mean(axis=0)
tau_right_grandavg = tau_array_right.mean(axis=0)
tau_side_diff_grandavg = tau_array_side_diff.mean(axis=0)


# visualize the left array
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_left_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('Topoplot Left GrandAvg')
# add colorbar 
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_lateral_left.png'))



# visualize the right array
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_right_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('Topoplot Right GrandAvg')
# add colorbar 
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_lateral_right.png'))


# visualize the difference array
fig, ax = plt.subplots(1)
im, cm = mne.viz.plot_topomap(data = tau_side_diff_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax)
fig.suptitle('Topoplot Left - Right (pooled over dist)')
# add colorbar 
ax_x_start = 0.85
ax_x_width = 0.03
ax_y_start = 0.2
ax_y_height = 0.6
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
unit_label = 'tau (s)'
clb.ax.set_title(unit_label) 
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', 'topo_group_lateral_left-right.png'))


