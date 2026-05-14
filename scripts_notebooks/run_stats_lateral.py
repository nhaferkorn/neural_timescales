"""This script computes statistics for the timescale estimates."""

# make imports 
import os
import sys
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import mne
import cmocean

from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from datetime import datetime

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")

# read in data for multiple subjects
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]
contrasts = ['right', 'left', 'high_left', 'high_right', 'low_right', 'low_left']


# initialize empty lists 
tau_right = []
tau_left = []
tau_high_left = []
tau_high_right = []
tau_low_right = []
tau_low_left = []

# create dict to map contrasts to empty lists
tau_raw_dict = {k: [] for k in contrasts}
tau_log_dict = {k: [] for k in contrasts}

infos = []
subj_list = []

# read timescale data
for sub in subjects:
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        infos.append(info)
        subj_list.append(sub)
    
    for contr in contrasts:
        file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'lateralization', f'{sub}_acf_params_{contr}.csv')
        
        if os.path.exists(file_path):

            # read file
            df = pd.read_csv(file_path)

            # create a new column for model fit
            df['model_fit'] = df[f'rsq_{contr}'].apply(lambda x: 'good' if x > 0.34 else 'poor')

            # select only rows with good fit
            df = df[df['model_fit'] == 'good']

            # add column for log-transformed tau
            df[f'tau_{contr}_log'] = np.log(df[f'tau_{contr}'])

            # log-transformed trial average
            df_tau_log_trialavg = df.groupby('chs')[f'tau_{contr}_log'].mean()

            # raw trial average
            df_tau_raw_trialavg = df.groupby('chs')[f'tau_{contr}'].mean()

            # reindex channels to match order of info object 
            df_tau_log_trialavg = df_tau_log_trialavg.reindex(info["ch_names"])
            df_tau_raw_trialavg = df_tau_raw_trialavg.reindex(info["ch_names"])
            
            # append to list in dict
            tau_raw_dict[contr].append(df_tau_raw_trialavg)
            tau_log_dict[contr].append(df_tau_log_trialavg)


# create arrays 
tau_log_arrays = {}  
for key in tau_log_dict:
    tau_log_arrays[key] = np.stack(tau_log_dict[key])


print(tau_log_arrays['high_left'].shape)
assert tau_log_arrays['high_left'].shape[0] == len(subj_list), 'shape mismatch'


tau_log_left_diff_array = tau_log_arrays['high_left'] - tau_log_arrays['low_left']
tau_log_right_diff_array = tau_log_arrays['high_right'] - tau_log_arrays['low_right']

# compute grand average (averaged over subjects)
tau_log_arrays_grandavg = {}

for key in tau_log_arrays:
    tau_log_arrays_grandavg[key] = (tau_log_arrays[key].mean(axis=0))

# make sure shape is right (32,)
assert tau_log_arrays_grandavg['high_left'].shape[0] == 32, 'shape mismatch'


## same for raw values (for plotting)
tau_raw_arrays = {}  
for key in tau_raw_dict:
    tau_raw_arrays[key] = np.stack(tau_raw_dict[key])


# makes sure shape is right
assert tau_raw_arrays['high_left'].shape[0] == len(subj_list), 'shape mismatch'


tau_raw_left_diff_array = tau_raw_arrays['high_left'] - tau_raw_arrays['low_left']

# compute grand average (averaged over subjects)
tau_raw_arrays_grandavg = {}

for key in tau_raw_arrays:
    tau_raw_arrays_grandavg[key] = (tau_raw_arrays[key].mean(axis=0))

# # (32,)
assert tau_log_arrays_grandavg['high_left'].shape[0] == 32, 'shape mismatch'

# raw values for plotting
tau_raw_left_high_grandavg = tau_raw_arrays_grandavg['high_left']
tau_raw_left_low_grandavg = tau_raw_arrays_grandavg['low_left']
tau_raw_right_high_grandavg = tau_raw_arrays_grandavg['high_right']
tau_raw_right_low_grandavg = tau_raw_arrays_grandavg['low_right']

# grand average raw difference array
tau_diff_left_grandavg = tau_raw_left_high_grandavg - tau_raw_left_low_grandavg
tau_diff_right_grandavg = tau_raw_right_high_grandavg - tau_raw_right_low_grandavg


# ## Contrast: Left Target Trials (High vs. Low Distraction)

# Compute Stats
print('Computing Cluster Permutation Test')
p_value = 0.05
tail = 0 

# compute the threshold for our t-value 
# len refers to the first dimension of our data (here: number of sub)
deg_of_free = len(tau_log_left_diff_array) - 1
threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

cluster_stats = mne.stats.permutation_cluster_1samp_test(
    tau_log_left_diff_array,
    threshold=threshold,
    verbose=True, 
    tail=tail,
    adjacency=ch_adjacency,
    n_permutations=5000, 
    seed=13)

T_obs, clusters, cluster_p_values, _ = cluster_stats

# print the results
print(cluster_p_values)
print(T_obs)
print(T_obs.shape)
print('Clusters', clusters)

# works, but not the most elegant way
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
print("Good clusters: %i" % len(good_cluster_inds))
print('Good cluster indices', good_cluster_inds)



# plot topographies
fig = plt.figure(figsize=(10, 4))
sel = mne.pick_types(infos[1], eeg=True)
info = mne.pick_info(infos[1], sel)

mask_params = dict(marker='.', markerfacecolor='w', markersize=11)
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# create empty mask (n_channels,)
mask = np.zeros(len(info['ch_names']), dtype=bool)


# fill mask if clusters exist
if len(good_cluster_inds) > 0:
    for clu_idx in good_cluster_inds:
        space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        mask[ch_inds] = True
else:
    print("No significant clusters found.")

vmax = np.max(np.abs([
    tau_raw_left_high_grandavg * 1000,
    tau_raw_left_low_grandavg * 1000
]))

vlim1 = (0, vmax)
vmax3 = np.max(np.abs(tau_diff_left_grandavg * 1000))
vmax3 = np.round(vmax3 * 2) / 2
vlim3 = (-vmax3, vmax3)

# plot topomaps (always runs)
im1, _ = mne.viz.plot_topomap(
    tau_raw_left_high_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax1,
    vlim=vlim1
)

im2, _ = mne.viz.plot_topomap(
    tau_raw_left_low_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax2,
    vlim=vlim1
)

im3, _ = mne.viz.plot_topomap(
    tau_diff_left_grandavg * 1000,
    info,
    axes=ax3,
    cmap=cmocean.cm.balance,  
    # cnorm=norm,
    mask=mask if mask.any() else None,  
    sensors=True,
    mask_params=mask_params,
    show=False,
    vlim=vlim3
)

# titles
ax1.set_title("High Dist")
ax2.set_title("Low Dist")
ax3.set_title("High Dist - Low Dist")

# colorbars
# shared colorbar for input arrays
cbar12 = fig.colorbar(
    im1,
    ax=[ax1, ax2],   
    location='bottom',
    shrink=0.4,
    pad=0.1
)

cbar12.set_label(r'$\tau$ (ms)')

# control third colorbar
cbar3 = fig.colorbar(im3, ax=ax3, location='bottom', shrink=0.6,  label=r'$\tau$ (ms)', pad=0.1)
vlim3 = (-vmax3, vmax3)
cbar3.set_ticks(np.linspace(-vmax3, vmax3, 5))
cbar3.ax.tick_params(labelsize=8)
# show only first decimal for tick labels
cbar3.set_ticklabels([f"{t:.1f}" for t in cbar3.get_ticks()])

# figure labels
ax1.text(-0.125, 1.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')

# save
fig.savefig(
    os.path.join(DERIV_DIR, 'figures', f'topos_lateralization_left_clusters_markers_{date}.png'),
    bbox_inches='tight'
)


## repeat for the right trials
# Compute Stats
print('Computing Cluster Permutation Test for Right Trials')
p_value = 0.05
tail = 0 

# compute the threshold for our t-value 
# len refers to the first dimension of our data (here: number of sub)
deg_of_free = len(tau_log_right_diff_array) - 1
threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

cluster_stats = mne.stats.permutation_cluster_1samp_test(
    tau_log_left_diff_array,
    threshold=threshold,
    verbose=True, 
    tail=tail,
    adjacency=ch_adjacency,
    n_permutations=5000, 
    seed=13)

T_obs, clusters, cluster_p_values, _ = cluster_stats

# print the results
print(cluster_p_values)
print(T_obs)
print(T_obs.shape)
print('Clusters', clusters)

# works, but not the most elegant way
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
print("Good clusters: %i" % len(good_cluster_inds))
print('Good cluster indices', good_cluster_inds)

# plot topographies
fig = plt.figure(figsize=(10, 4))
sel = mne.pick_types(infos[1], eeg=True)
info = mne.pick_info(infos[1], sel)

mask_params = dict(marker='.', markerfacecolor='w', markersize=11)
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# create empty mask (n_channels,)
mask = np.zeros(len(info['ch_names']), dtype=bool)

# fill mask if clusters exist
if len(good_cluster_inds) > 0:
    for clu_idx in good_cluster_inds:
        space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        mask[ch_inds] = True
else:
    print("No significant clusters found.")


# specify limits
vmax = np.max(np.abs([
    tau_raw_right_high_grandavg * 1000,
    tau_raw_right_low_grandavg * 1000
]))

vlim1 = (0, vmax)
vmax3 = np.max(np.abs(tau_diff_right_grandavg * 1000))
vmax3 = np.round(vmax3 * 2) / 2
vlim3 = (-vmax3, vmax3)

# plot topomaps (always runs)
im1, _ = mne.viz.plot_topomap(
    tau_raw_right_high_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax1
)

im2, _ = mne.viz.plot_topomap(
    tau_raw_right_low_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax2
)

im3, _ = mne.viz.plot_topomap(
    tau_diff_right_grandavg * 1000,
    info,
    cmap=cmocean.cm.balance,
    axes=ax3,
    mask=mask if mask.any() else None, 
    sensors=True,
    mask_params=mask_params,
    show=False
)

# titles
ax1.set_title("Right High Dist")
ax2.set_title("Right Low Dist")
ax3.set_title("High Dist - Low Dist" if mask.any() else "High Dist - Low Dist")

cbar12 = fig.colorbar(
    im1,
    ax=[ax1, ax2],   
    location='bottom',
    shrink=0.4,
    pad=0.1
)

cbar12.set_label(r'$\tau$ (ms)')

# control third colorbar
cbar3 = fig.colorbar(im3, ax=ax3, location='bottom', shrink=0.6,  label=r'$\tau$ (ms)', pad=0.1)
vlim3 = (-vmax3, vmax3)
cbar3.set_ticks(np.linspace(-vmax3, vmax3, 5))
cbar3.ax.tick_params(labelsize=8)
# show only first decimal for tick labels
cbar3.set_ticklabels([f"{t:.1f}" for t in cbar3.get_ticks()])

# figure labels
ax1.text(-0.125, 1.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')
# save
fig.savefig(
    os.path.join(DERIV_DIR, 'figures', f'topos_lateralization_right_clusters_markers_{date}.png'),
    bbox_inches='tight'
)

