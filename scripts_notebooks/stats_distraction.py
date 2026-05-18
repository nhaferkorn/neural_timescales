"""This script computes statistics for the distraction contrast and plots them as topographies."""

# make imports 
import os
import sys
import numpy as np
import scipy as sp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mne
import cmocean
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# specify subjects & list to exclude
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]

# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")

# Check how many poor model fit trials there are for high and low distraction trials (lower than 0.34%)
infos = []
subj = []
rsq_high = []
rsq_low = []
count_rsq_high_poor = []
count_rsq_low_poor = []
tau_raw_high = []
tau_raw_low = []
tau_log_high = []
tau_log_low = []


for subject in subjects:

    # construct filepath
    file_path_high = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_high.csv')
    file_path_low = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{subject}_acf_params_low.csv')
    info_path = os.path.join(DERIV_DIR, 'info', f'{subject}_info.fif')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)

    # check if file path exists 
    if os.path.exists(file_path_high):
        df_high = pd.read_csv(file_path_high)
    
    if os.path.exists(file_path_low):
        df_low = pd.read_csv(file_path_low)

        # collect info objects
        infos.append(info)

        # update subjects
        subj.append(subject)

        # create a new column for model fit
        df_high['model_fit'] = df_high['rsq_high'].apply(lambda x: 'good' if x > 0.34 else 'poor')
        df_low['model_fit'] = df_low['rsq_low'].apply(lambda x: 'good' if x > 0.34 else 'poor')

        # append count of rsq values
        rsq_high.append(df_high['rsq_high'].count())
        rsq_low.append(df_low['rsq_low'].count())

        # append number of poor fits to list
        count_rsq_high_poor.append(df_high['model_fit'].isin(['poor']).sum())
        count_rsq_low_poor.append(df_low['model_fit'].isin(['poor']).sum())

        # select only rows with good model fit
        df_high = df_high[df_high['model_fit'] == 'good']
        df_low = df_low[df_low['model_fit'] == 'good']

        # log-transform single-trial tau-value and store in new column
        df_high['tau_high_log'] = np.log(df_high['tau_high'])
        df_low['tau_low_log'] = np.log(df_low['tau_low'])
        
        # log-transformed trial average
        df_tau_high_log_trialavg = df_high.groupby('chs')['tau_high_log'].mean()
        df_tau_low_log_trialavg = df_low.groupby('chs')['tau_low_log'].mean()
        
        # raw trial-average (for plotting)
        df_tau_high_raw_trialavg = df_high.groupby('chs')['tau_high'].mean()
        df_tau_low_raw_trialavg = df_low.groupby('chs')['tau_low'].mean()

        # reindex channels to match order of info object 
        df_tau_high_log_trialavg = df_tau_high_log_trialavg.reindex(info["ch_names"])
        df_tau_low_log_trialavg = df_tau_low_log_trialavg.reindex(info["ch_names"])
    
        # reindex to match channel order of info object
        df_tau_high_raw_trialavg = df_tau_high_raw_trialavg.reindex(info["ch_names"])
        df_tau_low_raw_trialavg = df_tau_low_raw_trialavg.reindex(info["ch_names"])

        # append to list
        tau_raw_high.append(df_tau_high_raw_trialavg)
        tau_raw_low.append(df_tau_low_raw_trialavg)

        tau_log_high.append(df_tau_high_log_trialavg)
        tau_log_low.append(df_tau_low_log_trialavg)



# ## Summary Stats
# # # total number of trials
# print('Number of Trials')
# print(rsq_high)
# print(rsq_low)

# # mean number of poor trials rejected across all subjects
# print(np.array(count_rsq_high_poor).mean())
# print(np.array(count_rsq_low_poor).mean())

# # stack to get rsq arrays
# rsq_high_array = np.stack(rsq_high)
# rsq_low_array = np.stack(rsq_low)

# # compute & print grand mean of rsq
# rsq_high_grandavg = rsq_high_array.mean(axis=0)
# rsq_low_grandavg = rsq_low_array.mean(axis=0)

# print('PRINTING RSQ GRANDAVG')
# print('HIGH', rsq_high_grandavg)
# print('LOW', rsq_low_grandavg)

# stack arrays
tau_high_array = np.stack(tau_raw_high)
tau_low_array = np.stack(tau_raw_low)

# log-transformed arrays
tau_log_high_array = np.stack(tau_log_high)
tau_log_low_array = np.stack(tau_log_low)

print('Shape Tau High Array',tau_high_array.shape) # shape: subjects x channels

# # compute difference array
tau_diff_array = tau_high_array - tau_low_array

# log difference array
tau_log_diff_array = tau_log_high_array - tau_log_low_array # log-ratio

# average over subjects 
tau_high_grandavg = tau_high_array.mean(axis=0) 
tau_low_grandavg = tau_low_array.mean(axis=0)

# mean only over channels
tau_high_mean = tau_high_array.mean(axis=1)
tau_low_mean = tau_low_array.mean(axis=1)

print('TAU HIGH Trials', tau_high_mean)
print('TAU LOW Trials', tau_low_mean)

# compute difference array
tau_diff_grandavg = tau_high_grandavg - tau_low_grandavg


#########################################################################################################
# Plotting 
# Plot input data
fig = plt.figure(figsize=(10, 4))
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
im1, _ = mne.viz.plot_topomap(data = tau_high_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
im2, _ = mne.viz.plot_topomap(data = tau_low_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax2)
im3, _ = mne.viz.plot_topomap(data = tau_diff_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax3)
# add subtitles
ax1.set_title("High Dist")
ax2.set_title("Low Dist")
ax3.set_title("High Dist-Low Dist")
# add colorbars directly next to the topomap axes
fig.colorbar(im1, ax=ax1, location = 'bottom', shrink=0.5, label='tau (s)',  pad = 0.1)
fig.colorbar(im2, ax=ax2, shrink=0.5, location = 'bottom', label='tau (s)', pad = 0.1)
fig.colorbar(im3, ax=ax3, shrink=0.5, location = 'bottom', label='tau (s)', pad = 0.1)
# add figure numbering 
ax1.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'topos_distraction_{date}.pdf'),bbox_inches='tight')


## as boxplots
# boxplot with custom colors
tau_high_mean = tau_high_array.mean(axis=1)
tau_low_mean = tau_low_array.mean(axis=1)

colors = ['lightsteelblue', 'steelblue']
labels=['Low Dist', 'High Dist']
fig, ax = plt.subplots()

bplot = ax.boxplot([tau_low_mean*1000, tau_high_mean*1000],
                    # showmeans=True,
                    medianprops={"color": "white", "linewidth": 0.8},
                    patch_artist=True,
                    tick_labels=labels, widths=0.5)
# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Tau (ms)', fontsize=16, color='dimgray')
ax.set_xlabel('Condition',fontsize=16, color='dimgray')
ax.set_ylim(0, 100)
ax.tick_params(axis='y', color='gray', length=4, direction='out',
                       labelcolor='dimgray', grid_color='blue', labelsize=14)
ax.tick_params(axis='x', color='gray', length=4, direction='out',
                       labelcolor='dimgray', grid_color='blue', labelsize=14)

# remove top spine
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()

# save plot
fig.savefig(os.path.join(DERIV_DIR, 'figures',  f'boxplot_distraction_{date}.png'))



# barplot distraction contrast
# turn numpy arrays into pandas dataframe in order to make better plots
tau_high_pd = pd.DataFrame(tau_high_array.mean(axis=1))
tau_low_pd = pd.DataFrame(tau_low_array.mean(axis=1))


col = tau_high_pd.columns[0]

df = pd.concat([
    tau_low_pd.rename(columns={col: "Tau"}).assign(Group='Low Dist'),
    tau_high_pd.rename(columns={col: "Tau"}).assign(Group='High Dist')
])

# add seaborn barplot
fig, ax = plt.subplots()

sns.barplot(
    data=df, x='Group', y='Tau', order=['Low Dist', 'High Dist'],
    errorbar='se', capsize=.1, alpha=.8,
    ax=ax, palette=colors, width=0.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*1000:.0f}'))
ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
ax.set_xlabel('Condition', fontsize=14, color='dimgray')
ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
ax.set_ylim(0, 0.07) 
# ax.set_xticklabels(['YA', 'OA'], fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style ticks (better approach)
ax.tick_params(axis='x', labelsize=12, colors='dimgray')
ax.tick_params(axis='y', labelsize=12, colors='dimgray')
# fig.suptitle('All Fixation', fontsize=16, color='dimgray')
fig.subplots_adjust(top=0.85)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_distraction_{date}.pdf'), bbox_inches="tight")



# Compute Stats, maybe start with simple t-test
print('Computing Cluster Permutation Test')
p_value = 0.05

tail = 0 

# compute the threshold for our t-value 
# len refers to the first dimension of our data (here: number of sub)
deg_of_free = len(tau_log_diff_array) - 1
threshold = sp.stats.t.ppf(1 - p_value/ (1 + (tail == 0)), deg_of_free)

ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

cluster_stats = mne.stats.permutation_cluster_1samp_test(
    tau_log_diff_array,
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

# this kinda works, but is definitely not the most elegant way
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
print("Good clusters: %i" % len(good_cluster_inds))
print('Good cluster indices', good_cluster_inds)


pos = mne.find_layout(infos[1], ch_type='eeg').pos

# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):

    # unpack cluster information, get unique indices per cluster
    space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)

    # get topography for stats and average across time
    T_obs_map = T_obs

    # create a spatial mask
    mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True


# check info object
# ch_names = np.array(infos[1]['ch_names'])
# print(ch_names[ch_inds])

# for idx, name in enumerate(infos[1]['ch_names']):
#     print(idx, name)




# plot topographies with cluster sensors marked in white
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
    tau_high_grandavg * 1000,
    tau_low_grandavg * 1000
]))

vlim1 = (0, vmax)
vmax3 = np.max(np.abs(tau_diff_grandavg * 1000))
vmax3 = np.round(vmax3 * 2) / 2
vlim3 = (-vmax3, vmax3)

im1, _ = mne.viz.plot_topomap(
    tau_high_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax1,
    vlim=vlim1
)

im2, _ = mne.viz.plot_topomap(
    tau_low_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax2,
    vlim=vlim1
)

im3, _ = mne.viz.plot_topomap(
    tau_diff_grandavg * 1000,
    info,
    axes=ax3,
    cmap=cmocean.cm.balance,  
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

ax1.text(0.0, 1.015, "a", transform=ax1.transAxes,
         fontsize=16, fontweight='bold',
         va='bottom', ha='left')

ax2.text(0.0, 1.015, "b", transform=ax2.transAxes,
         fontsize=16, fontweight='bold',
         va='bottom', ha='left')

ax3.text(0.0, 1.015, "c", transform=ax3.transAxes,
         fontsize=16, fontweight='bold',
         va='bottom', ha='left')

# save
fig.savefig(
    os.path.join(DERIV_DIR, 'figures', f'topos_distraction_markers_allclusters_{date}.png'),
    bbox_inches='tight')


## compute descriptive stats
# mean across channels for each subject
tau_high_subject_mean = tau_high_array.mean(axis=1)
tau_low_subject_mean = tau_low_array.mean(axis=1)

# grand mean across subjects
tau_high_grandavg = tau_high_subject_mean.mean()
tau_low_grandavg = tau_low_subject_mean.mean()

# standard deviation across subjects
tau_high_sd = tau_high_subject_mean.std(ddof=1)
tau_low_sd = tau_low_subject_mean.std(ddof=1)

print('High Grandavg', tau_high_grandavg)
print('High SD', tau_high_sd)

print('Low Grandavg', tau_low_grandavg)
print('Low SD', tau_low_sd)