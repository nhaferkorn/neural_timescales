"""This script computes statistics for the timescale estimates.
Age Contrast (separarely or encoding and retrieval timescales)"""

# make imports 
import os
import sys
import numpy as np
import scipy as sp
import seaborn as sns
from scipy.stats import t
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import mne
import cmocean
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


from mne.stats import ttest_ind_no_p
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# set seed for reproducibility
seed=13
# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")

# read in data for multiple subjects
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]

# Old Adults vs. Young Adults 
infos = []
sub_young = []
sub_old = []
tau_enc_old = []
tau_enc_young = []
tau_enc_old_log = []
tau_enc_young_log = []

# lists for model fits
rsq_young = []
rsq_old = []
count_rsq_young_poor = []
count_rsq_old_poor = []

for sub in subjects:
    behavioral_path = os.path.join(
        DERIV_DIR, 'behavioral', 'summary_stats', f'{sub}_summary.csv'
    )

    if not os.path.exists(behavioral_path):
        continue

    df_behav = pd.read_csv(behavioral_path)

    if df_behav['age'].iloc[0] == 'young':
        sub_young.append(sub)
    else:
        sub_old.append(sub)


# check numbers
print('# YOUNG', len(sub_young))
print('# OLD', len(sub_old))


# # for young adults
for sub in sub_young:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_enc_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'encoding', f'{sub}_acf_params_encoding.csv')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        # append info objects
        infos.append(info)
    
    if os.path.exists(timescales_enc_path):
        df_young = pd.read_csv(timescales_enc_path)

        ## outlier removal
        # create a new column for model fit
        df_young['model_fit'] = df_young['rsq_encoding'].apply(lambda x: 'good' if x > 0.34 else 'poor')
     
        # append count of rsq values
        rsq_young.append(df_young['rsq_encoding'].count())
    
        # append number of poor fits to list
        count_rsq_young_poor.append(df_young['model_fit'].isin(['poor']).sum())
    
        # select only rows with good model fit
        df_young = df_young[df_young['model_fit'] == 'good']

        # log-transform single-trial tau-value and store in new column
        df_young['tau_encoding_log'] = np.log(df_young['tau_encoding'])

        # log tau estimate 
        df_tau_young_log_trialavg = df_young.groupby('chs')['tau_encoding_log'].mean()

        # raw trial-average (for plotting purposes)
        df_tau_young_raw_trialavg = df_young.groupby('chs')['tau_encoding'].mean()

        # reindex channels to match order of info object 
        df_tau_young_log_trialavg = df_tau_young_log_trialavg.reindex(info["ch_names"])
        df_tau_young_raw_trialavg = df_tau_young_raw_trialavg.reindex(info["ch_names"])
        
        # append
        tau_enc_young_log.append(df_tau_young_log_trialavg)
        tau_enc_young.append(df_tau_young_raw_trialavg)



# # for old adults
for sub in sub_old:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_enc_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'encoding', f'{sub}_acf_params_encoding.csv')

     # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        # append info objects
        infos.append(info)
    
    if os.path.exists(timescales_enc_path):
        df_old = pd.read_csv(timescales_enc_path)

        df_old['model_fit'] = df_old['rsq_encoding'].apply(lambda x: 'good' if x > 0.34 else 'poor')
     
        # append count of rsq values
        rsq_old.append(df_old['rsq_encoding'].count())
    
        # append number of poor fits to list
        count_rsq_old_poor.append(df_old['model_fit'].isin(['poor']).sum())
    
        # select only rows with good model fit
        df_old = df_old[df_old['model_fit'] == 'good']

        # raw tau encoding & log-transformed tau encoding
        df_old['tau_encoding_log'] = np.log(df_old['tau_encoding'])
        
        df_tau_old_log_trialavg = df_old.groupby('chs')['tau_encoding_log'].mean()

        # raw trial-average (for plotting purposes)
        df_tau_old_raw_trialavg = df_old.groupby('chs')['tau_encoding'].mean()

        # reindex channels to match order of info object 
        df_tau_old_log_trialavg = df_tau_old_log_trialavg.reindex(info["ch_names"])
        df_tau_old_raw_trialavg = df_tau_old_raw_trialavg.reindex(info["ch_names"])
        
        # append
        tau_enc_old_log.append(df_tau_old_log_trialavg)
        tau_enc_old.append(df_tau_old_raw_trialavg)


print('# TAU ENC YOUNG', len(tau_enc_young))
print('# TAU ENC OLD', len(tau_enc_old))

# print number of rejected trials
# total number of trials
print('Mean Number of Trials Young', np.array(rsq_young).mean())
print('Mean Number of Trials Old',np.array(rsq_old).mean())

# mean number of poor trials rejected across all subjects
print('Number of Poor Trials Young', np.array(count_rsq_young_poor).mean())
print('Number of Poor Trials Old', np.array(count_rsq_old_poor).mean())


# percentage rejected:
print('percent rejected young', (np.array(count_rsq_young_poor).mean()/ np.array(rsq_young).mean()))
print('percent rejected old', (np.array(count_rsq_old_poor).mean()/ np.array(rsq_old).mean()))

# # convert lists into numpy arrays
# for raw 
tau_young_raw_array = np.stack(tau_enc_young)
tau_old_raw_array = np.stack(tau_enc_old)


# for log-transformed
tau_young_log_array = np.stack(tau_enc_young_log)
tau_old_log_array = np.stack(tau_enc_old_log)


# mean over channels
tau_young_mean = tau_young_raw_array.mean(axis=1)
tau_old_mean = tau_old_raw_array.mean(axis=1)


# average over subjects 
tau_young_grandavg = tau_young_raw_array.mean(axis=0) 
tau_old_grandavg = tau_old_raw_array.mean(axis=0)
tau_diff_grandavg = tau_old_grandavg - tau_young_grandavg
tau_diff_young_old_grandavg = tau_young_grandavg - tau_old_grandavg


# Boxplot for Age Encoding Contrast 
# colors = ['lightcoral', 'maroon']
# labels=['YA', 'OA']

# fig, ax = plt.subplots()

# bplot = ax.boxplot([tau_young_mean, tau_old_mean],
#                     # showmeans=False,
#                     medianprops={"color": "white", "linewidth": 0.8},
#                     patch_artist=True,
#                     tick_labels=labels, 
#                     widths = 0.5)
# # fill with colors
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# ax.set_ylabel('Log Tau', fontsize=14, color='dimgray')
# # ax.set_ylim(0, 100)
# ax.tick_params(axis='y', color='gray', length=4, direction='out',
#                        labelcolor='dimgray', grid_color='blue', labelsize=14)
# ax.tick_params(axis='x', color='gray', length=4, direction='out',
#                        labelcolor='dimgray', grid_color='blue', labelsize=14)
# ax.set_title(f'Age Contrast Encoding: YA (n={len(tau_enc_young)}) vs. OA (n={len(tau_enc_old)})', fontsize = 'x-large')

# # remove top spine
# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)

# fig.tight_layout()

# # save plot
# fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'barplots', f'young_old_encoding_{date}.png'))


tau_young_pd = pd.DataFrame(tau_young_raw_array.mean(axis=1))
tau_old_pd = pd.DataFrame(tau_old_raw_array.mean(axis=1))

col = tau_young_pd.columns[0]


# define colors
colors = ['darkseagreen', 'seagreen']

df = pd.concat([
    tau_young_pd.rename(columns={col: "Tau"}).assign(Group='YA'),
    tau_old_pd.rename(columns={col: "Tau"}).assign(Group='OA')
])

fig, ax = plt.subplots()

sns.barplot(
    data=df, x='Group', y='Tau',
    errorbar="se", capsize=.1, alpha=1,
    ax=ax, palette=colors, width=0.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*1000:.0f}'))
ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
ax.set_xlabel('Age', fontsize=14, color='dimgray')
ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
ax.set_ylim(0, 0.07)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style ticks (better approach)
ax.tick_params(axis='x', labelsize=12, colors='dimgray')
ax.tick_params(axis='y', labelsize=12, colors='dimgray')
ax.text(-0.05, 1.1, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center')
# ax.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.suptitle('Encoding', fontsize=16, color='dimgray')
# fig.set_constrained_layout('constrained')
# fig.tight_layout()
fig.subplots_adjust(top=0.85)
# fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72, hspace=0.2,wspace=0.2)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_encoding.pdf'), bbox_inches="tight")


# # # average over subjects 
# # tau_young_grandavg = tau_young_array.mean(axis=0) 
# # tau_old_grandavg = tau_old_array.mean(axis=0)
# # tau_diff_grandavg = tau_old_grandavg - tau_young_grandavg

# # # visualize input arrays
# # ## Young vs. Old
# # fig = plt.figure()
# # fig.suptitle("Topographies: Age Contrast (Encoding)", fontsize = 'xx-large')
# # gs = GridSpec(1, 2)
# # ax1 = fig.add_subplot(gs[0, 0])
# # ax2 = fig.add_subplot(gs[0, 1])
# # im1, _ = mne.viz.plot_topomap(data = tau_young_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
# # im2, _ = mne.viz.plot_topomap(data = tau_old_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax2)
# # # add subtitles
# # ax1.set_title("YA")
# # ax2.set_title("OA")
# # # add colorbars directly next to the topomap axes
# # fig.colorbar(im1, ax=ax1, location = 'bottom', shrink=0.7, label='tau (s)',  pad = 0.1)
# # fig.colorbar(im2, ax=ax2, shrink=0.7, location = 'bottom', label='tau (s)', pad = 0.1)
# # fig.tight_layout()
# # fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', f'topos_age_encoding_{date}.png'))



# # Statistical test - Independent Groups (permutation_cluster_test)
# tfce = dict(start=0, step=0.2)  # ideally start and step would be small
ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

# set threshold for two-sided t-test
n1 = tau_old_log_array.shape[0]
n2 = tau_young_log_array.shape[0]

df = n1 + n2 - 2
pval = 0.05
thresh = t.ppf(1 - pval / 2, df)  

# run permutation test
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
    [tau_old_log_array, tau_young_log_array],
    n_permutations=5000,
    stat_fun=ttest_ind_no_p,
    adjacency = ch_adjacency,
    threshold=thresh,
    tail=0,
    seed=seed
)

print(clusters)
print(cluster_p_values)


# this kinda works, but is definitely not the most elegant way
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
print("Good clusters: %i" % len(good_cluster_inds))
print('Good cluster indices', good_cluster_inds)


# find the relevant sensors
pos = mne.find_layout(infos[1], ch_type='eeg').pos

T_obs_max = 5.
T_obs_min = -T_obs_max

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
ch_names = np.array(infos[1]['ch_names'])
print(ch_names[ch_inds])


print(infos[1]['ch_names'])

for idx, name in enumerate(infos[1]['ch_names']):
    print(idx, name)


# plot all significant sensors as white dots
fig = plt.figure(figsize=(10, 4))
sel = mne.pick_types(infos[1], eeg=True)
info = mne.pick_info(infos[1], sel)
# how to mark sensors in the cluster
mask_params = dict(marker='.', markerfacecolor='w', markersize=11)
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# find the relevant sensors
pos = mne.find_layout(infos[1], ch_type='eeg').pos

T_obs_max = 5.
T_obs_min = -T_obs_max

vmax = np.max(np.abs(tau_diff_grandavg))
vmin = -vmax

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


    im1, _ = mne.viz.plot_topomap(data = tau_young_grandavg*1000, pos = info, cmap = 'viridis', ch_type='eeg', axes = ax1)
    im2, _ = mne.viz.plot_topomap(data = tau_old_grandavg*1000, pos = info, cmap = 'viridis', ch_type='eeg', axes = ax2)
    im3, _ = mne.viz.plot_topomap(tau_diff_young_old_grandavg*1000, info, ch_type='eeg', mask=mask, 
                            axes=ax3, sensors=False,
                            mask_params=mask_params,
                            # vlim=(vmin, vmax),
                            show=False,cmap='viridis')
    # add subtitles
    ax1.set_title("Young")
    ax2.set_title("Old")
    ax3.set_title("Young-Old")
# add colorbars directly next to the topomap axes
fig.colorbar(im1, ax=ax1, location = 'bottom', shrink=0.5, label=r'$\tau$ (ms)',  pad = 0.1)
fig.colorbar(im2, ax=ax2, shrink=0.5, location = 'bottom', label=r'$\tau$ (ms)', pad = 0.1)
fig.colorbar(im3, ax=ax3, shrink=0.5, location = 'bottom', label=r'$\tau$ (ms)', pad = 0.1)

# add figure numbering 
ax1.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'topos_age_encoding_clusters_{date}.pdf'),bbox_inches='tight')


# only plot the difference plot
# find the relevant sensors
pos = mne.find_layout(infos[1], ch_type='eeg').pos

T_obs_max = 5.
T_obs_min = -T_obs_max

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
ch_names = np.array(infos[1]['ch_names'])
# print(ch_names[ch_inds])

# print(infos[1]['ch_names'])

# for idx, name in enumerate(infos[1]['ch_names']):
#     print(idx, name)


diff = tau_diff_young_old_grandavg * 1000

limit = np.max(np.abs(diff))
limit = np.ceil(limit)

norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)


# plot only difference plot
fig, ax = plt.subplots(figsize=(4,4))

sel = mne.pick_types(infos[1], eeg=True)
info = mne.pick_info(infos[1], sel)

mask_params = dict(
    marker='o',
    markerfacecolor='white',
    markeredgecolor='black',
    # linewidth=1.2,
    markersize=6
)

mask = np.zeros(info['nchan'], dtype=bool)
for clu_idx in good_cluster_inds:
    space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    mask[ch_inds] = True

im, _ = mne.viz.plot_topomap(
    tau_diff_young_old_grandavg * 1000,
    info,
    mask=mask,
    axes=ax,
    cmap=cmocean.cm.balance,  
    cnorm=norm,
    sensors=True,
    mask_params=mask_params,
    show=False
)

ax.set_title("Encoding: Young-Old")

cbar = fig.colorbar(im, ax=ax, location='bottom', shrink=0.5,
             label=r'$\tau$ (ms)', pad=0.1)

step = limit / 2
ticks = [-limit, -step, 0, step, limit]
ticks = [int(t) for t in ticks]

cbar.set_ticks(ticks)

ax.text(-0.125, 1.1, 'a',
        transform=ax.transAxes,
        fontsize=14, fontweight='bold')

fig.tight_layout()

fig.savefig(os.path.join(DERIV_DIR, 'figures',
            f'topos_age_difference_encoding_clusters_{date}.pdf'))


### plot with new colormap
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
    tau_young_grandavg * 1000,
    tau_old_grandavg * 1000
]))

vlim1 = (0, vmax)
vmax3 = np.max(np.abs(tau_diff_grandavg * 1000))
vmax3 = np.round(vmax3 * 2) / 2
vlim3 = (-vmax3, vmax3)

# plot topomaps (always runs)
im1, _ = mne.viz.plot_topomap(
    tau_young_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax1,
    vlim=vlim1
)

im2, _ = mne.viz.plot_topomap(
    tau_old_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax2,
    vlim=vlim1
)

im3, _ = mne.viz.plot_topomap(
    tau_diff_young_old_grandavg * 1000,
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
ax1.set_title("Young")
ax2.set_title("Old")
ax3.set_title("Young-Old")

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
    os.path.join(DERIV_DIR, 'figures', f'topos_age_markers_enc_allclusters_colormap_{date}.png'),
    bbox_inches='tight')

