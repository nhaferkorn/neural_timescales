"""This script computes statistics for the timescale estimates.
Age Contrast (Fixation all"""

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


from mne.stats import ttest_ind_no_p
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# set seed for reproducibility
seed=13

# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")


# specify subjects & list to exclude
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]

# Old Adults vs. Young Adults 
infos = []
sub_young = []
sub_old = []
tau_old = []
tau_young = []
missing = []

# for log-transformed values
tau_young_log = []
tau_old_log =[]

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


# check number of old and young adults
# check numbers
print('# YOUNG', len(sub_young))
print('# OLD', len(sub_old))


# # for young adults
for sub in sub_young:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'{sub}_acf_params_fix_all.csv')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        # append info objects
        infos.append(info)
    
    if os.path.exists(timescales_path):
        df_young = pd.read_csv(timescales_path)

        # compute trial-average per channel 
        df_tau_young_trialavg = df_young.groupby('chs')['tau_fix_all'].mean()


        # also first log-transform single-trial tau values
        df_tau_young_log = (np.log(df_young['tau_fix_all'])
                            .groupby(df_young['chs'])
                            .mean()
                            )

        # reindex channels to match order of info object 
        df_tau_young_trialavg = df_tau_young_trialavg.reindex(info["ch_names"])

        df_tau_young_log = df_tau_young_log.reindex(info["ch_names"])
        
        # append to list
        tau_young.append(df_tau_young_trialavg)
        tau_young_log.append(df_tau_young_log)


# # for old adults
for sub in sub_old:

    # specify paths
    info_path = os.path.join(DERIV_DIR, 'info', f'{sub}_info.fif')
    timescales_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'{sub}_acf_params_fix_all.csv')

     # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)
        # append info objects
        infos.append(info)
    
    if os.path.exists(timescales_path):
        df_old = pd.read_csv(timescales_path)

        # compute trial-average per channel 
        df_tau_old_trialavg = df_old.groupby('chs')['tau_fix_all'].mean()

        # also first log-transform single-trial tau values
        df_tau_old_log = (np.log(df_old['tau_fix_all'])
                            .groupby(df_old['chs'])
                            .mean()
                            )

        # reindex channels to match order of info object 
        df_tau_old_trialavg = df_tau_old_trialavg.reindex(info["ch_names"])

        df_tau_old_log = df_tau_old_log.reindex(info["ch_names"])

        tau_old.append(df_tau_old_trialavg)
        tau_old_log.append(df_tau_old_log)

    


# # convert lists into numpy arrays
tau_young_array = np.stack(tau_young)
tau_old_array = np.stack(tau_old)

tau_young_log_array = np.stack(tau_young_log)
tau_old_log_array = np.stack(tau_old_log)


# check dimensions
print(tau_young_array.shape) # (n_sub, n_chs)
print(tau_old_array.shape)


# mean over channels
tau_young_mean = tau_young_array.mean(axis=1)
tau_old_mean = tau_old_array.mean(axis=1)


# # average over subjects 
tau_young_grandavg = tau_young_array.mean(axis=0) 
tau_old_grandavg = tau_old_array.mean(axis=0)
tau_diff_grandavg = tau_old_grandavg - tau_young_grandavg
tau_diff_young_old_grandavg = tau_young_grandavg - tau_old_grandavg

# # # compute standard deviation
# import statistics

# tau_mean_young = tau_young_array.mean(axis=(0, 1))
# tau_mean_old = tau_old_array.mean(axis=(0, 1))

# print('TAU MEAN YA', tau_mean_young)
# print('TAU MEAN OA', tau_mean_old)

# tau_young_sd = statistics.stdev(tau_young_grandavg)
# tau_old_sd = statistics.stdev(tau_old_grandavg)

# print('SD YOUNG', tau_young_sd)
# print('SD OLD', tau_old_sd)

# same for log arrays
tau_young_log_grandavg = tau_young_log_array.mean(axis=0)
tau_old_log_grandavg = tau_old_log_array.mean(axis=0)

# visualize input arrays
## Young vs. Old
fig = plt.figure()
fig.suptitle("Topographies: Age Contrast (Fix All)", fontsize = 'xx-large')
gs = GridSpec(1, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
im1, _ = mne.viz.plot_topomap(data = tau_young_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
im2, _ = mne.viz.plot_topomap(data = tau_old_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax2)
# add subtitles
ax1.set_title(f"YA (n={len(tau_young)})")
ax2.set_title(f"OA (n={len(tau_old)})")
# add colorbars directly next to the topomap axes
fig.colorbar(im1, ax=ax1, location = 'bottom', shrink=0.7, label='tau (s)',  pad = 0.1)
fig.colorbar(im2, ax=ax2, shrink=0.7, location = 'bottom', label='tau (s)', pad = 0.1)
fig.tight_layout()
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', f'topos_age_fix_all_{date}.png'))


## Difference Plot
fig, ax1 = plt.subplots(1)
fig.suptitle(f"Difference Plot: Old (n={len(tau_old)}) - Young (n={len(tau_young)})", fontsize = 'xx-large')
im1, _ = mne.viz.plot_topomap(data = tau_diff_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
# add subtitles
# ax1.set_title("Difference: High Dist - Low Dist")
fig.colorbar(im1, ax=ax1, location = 'right', shrink=0.7, label='tau (s)',  pad = 0.1)
fig.tight_layout()
fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'topos', f'topo_old_young_diff_{date}.png'))


# plot young, old and difference in one plot
fig = plt.figure(figsize=(10, 4))
# fig.suptitle("Topographies Age Effect", fontsize = 'xx-large')
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
im1, _ = mne.viz.plot_topomap(data = tau_young_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax1)
im2, _ = mne.viz.plot_topomap(data = tau_old_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax2)
im3, _ = mne.viz.plot_topomap(data = tau_diff_grandavg, pos = infos[1], cmap = 'viridis', ch_type='eeg', axes = ax3)
# add subtitles
ax1.set_title("Young")
ax2.set_title("Old")
ax3.set_title("Young-Old")
# add colorbars directly next to the topomap axes
fig.colorbar(im1, ax=ax1, location = 'bottom', shrink=0.6, label='tau (s)',  pad = 0.1)
fig.colorbar(im2, ax=ax2, shrink=0.6, location = 'bottom', label='tau (s)', pad = 0.1)
fig.colorbar(im3, ax=ax3, shrink=0.6, location = 'bottom', label='tau (s)', pad = 0.1)
# fig.subplots_adjust(top=0.1, wspace=0.4, hspace=0.4)
# add figure numbering
ax1.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', ha='center', va='center', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', ha='center', va='center', transform=ax3.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'topos_age_{date}.pdf'),bbox_inches='tight')


# # Plot Mean Tau for Old and Young Adults as Barplot
# tau_mean_young = tau_young_array.mean(axis=(0, 1))
# tau_mean_old = tau_old_array.mean(axis=(0, 1))

# print('TAU MEAN YA', tau_mean_young)
# print('TAU MEAN OA', tau_mean_old)

# # Create the figure and axis
# fig, ax = plt.subplots()

# ax.boxplot([tau_young_array.mean(axis=1), tau_old_array.mean(axis=1)],
#            labels=['Young', 'Old'])

# ax.set_ylabel('Mean Tau (s)')
# ax.set_title('Age Contrast: YA vs. OA', fontsize = 'xx-large')
# # # Add the legend
# # fig.legend(handles, legend_labels, loc='upper right', title="Age Group")

# # save plot
# fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'barplots', f'young_old_{date}.png'))


# Boxplot for Age Contrast:
colors = ['plum', 'rebeccapurple']
labels=['YA', 'OA']
tau_young_mean = tau_young_array.mean(axis=1)
tau_old_mean =  tau_old_array.mean(axis=1)

fig, ax = plt.subplots()

bplot = ax.boxplot([tau_young_mean*1000, tau_old_mean*1000],
                    # showmeans=True,
                    medianprops={"color": "white", "linewidth": 0.8},
                    patch_artist=True,
                    tick_labels=labels, 
                    widths = 0.5, showfliers=False)

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Tau (ms)', color='dimgray', fontsize=16)
ax.set_xlabel('Age Group',fontsize=16, color='dimgray')
ax.set_ylim(0, 100)
ax.tick_params(axis='y', color='b', length=4, direction='out',
                       labelcolor='dimgray', grid_color='blue',labelsize=14)
ax.tick_params(axis='x', color='b', length=4, direction='out',
                       labelcolor='dimgray', grid_color='blue', labelsize=14)
# ax.set_title(f'Age Contrast: Young (n={len(tau_young)}) vs. Old (n={len(tau_old)})', fontsize = 'x-large')
ax.set_title('All Fixation', fontsize = 'x-large', color='dimgray')
# remove top spine
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# get coordinates of axes
print(ax.get_position())
# Place label to the left of y-axis, aligned with the top (title level)
ax.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.tight_layout()

# save plot
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'boxplot_young_old_{date}.png'))



# add barplot for age contrast 
# colors = ['plum', 'rebeccapurple']
# labels = ['YA', 'OA']

# # turn numpy arrays into pandas dataframe in order to make better plots
# tau_young_pd = pd.DataFrame(tau_young_array.mean(axis=1))
# tau_old_pd = pd.DataFrame(tau_old_array.mean(axis=1))

# tau_young_mean = tau_young_array.mean()
# tau_old_mean = tau_old_array.mean()

# fig, ax = plt.subplots()

# bplot = ax.bar(
#     [0, 1],
#     [tau_young_mean * 1000, tau_old_mean * 1000],
#     tick_label=labels,
#     width=0.3
# )

# # fill with colors
# for patch, color in zip(bplot, colors):
#     patch.set_facecolor(color)

# ax.set_ylabel('Mean Tau (ms)', color='dimgray', fontsize=16)
# ax.set_xlabel('Age Group', fontsize=16, color='dimgray')
# ax.set_ylim(0, 100)

# ax.tick_params(axis='y', labelcolor='dimgray')
# ax.tick_params(axis='x', labelcolor='dimgray', labelsize=14)

# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)

# ax.text(-0.1, 1.0, 'a', transform=ax.transAxes,
#         fontsize=14, fontweight='bold')

# fig.tight_layout()
# # save plot
# fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'barplots', f'barplot_young_old_{date}.png'))

# col = tau_young_pd.columns[0]

# df = pd.concat([
#     tau_young_pd.rename(columns={col: "Tau"}).assign(Group='YA'),
#     tau_old_pd.rename(columns={col: "Tau"}).assign(Group='OA')
# ])

# # add seaborn barplot
# fig, ax = plt.subplots()

# sns.barplot(
#     data=df, x='Group', y='Tau',
#     errorbar="se", capsize=.1, alpha=1,
#     ax=ax, palette=colors, width=0.5)
# ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*1000:.0f}'))
# ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
# ax.set_xlabel('Age', fontsize=14, color='dimgray')
# ax.set_ylabel('Tau (ms)', fontsize=14, color='dimgray')
# # ax.set_xticklabels(['YA', 'OA'], fontsize=12)
# ax.set_ylim(0, 0.07)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # Style ticks (better approach)
# ax.tick_params(axis='x', labelsize=12, colors='dimgray')
# ax.tick_params(axis='y', labelsize=12, colors='dimgray')
# ax.text(-0.05, 1.1, 'a', transform=ax.transAxes,
#             fontsize=18, fontweight='bold', va='center')
# # ax.text(-0.125, 1.1, 'a', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
# fig.suptitle('All Fixation', fontsize=16, color='dimgray')
# fig.subplots_adjust(top=0.85)
# fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_fix_all_{date}.pdf'), bbox_inches="tight")



# # Statistical test - Independent Groups (permutation_cluster_test)
# tfce = dict(start=0, step=0.2)  # ideally start and step would be small
ch_adjacency, ch_names = mne.channels.read_ch_adjacency(fname='biosemi32')

# assert list(ch_names) == list(info["ch_names"]), 'clusters are wrong'

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

good_clusters_inds = np.where(cluster_p_values < 0.05)[0]
print("Good clusters: %i" % len(good_clusters_inds))
print('Good cluster indices', good_clusters_inds)


### final attempt - to actually plot all significant sensorsa
fig = plt.figure(figsize=(10, 4))
sel = mne.pick_types(infos[1], eeg=True)
info = mne.pick_info(infos[1], sel)
# how to mark sensors in the cluster
mask_params = dict(marker='.', markerfacecolor='w', markersize=11)
gs = GridSpec(1, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])


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
if len(good_clusters_inds) > 0:
    for clu_idx in good_clusters_inds:
        space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        mask[ch_inds] = True
else:
    print("No significant clusters found.")


# specify limits
vmax1 = np.max(np.abs([
    tau_young_grandavg*1000,
    tau_old_grandavg*1000,
]))

vmax1 = np.ceil(vmax1 / 20) * 20
vlim1 = (0, vmax1)
vmax3 = np.max(np.abs(tau_diff_young_old_grandavg*1000))
vmax3 = np.round(vmax3)
vlim3 = (-vmax3, vmax3)

# plot topomaps (always runs)
im1, _ = mne.viz.plot_topomap(
    tau_young_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax1
)

im2, _ = mne.viz.plot_topomap(
    tau_old_grandavg * 1000,
    info,
    cmap=cmocean.cm.ice,
    axes=ax2
)

im3, _ = mne.viz.plot_topomap(
    tau_diff_young_old_grandavg * 1000,
    info,
    cmap=cmocean.cm.balance,
    axes=ax3,
    mask=mask if mask.any() else None, 
    sensors=True,
    mask_params=mask_params,
    show=False
)

# titles
ax1.set_title("Young")
ax2.set_title("Old")
ax3.set_title("Young-Old")

# shared colorbar
cbar12 = fig.colorbar(
    im1,
    ax=[ax1, ax2],   
    location='bottom',
    shrink=0.4,
    pad=0.1
)

cbar12.set_label(r'$\tau$ (ms)')
cbar12.set_ticks([0, 20, 40, 60])
# cbar12.set_ticks(np.linspace(0, vmax1, 5))
# cbar12.set_ticklabels([f"{t:.1f}" for t in cbar12.get_ticks()])

# control third colorbar
cbar3 = fig.colorbar(im3, ax=ax3, location='bottom', shrink=0.6,  label=r'$\tau$ (ms)', pad=0.1)
vlim3 = (-vmax3, vmax3)
# cbar3.set_ticks(np.linspace(-vmax3, vmax3, 5))
cbar3.set_ticks(ticks = [-12, -6, 0, 6, 12])  # [-12, -8, -4, 0, 4, 8, 12])
cbar3.ax.tick_params(labelsize=8)
# show only first decimal for tick labels
cbar3.set_ticklabels([f"{t:.0f}" for t in cbar3.get_ticks()])

# figure labels
ax1.text(-0.125, 1.1, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold')
ax2.text(-0.125, 1.1, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold')
ax3.text(-0.125, 1.1, 'c', transform=ax3.transAxes, fontsize=14, fontweight='bold')
# save
fig.savefig(
    os.path.join(DERIV_DIR, 'figures', f'topos_age_markers_allclusters_{date}.pdf'),bbox_inches='tight')






# tau_old_subject_mean = tau_old_array.mean(axis=1)  # mean per subject
# tau_old_grandavg = tau_old_subject_mean.mean()     # mean across subjects

# tau_young_subject_mean = tau_young_array.mean(axis=1)
# tau_young_grandavg = tau_young_subject_mean.mean()

# print('Young Grandavg', tau_young_grandavg)
# print('Old GrandAvg', tau_old_grandavg)



# tau_old_subject_mean = tau_old_array.mean(axis=1)  # mean per subject
# tau_old_grandavg = tau_old_subject_mean.mean()     # mean across subjects

# tau_young_subject_mean = tau_young_array.mean(axis=1)
# tau_young_grandavg = tau_young_subject_mean.mean()

# print('Young Grandavg', tau_young_grandavg)
# print('Old GrandAvg', tau_old_grandavg)


# # mean across channels for each subject
# tau_old_subject_mean = tau_old_array.mean(axis=1)
# tau_young_subject_mean = tau_young_array.mean(axis=1)

# # grand mean across subjects
# tau_old_grandavg = tau_old_subject_mean.mean()
# tau_young_grandavg = tau_young_subject_mean.mean()

# # standard deviation across subjects
# tau_old_sd = tau_old_subject_mean.std(ddof=1)
# tau_young_sd = tau_young_subject_mean.std(ddof=1)

# print('Young Grandavg', tau_young_grandavg)
# print('Young SD', tau_young_sd)

# print('Old Grandavg', tau_old_grandavg)
# print('Old SD', tau_old_sd)


# one whole-scalp average per subject
tau_young_subject_mean = tau_young_array.mean(axis=1)
tau_old_subject_mean = tau_old_array.mean(axis=1)

# grand mean across subjects
tau_young_grandavg = tau_young_subject_mean.mean()
tau_old_grandavg = tau_old_subject_mean.mean()

# SD across subjects
tau_young_sd = tau_young_subject_mean.std(ddof=1)
tau_old_sd = tau_old_subject_mean.std(ddof=1)

print('Young Mean', tau_young_grandavg)
print('Young SD', tau_young_sd)

print('Old Mean', tau_old_grandavg)
print('Old SD', tau_old_sd)