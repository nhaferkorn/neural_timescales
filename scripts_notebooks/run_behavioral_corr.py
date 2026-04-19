"""This script computes and visualizes timescale and cognitive-behavioral correlations. """

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import cmocean
import textwrap
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from datetime import datetime
from functools import reduce
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter


# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.behavioral import read_behav_data, process_enc_data, process_ret_data, calculate_encoding_task_performance, calculate_retrieval_task_performance, create_behavioral_summary, calculate_hitrate, calculate_fa_rate, corr_heatmap_with_pval

# specify subjects & list to exclude
exclude = [102, 105, 115, 116, 123]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]


## read behavioral data 
dfs = []

for sub in subjects:

    # read summary data
    file_path = os.path.join(DERIV_DIR, 'behavioral', 'summary_stats', f'{sub}_summary.csv')

    # load encoding data
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dfs.append(df)

df_behav = pd.concat(dfs, axis = 0, join='outer')


dfs_enc = []
dfs_ret = []
dfs_all = []

## load timescale data
# load model parameters for each subject 
for sub in subjects:
    
    # file path
    enc_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'encoding', f'{sub}_acf_params_encoding.csv')
    ret_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'retrieval', f'{sub}_acf_params_retrieval.csv')
    all_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'fix_all', f'{sub}_acf_params_fix_all.csv')
    
    # check if file exists before loading
    if os.path.exists(enc_path):
        df_enc = pd.read_csv(enc_path)
        dfs_enc.append(df_enc)
    
    if os.path.exists(ret_path):
        df_ret = pd.read_csv(ret_path)
        dfs_ret.append(df_ret)
    
    if os.path.exists(all_path):
        df_all = pd.read_csv(all_path)
        dfs_all.append(df_all)

# concatenate into one file (subjects will be stacked on top of each other)
df_encoding = pd.concat(dfs_enc, axis = 0, join='outer')
df_retrieval = pd.concat(dfs_ret, axis = 0, join='outer')
df_all = pd.concat(dfs_all, axis = 0, join='outer')


# select only sensors that were part of significant cluster, with lowest p-value
clusters = ['CP5', 'P7', 'P3' ,'O1']


df_enc_clustered = df_encoding[df_encoding['chs'].isin(clusters)]
df_ret_clustered = df_retrieval[df_retrieval['chs'].isin(clusters)]
df_all_clustered = df_all[df_all['chs'].isin(clusters)]

# add log columns to original clustered data
df_enc_clustered['log_tau_encoding'] = np.log(df_enc_clustered['tau_encoding'])
df_ret_clustered['log_tau_retrieval'] = np.log(df_ret_clustered['tau_retrieval'])
df_all_clustered['log_tau_fix_all'] = np.log(df_all_clustered['tau_fix_all'])

## Log Estimates
# then average per subject
df_avg_log_enc_tau = (
    df_enc_clustered
    .groupby('sub')['log_tau_encoding']
    .mean()
    .reset_index()
)

df_avg_log_ret_tau = (
    df_ret_clustered
    .groupby('sub')['log_tau_retrieval']
    .mean()
    .reset_index()
)


## Raw Estimates
df_avg_raw_enc_tau = (
    df_enc_clustered
    .groupby('sub')['tau_encoding']
    .mean()
    .reset_index()
)

df_avg_raw_ret_tau = (
    df_ret_clustered
    .groupby('sub')['tau_retrieval']
    .mean()
    .reset_index()
)

df_avg_raw_all_tau = (
    df_all_clustered 
    .groupby('sub')['tau_fix_all']
    .mean()
    .reset_index()
)



## also load distraction timescale estimates
# load model parameters for each subject 
# dfs_high = []
# dfs_low = []

# for sub in subjects:
    
#     # file path
#     high_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{sub}_acf_params_high_good.csv')
#     low_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'{sub}_acf_params_low_good.csv')


#     # check if file exists before loading
#     if os.path.exists(high_path):
#         df_high = pd.read_csv(high_path)
#         dfs_high.append(df_high)
    
#     if os.path.exists(low_path):
#         df_low = pd.read_csv(low_path)
#         dfs_low.append(df_low)

# # concatenate into one file (subjects stacked on top of each other)
# df_high = pd.concat(dfs_high, axis = 0, join='outer')
# df_low = pd.concat(dfs_low, axis = 0, join='outer')


# # print(df_high.head())
# # print(df_low.head())

# # select only sensors that were part of significant 
# clusters = ['CP5' 'P7' 'P3' 'O1']
# df_high_clustered = df_high[df_high['chs'].isin(clusters)]
# df_low_clustered = df_low[df_low['chs'].isin(clusters)]

# print(df_high_clustered)
# print(df_low_clustered)

# # grand average high dist tau per subject
# # df_avg_high_tau = df_high['tau_high'].groupby(df_high['sub']).mean()
# # print(df_avg_high_tau)

# # # grand average retrieval tau per subject
# # df_avg_low_tau = df_low['tau_low'].groupby(df_low['sub']).mean()
# # print(df_avg_low_tau)

# ## clustered subject means
# df_avg_clu_high_tau = df_high_clustered['tau_high'].groupby(df_high_clustered['sub']).mean()
# print(df_avg_clu_high_tau)
# df_avg_clu_low_tau = df_low_clustered['tau_low'].groupby(df_low_clustered['sub']).mean()
# print(df_avg_clu_low_tau)

# # but also log-transform them
# df_log_avg_high_tau = (
#     np.log(df_high_clustered['tau_high'])
#     .groupby(df_high_clustered['sub'])
#     .mean()
#     .reset_index()
# )

# df_log_avg_low_tau = (
#     np.log(df_low_clustered['tau_low'])
#     .groupby(df_low_clustered['sub'])
#     .mean()
#     .reset_index()
# )



# merge timescale data and behavioral data
dfs = [df_behav, df_avg_log_enc_tau, df_avg_log_ret_tau, df_avg_raw_enc_tau, df_avg_raw_ret_tau, df_avg_raw_all_tau]
df_merged = reduce(lambda left, right: pd.merge(left, right, on="sub"), dfs)

# set paths
path_enc = os.path.join(DERIV_DIR, 'figures', f'corrmap_clusters_rawtau_rawrt_encoding_{date}.png')
path_ret = os.path.join(DERIV_DIR, 'figures', f'corrmap_clusters_rawtau_rawrt_retrieval_{date}.png')
path_all = os.path.join(DERIV_DIR, 'figures', f'corrmap_clusters_rawtau_fix_all_{date}.png')

# # for encoding timescales (log edition)
# labels = {
#     'log_tau_encoding': 'Tau (Enc)',
#     'RT_enc_log_mean': 'RT (Enc)',
#     'hitrate_targets': 'Hits (Targets)'
# }

# data = df_merged[['log_tau_encoding','RT_enc_log_mean','hitrate_targets']].copy()

# # optional relabeling
# data = data.rename(columns=labels)

# corr = data.corr(method='spearman')

# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
# np.fill_diagonal(mask, False)


# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

# ax = fig.add_subplot(gs[0])      # heatmap axis
# cax = fig.add_subplot(gs[1])     # colorbar axis


# sns.heatmap(
#     corr,
#     annot=True,
#     annot_kws={"fontsize": 18},
#     fmt='.2f',
#     linewidths=0.5,
#     cmap=cmocean.cm.tempo,
#     mask=mask,
#     ax=ax,
#     cbar_ax=cax
# )

# ax.set_title('Encoding Correlations: Log Tau & Log RT', fontsize=20, pad=20)
# cax.set_ylabel(
#     r'Correlation ($\rho$)',
#     fontsize=12,
#     rotation=90,
#     labelpad=20,
#     va='center'   # vertical alignment
# )


# # add p-values
# # Create a function to calculate and format p-values
# p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[1]):
#         x = data.iloc[:, i]
#         y = data.iloc[:, j]
#         mask = ~np.logical_or(np.isnan(x), np.isnan(y))
#         if np.sum(mask) > 0:
#             p_values[i, j] = spearmanr(x[mask], y[mask])[1]

# p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# # Create a mask for the p-values heatmap
# mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# # Calculate the highest and lowest correlation coefficients
# max_corr = np.max(corr.max())
# min_corr = np.min(corr.min())

# # Get colormap + normalization from heatmap
# cmap = ax.collections[0].cmap
# norm = ax.collections[0].norm

# # Annotate the heatmap with p-values and change text color based on correlation value
# for i in range(p_values.shape[0]):
#     for j in range(p_values.shape[1]):
#         if mask_pvalues[i, j]:
#             p_value = p_values.iloc[i, j]
#             if not np.isnan(p_value):
#                 correlation_value = corr.iloc[i, j]
#                 # luminance based text colors
#                 rgba = cmap(norm(correlation_value))
#                 r, g, b, _ = rgba
#                 luminance = 0.299*r + 0.587*g + 0.114*b
#                 text_color = 'white' if luminance < 0.5 else 'black'

#                 if p_value <= 0.01:
#                 #include double asterisks for p-value <= 0.01
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)
#                 elif p_value <= 0.05:
#                 #include single asterisks for p-value <= 0.05
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)
#                 else:
#                     ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)

# # wrap labels
# x_labels = [textwrap.fill(t.get_text(), 15) for t in ax.get_xticklabels()]
# y_labels = [textwrap.fill(t.get_text(), 15) for t in ax.get_yticklabels()]

# ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
# ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)

# # ax.tick_params(axis='both', labelsize=14)

# # fig.subplots_adjust(top=0.85)

# # fig.tight_layout()
# fig.savefig(path_enc_log, bbox_inches='tight')


## Correlation Matrix - Encoding Phase
labels = {
    'tau_encoding': 'Tau (Enc)',
    'RT_enc_mean': 'RT (Enc)',
    #'hitrate_targets': 'Hits (Targets)',
    'enc_performance': 'Performance (Enc)'
}

data = df_merged[['tau_encoding','RT_enc_mean','enc_performance']].copy()

# optional relabeling
data = data.rename(columns=labels)

corr = data.corr(method='spearman')

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask, False)


fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

ax = fig.add_subplot(gs[0])      # heatmap axis
cax = fig.add_subplot(gs[1])     # colorbar axis


sns.heatmap(
    corr,
    annot=True,
    annot_kws={"fontsize": 18},
    fmt='.2f',
    linewidths=0.5,
    cmap=cmocean.cm.matter,
    mask=mask,
    ax=ax,
    cbar_ax=cax
)

# ax.set_title('Retrieval Correlations: Raw Tau & Raw RT', fontsize=20, pad=20)
cax.set_ylabel(
    r'Correlation ($\rho$)',
    fontsize=13,
    rotation=90,
    labelpad=20,
    va='center'   # vertical alignment
)

cax.yaxis.set_ticks_position('left')

# add p-values
# Create a function to calculate and format p-values
p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[1]):
        x = data.iloc[:, i]
        y = data.iloc[:, j]
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        if np.sum(mask) > 0:
            p_values[i, j] = spearmanr(x[mask], y[mask])[1]

p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# Create a mask for the p-values heatmap
mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# Calculate the highest and lowest correlation coefficients
max_corr = np.max(corr.max())
min_corr = np.min(corr.min())

# Get colormap + normalization from heatmap
cmap = ax.collections[0].cmap
norm = ax.collections[0].norm

# Annotate the heatmap with p-values and change text color based on correlation value
for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
        if mask_pvalues[i, j]:
            p_value = p_values.iloc[i, j]
            if not np.isnan(p_value):
                correlation_value = corr.iloc[i, j]
                # luminance based text colors
                rgba = cmap(norm(correlation_value))
                r, g, b, _ = rgba
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_color = 'white' if luminance < 0.5 else 'black'

                if p_value <= 0.01:
                #include double asterisks for p-value <= 0.01
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                elif p_value <= 0.05:
                #include single asterisks for p-value <= 0.05
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                else:
                    ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)

# wrap labels
x_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_xticklabels()]
y_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_yticklabels()]

ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)

# fig.tight_layout()
fig.savefig(path_enc,  bbox_inches='tight')


##################################################################################################

# # for retrieval phase (log values)
# labels = {
#     'log_tau_retrieval': 'Tau (Ret)',
#     'RT_ret_log_mean': 'RT (Ret)',
#     'hitrate_targets': 'Hits (Targets)'
# }

# data = df_merged[['log_tau_retrieval','RT_ret_log_mean','hitrate_targets']].copy()

# # optional relabeling
# data = data.rename(columns=labels)

# corr = data.corr(method='spearman')

# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
# np.fill_diagonal(mask, False)

# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

# ax = fig.add_subplot(gs[0])      # heatmap axis
# cax = fig.add_subplot(gs[1])     # colorbar axis


# sns.heatmap(
#     corr,
#     annot=True,
#     annot_kws={"fontsize": 18},
#     fmt='.2f',
#     linewidths=0.5,
#     cmap=cmocean.cm.matter,
#     mask=mask,
#     ax=ax,
#     cbar_ax=cax
#     #cbar_kws=dict(location="left", orientation='horizontal')
# )

# ax.set_title('Retrieval Correlations: Log Tau & Log RT', fontsize=20, pad=20)
# cax.set_ylabel(
#     r'Correlation ($\rho$)',
#     fontsize=12,
#     rotation=90,
#     labelpad=20,
#     va='center'   # vertical alignment
# )


# # add p-values
# # Create a function to calculate and format p-values
# p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[1]):
#         x = data.iloc[:, i]
#         y = data.iloc[:, j]
#         mask = ~np.logical_or(np.isnan(x), np.isnan(y))
#         if np.sum(mask) > 0:
#             p_values[i, j] = spearmanr(x[mask], y[mask])[1]

# p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# # Create a mask for the p-values heatmap
# mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# # Calculate the highest and lowest correlation coefficients
# max_corr = np.max(corr.max())
# min_corr = np.min(corr.min())

# # Get colormap + normalization from heatmap
# cmap = ax.collections[0].cmap
# norm = ax.collections[0].norm

# # Annotate the heatmap with p-values and change text color based on correlation value
# for i in range(p_values.shape[0]):
#     for j in range(p_values.shape[1]):
#         if mask_pvalues[i, j]:
#             p_value = p_values.iloc[i, j]
#             if not np.isnan(p_value):
#                 correlation_value = corr.iloc[i, j]
#                 # luminance based text colors
#                 rgba = cmap(norm(correlation_value))
#                 r, g, b, _ = rgba
#                 luminance = 0.299*r + 0.587*g + 0.114*b
#                 text_color = 'white' if luminance < 0.5 else 'black'

#                 if p_value <= 0.01:
#                 #include double asterisks for p-value <= 0.01
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)
#                 elif p_value <= 0.05:
#                 #include single asterisks for p-value <= 0.05
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)
#                 else:
#                     ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=10,
#                             color=text_color)

# # wrap labels
# x_labels = [textwrap.fill(t.get_text(), 15) for t in ax.get_xticklabels()]
# y_labels = [textwrap.fill(t.get_text(), 15) for t in ax.get_yticklabels()]

# ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
# ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)


# # fig.tight_layout()
# fig.savefig(path_ret_log, bbox_inches='tight')



####
# for retrieval phase (raw values)
labels = {
    'tau_retrieval': 'Tau (Ret)',
    'RT_ret_mean': 'RT (Ret)',
    'hitrate_targets': 'Hits (Targets)',
    'hitrate_distractors': 'Hits (Distractors)',
    'hitrate_targets_highconf': 'Hits (High Conf)'
    # 'fa_rate': 'False Alarms'
}

data = df_merged[['tau_retrieval','RT_ret_mean','hitrate_targets', 'hitrate_distractors', 'hitrate_targets_highconf']].copy()

# optional relabeling
data = data.rename(columns=labels)

corr = data.corr(method='spearman')

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask, False)

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

ax = fig.add_subplot(gs[0])      # heatmap axis
cax = fig.add_subplot(gs[1])     # colorbar axis


sns.heatmap(
    corr,
    annot=True,
    annot_kws={"fontsize": 18},
    fmt='.2f',
    linewidths=0.5,
    cmap=cmocean.cm.tempo,
    mask=mask,
    ax=ax,
    cbar_ax=cax
)

# ax.set_title('Retrieval Correlations: Raw Tau & Raw RT', fontsize=20, pad=20)
cax.set_ylabel(
    r'Correlation ($\rho$)',
    fontsize=12,
    rotation=90,
    labelpad=20,
    va='center'   # vertical alignment
)


# add p-values
# Create a function to calculate and format p-values
p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[1]):
        x = data.iloc[:, i]
        y = data.iloc[:, j]
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        if np.sum(mask) > 0:
            p_values[i, j] = spearmanr(x[mask], y[mask])[1]

p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# Create a mask for the p-values heatmap
mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# Calculate the highest and lowest correlation coefficients
max_corr = np.max(corr.max())
min_corr = np.min(corr.min())

# Get colormap + normalization from heatmap
cmap = ax.collections[0].cmap
norm = ax.collections[0].norm

# Annotate the heatmap with p-values and change text color based on correlation value
for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
        if mask_pvalues[i, j]:
            p_value = p_values.iloc[i, j]
            if not np.isnan(p_value):
                correlation_value = corr.iloc[i, j]
                # luminance based text colors
                rgba = cmap(norm(correlation_value))
                r, g, b, _ = rgba
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_color = 'white' if luminance < 0.5 else 'black'

                if p_value <= 0.01:
                #include double asterisks for p-value <= 0.01
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                elif p_value <= 0.05:
                #include single asterisks for p-value <= 0.05
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                else:
                    ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)

# wrap labels
x_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_xticklabels()]
y_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_yticklabels()]

ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)


# fig.tight_layout()
fig.savefig(path_ret, bbox_inches='tight')



# ## Correlation Matrix - Encoding Phase
# labels = {
#     'tau_encoding': 'Tau (Enc)',
#     'RT_enc_mean': 'RT (Enc)',
#     #'hitrate_targets': 'Hits (Targets)',
#     'enc_performance': 'Classification (Enc)'
# }

# data = df_merged[['tau_encoding','RT_enc_mean','enc_performance']].copy()

# # optional relabeling
# data = data.rename(columns=labels)

# corr = data.corr(method='spearman')

# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
# np.fill_diagonal(mask, False)


# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

# ax = fig.add_subplot(gs[0])      # heatmap axis
# cax = fig.add_subplot(gs[1])     # colorbar axis


# sns.heatmap(
#     corr,
#     annot=True,
#     annot_kws={"fontsize": 18},
#     fmt='.2f',
#     linewidths=0.5,
#     cmap=cmocean.cm.tempo,
#     mask=mask,
#     ax=ax,
#     cbar_ax=cax
# )

# # ax.set_title('Encoding Correlations: Raw Tau & Raw RT', fontsize=20, pad=20)
# cax.set_ylabel(
#     r'Correlation ($\rho$)',
#     fontsize=12,
#     rotation=90,
#     labelpad=20,
#     va='center'   # vertical alignment
# )


# # add p-values
# # Create a function to calculate and format p-values
# p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[1]):
#         x = data.iloc[:, i]
#         y = data.iloc[:, j]
#         mask = ~np.logical_or(np.isnan(x), np.isnan(y))
#         if np.sum(mask) > 0:
#             p_values[i, j] = spearmanr(x[mask], y[mask])[1]

# p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# # Create a mask for the p-values heatmap
# mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# # Calculate the highest and lowest correlation coefficients
# max_corr = np.max(corr.max())
# min_corr = np.min(corr.min())

# # Get colormap + normalization from heatmap
# cmap = ax.collections[0].cmap
# norm = ax.collections[0].norm

# # Annotate the heatmap with p-values and change text color based on correlation value
# for i in range(p_values.shape[0]):
#     for j in range(p_values.shape[1]):
#         if mask_pvalues[i, j]:
#             p_value = p_values.iloc[i, j]
#             if not np.isnan(p_value):
#                 correlation_value = corr.iloc[i, j]
#                 # luminance based text colors
#                 rgba = cmap(norm(correlation_value))
#                 r, g, b, _ = rgba
#                 luminance = 0.299*r + 0.587*g + 0.114*b
#                 text_color = 'white' if luminance < 0.5 else 'black'

#                 if p_value <= 0.01:
#                 #include double asterisks for p-value <= 0.01
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)
#                 elif p_value <= 0.05:
#                 #include single asterisks for p-value <= 0.05
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)
#                 else:
#                     ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)

# # wrap labels
# x_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_xticklabels()]
# y_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_yticklabels()]

# ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
# ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)


# # fig.tight_layout()
# fig.savefig(path_enc_raw, bbox_inches='tight')


################################################################################################################


## Correlation Matrix - Across Phases (Encoding & Retrieval)
labels = {
    'tau_fix_all': 'Tau (Global)',
    'RT_enc_mean': 'RT (Enc)',
    'RT_ret_mean': 'RT (Ret)',
    'hitrate_targets': 'Hits (Targets)',
    'hitrate_distractors': 'Hits (Distractors)'
}

data = df_merged[['tau_fix_all','RT_enc_mean','RT_ret_mean', 'hitrate_targets', 'hitrate_distractors']].copy()

# optional relabeling
data = data.rename(columns=labels)

corr = data.corr(method='spearman')

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
np.fill_diagonal(mask, False)


fig = plt.figure(figsize=(12, 8))
gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

ax = fig.add_subplot(gs[0])      # heatmap axis
cax = fig.add_subplot(gs[1])     # colorbar axis


sns.heatmap(
    corr,
    annot=True,
    annot_kws={"fontsize": 18},
    fmt='.2f',
    linewidths=0.5,
    cmap=cmocean.cm.tempo,
    mask=mask,
    ax=ax,
    cbar_ax=cax
)

# ax.set_title('Encoding Correlations: Raw Tau & Raw RT', fontsize=20, pad=20)
cax.set_ylabel(
    r'Correlation ($\rho$)',
    fontsize=12,
    rotation=90,
    labelpad=20,
    va='center'   # vertical alignment
)


# add p-values
# Create a function to calculate and format p-values
p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[1]):
        x = data.iloc[:, i]
        y = data.iloc[:, j]
        mask = ~np.logical_or(np.isnan(x), np.isnan(y))
        if np.sum(mask) > 0:
            p_values[i, j] = spearmanr(x[mask], y[mask])[1]

p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# Create a mask for the p-values heatmap
mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# Calculate the highest and lowest correlation coefficients
max_corr = np.max(corr.max())
min_corr = np.min(corr.min())

# Get colormap + normalization from heatmap
cmap = ax.collections[0].cmap
norm = ax.collections[0].norm

# Annotate the heatmap with p-values and change text color based on correlation value
for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
        if mask_pvalues[i, j]:
            p_value = p_values.iloc[i, j]
            if not np.isnan(p_value):
                correlation_value = corr.iloc[i, j]
                # luminance based text colors
                rgba = cmap(norm(correlation_value))
                r, g, b, _ = rgba
                luminance = 0.299*r + 0.587*g + 0.114*b
                text_color = 'white' if luminance < 0.5 else 'black'

                if p_value <= 0.01:
                #include double asterisks for p-value <= 0.01
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                elif p_value <= 0.05:
                #include single asterisks for p-value <= 0.05
                    ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)
                else:
                    ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=12,
                            color=text_color)

# wrap labels
x_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_xticklabels()]
y_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_yticklabels()]

ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)

# fig.tight_layout()
fig.savefig(path_all, bbox_inches='tight')





#########################################################################################
## Correlation Matrix - Across Phases (Encoding & Retrieval)
# path_enc_ret = os.path.join(DERIV_DIR, 'figures', f'corrmap_clusters_raw_enc_ret_across_phase_{date}.png')


# labels = {
#     'tau_encoding': 'Tau (Enc)',
#     'tau_retrieval': 'Tau (Ret)',
#     'RT_ret_mean': 'RT (Ret)',
#     'hitrate_targets': 'Hits (Targets)',
# }

# data = df_merged[['tau_encoding','tau_retrieval','RT_ret_mean', 'hitrate_targets']].copy()

# # optional relabeling
# data = data.rename(columns=labels)

# corr = data.corr(method='spearman')

# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
# np.fill_diagonal(mask, False)


# fig = plt.figure(figsize=(12, 8))
# gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.15)

# ax = fig.add_subplot(gs[0])      # heatmap axis
# cax = fig.add_subplot(gs[1])     # colorbar axis


# sns.heatmap(
#     corr,
#     annot=True,
#     annot_kws={"fontsize": 18},
#     fmt='.2f',
#     linewidths=0.5,
#     cmap=cmocean.cm.tempo,
#     mask=mask,
#     ax=ax,
#     cbar_ax=cax
# )

# # ax.set_title('Encoding Correlations: Raw Tau & Raw RT', fontsize=20, pad=20)
# cax.set_ylabel(
#     r'Correlation ($\rho$)',
#     fontsize=12,
#     rotation=90,
#     labelpad=20,
#     va='center'   # vertical alignment
# )


# # add p-values
# # Create a function to calculate and format p-values
# p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
# for i in range(corr.shape[0]):
#     for j in range(i+1, corr.shape[1]):
#         x = data.iloc[:, i]
#         y = data.iloc[:, j]
#         mask = ~np.logical_or(np.isnan(x), np.isnan(y))
#         if np.sum(mask) > 0:
#             p_values[i, j] = spearmanr(x[mask], y[mask])[1]

# p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

# # Create a mask for the p-values heatmap
# mask_pvalues = np.triu(np.ones_like(p_values), k=1)

# # Calculate the highest and lowest correlation coefficients
# max_corr = np.max(corr.max())
# min_corr = np.min(corr.min())

# # Get colormap + normalization from heatmap
# cmap = ax.collections[0].cmap
# norm = ax.collections[0].norm

# # Annotate the heatmap with p-values and change text color based on correlation value
# for i in range(p_values.shape[0]):
#     for j in range(p_values.shape[1]):
#         if mask_pvalues[i, j]:
#             p_value = p_values.iloc[i, j]
#             if not np.isnan(p_value):
#                 correlation_value = corr.iloc[i, j]
#                 # luminance based text colors
#                 rgba = cmap(norm(correlation_value))
#                 r, g, b, _ = rgba
#                 luminance = 0.299*r + 0.587*g + 0.114*b
#                 text_color = 'white' if luminance < 0.5 else 'black'

#                 if p_value <= 0.01:
#                 #include double asterisks for p-value <= 0.01
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})**',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)
#                 elif p_value <= 0.05:
#                 #include single asterisks for p-value <= 0.05
#                     ax.text(i + 0.5, j + 0.65, f'(p <= {p_value:.2f})*',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)
#                 else:
#                     ax.text(i + 0.5, j + 0.65, f'(p = {p_value:.2f})',
#                             horizontalalignment='center',
#                             verticalalignment='center',
#                             fontsize=12,
#                             color=text_color)

# # wrap labels
# x_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_xticklabels()]
# y_labels = [textwrap.fill(t.get_text(), 20) for t in ax.get_yticklabels()]

# ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)
# ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)

# # fig.tight_layout()
# fig.savefig(path_enc_ret, bbox_inches='tight')












###########################################################################################

## Scatterplot with regression line: correlation between tau_encoding and tau_retrieval
x = df_merged['tau_retrieval']
y = df_merged['tau_encoding']

rho, p = spearmanr(x, y)

print("Spearman correlation (rho):", rho)
print("p-value:", p)


# plot as scatterplot 
fig, ax = plt.subplots()
sns.regplot(data=df_merged, x='tau_encoding', y='tau_retrieval', color='midnightblue')
# change scale of axis to display ms
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*1000:.0f}'))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*1000:.0f}'))
ax.set_xlabel(r'$\tau_{enc}$ (ms)', fontsize=14)
ax.set_ylabel(r'$\tau_{ret}$ (ms)', fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# add rho and v-value string
ax.text(0.03, 0.95, rf"$\rho = {rho:.2f}$" + "\n" + rf"$p < 0.001$", ha="left", va="top", transform=ax.transAxes)
# ax.text(-0.125, 1.1, 'b', ha='center', va='center', transform=ax.transAxes, fontsize=14, fontweight='bold')
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'regplot_tauenc_tauret_{date}.pdf'), bbox_inches="tight")



