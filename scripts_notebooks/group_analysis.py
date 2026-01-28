"""This script performs timescale analysis."""

# make imports 
import os
import glob
import sys
import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# import paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest

# set system variables
# subjects = [101, 103, 104, 105]

import os
import pandas as pd

subjects = list(range(101, 152))

## TIMESCALES
# load model parameters for each subject 
dfs = []

for sub in subjects:
    print(f"LOADING ACF MODEL PARAMS FOR SUB-{sub}\n")
    
    # file path
    file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc.csv')
    
    # check if file exists before loading
    if not os.path.exists(file_path):
        print(f"File not found for subject {sub}, skipping...")
        continue
    
    # read CSV
    df = pd.read_csv(file_path)
    dfs.append(df)


# concat vs. merge vs. join!??
df_glob = pd.concat(dfs, axis = 0, join='outer')
print(df_glob.head())

print('Shape of merged data frame', df_glob.shape)

# save df_glob
df_glob.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales','group', 'group_level_acf_params_evoked_enc.csv'), sep = ',', header = True, index = False)

# # compute mean RSQ
# print(df_glob['rsq'].groupby(df_glob['sub']).describe())

# compute descriptive stats of tau (mean across all channels)
print('TAU SUMMARY\n')
# print(df_glob['tau_ms'].groupby(df_glob['sub']).describe())


df_grouped = df_glob['tau_ms'].groupby(df_glob['sub']).describe()
print(df_grouped)


# okay, so here we are plotting mean over all electrodes
# df_grouped['mean'].plot(
#     kind='bar',
#     color='darkblue',
#     title = 'mean tau across all chs'
# )

df_grouped2 = df_glob['avg_tau_parietal'].groupby(df_glob['sub']).describe()
print(df_grouped2)


# okay, so here we are plotting mean over frontal cluster
df_grouped2['mean'].plot(
    kind='bar',
    color='darkgreen',
    title = 'mean tau across parietal chs'
)


# # group by channels
# print('GROUP BY CHANNELS SUMMARY\n')
# print(df_glob['tau_ms'].groupby(df_glob['chs']).describe())


# print(df_glob['avg_tau_frontal'].groupby(df_glob['sub']))


# first_5_subs = df_glob[df_glob['sub'].isin([145, 146, 147, 148, 149, 150])]
# seaborn.boxplot(data=first_5_subs, x='sub', y='tau_ms')

# plt.title('Tau (ms) for 5 Subjects')
# plt.xlabel('Subject')
# plt.ylabel('Tau (ms)')

# plot distribution of raw tau values (density plot)
# df_glob['tau_ms'].plot.density(title='Tau (ms) Density')