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

# statistical analysis for single subject
# sub = sys.argv[1]

# initialize empty lists
tau_high = []
tau_low = []


subjects = ['%d' % i for i in range(101, 152)]

for sub in subjects:

    # specify paths
    info_path =  os.path.join(DERIV_DIR,'info', f'sub-{sub}_info.fif')
    file_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'distraction', f'sub-{sub}_acf_params_singletrial_combined.csv')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        df_high = df[['sub', 'epoch', 'chs', 'tau_high', 'tau_high_trialavg']]
        df_low = df[['sub', 'epoch', 'chs', 'tau_low', 'tau_low_trialavg']]

        # average all tau_high values (across electrodes and across epochs)
        tau_high.append(df_high['tau_high'].mean())
        tau_low.append(df_low['tau_low'].mean())

print(tau_low)
print(len(tau_low))


# convert them into numpy arrays
np_high = np.array(tau_high)
np_low = np.array(tau_low)


# print the arrays
print(np_high)
print(np_low)


# run Wilcoxon signed rank test (on trial & electrode averaged tau values ) - contrasting high and low distraction conditions
result = sp.stats.wilcoxon(np_high, np_low, alternative = 'two-sided') 
print(result)




##############################################################################################################################

                                    # Encoding Timescales vs. Retrieval Timescales

##############################################################################################################################
# initialize empty lists
tau_enc = []
tau_ret = []


subjects = ['%d' % i for i in range(101, 152)]

for sub in subjects:

    # specify paths
    info_path =  os.path.join(DERIV_DIR,'info', f'sub-{sub}_info.fif')

    enc_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'encoding', f'sub-{sub}_acf_params_single_trials_enc.csv')
    ret_path = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', 'single_trials', 'retrieval', f'sub-{sub}_acf_params_single_trials_ret.csv')

    # read info object
    if os.path.exists(info_path):
        info = mne.io.read_info(info_path)

    if os.path.exists(enc_path):

        df_enc = pd.read_csv(enc_path)
        df_ret = pd.read_csv(ret_path)

        df_enc = df_enc[['sub', 'epoch', 'channel', 'tau_enc']]
        df_ret= df_ret[['sub', 'epoch', 'channel', 'tau_ret']]

        # average all tau_high values (across electrodes and across epochs)
        tau_enc.append(df_enc['tau_enc'].mean())
        tau_ret.append(df_ret['tau_ret'].mean())

print(tau_enc)
print(len(tau_enc))
print(len(tau_ret))


# convert into numpy arrays
np_enc = np.array(tau_enc)
np_ret = np.array(tau_ret)


# run Wilcoxon signed rank test (on trial & electrode averaged tau values ) - contrasting encoding & retrieval phase
result_2 = sp.stats.wilcoxon(np_enc, np_ret, alternative = 'two-sided') 
print(result_2)