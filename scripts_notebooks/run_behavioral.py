"""This script processes the behavioral data from the encoding and retrieval phase."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.behavioral import read_behav_data, process_enc_data, process_ret_data, calculate_encoding_accuracy, create_behavioral_summary, calculate_retrieval_accuracy, calculate_hitrate, calculate_fa_rate

# set system variables
param1 = sys.argv[1]

# # print behavioral data from retrieval phase
data_retrieval = read_behav_data(sub = param1, phase = 'ret')
print("BEHAVIORAL DATA FROM RETRIEVAL:", data_retrieval.head(5))

# data_enc = process_enc_data(sub = param1, data=data_encoding)
data_ret = process_ret_data(sub = param1, data=read_behav_data(phase='ret', sub=param1))
data_enc = process_enc_data(sub = param1, data=read_behav_data(phase='enc', sub=param1))


# calculate updated encoding accuracy
# hitrate_low, hitrate_high, hitrate_dist = calculate_hitrate(data=data_ret, trial_wise=True)

# print(hitrate_low)
# print(hitrate_high)
# print(hitrate_dist)


farate = calculate_fa_rate(data_ret)

print("THIS IS THE FA RATE", farate)


# accuracy_retrieval = calculate_retrieval_accuracy(data_ret)
# print('ACCURACY RETRIEVAL', accuracy_retrieval)

# hitrate = calculate_hitrate(data_ret)

# ## check whether dropping NAN values has worked
# assert data_enc['Enc_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"
# assert data_ret['Ret_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"

print('\nCREATING BEHAVIORAL SUMMARY\n')
create_behavioral_summary(sub=param1, data_enc=data_enc, data_ret=data_ret)
