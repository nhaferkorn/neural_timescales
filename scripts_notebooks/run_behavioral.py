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
from timescales_memory.behavioral import read_behav_data, process_enc_data, process_ret_data, calculate_encoding_task_performance, calculate_retrieval_task_performance, create_behavioral_summary, calculate_hitrate, calculate_fa_rate, calculate_hitrate_confidence, compute_dprime_targets_only


# set system variables
sub = sys.argv[1]

# behavioral data from encoding phase
data_encoding = read_behav_data(sub=sub, phase='enc')
print("BEHAVIORAL DATA FROM ENCODING:", data_encoding.head(5))
data_enc = process_enc_data(sub=sub, data=data_encoding)

# behavioral data from retrieval phase
data_retrieval = read_behav_data(sub=sub, phase='ret')
print("BEHAVIORAL DATA FROM RETRIEVAL:", data_retrieval.head(5))
data_ret = process_ret_data(sub=sub, data=data_retrieval)


calculate_hitrate_confidence(data_ret)

print('Task performance Retrieval Phase:', calculate_retrieval_task_performance(data_ret))

print('These are the hit rates:\n')
hitrate_low, hitrate_high, hitrate_targets, hitrate_dist = calculate_hitrate(data=data_ret, trial_wise=True)
print(hitrate_low)
print(hitrate_high)
print(hitrate_targets)
print(hitrate_dist)


hitrate = calculate_hitrate(data=data_ret)
print('\nThis is the global hitrate:', hitrate)

farate = calculate_fa_rate(data_ret)
print("This is the farate:", farate, '\n')


d_prime = compute_dprime_targets_only(data_ret)
print("This is d prime:", d_prime, '\n')


create_behavioral_summary(sub=sub, data_enc=data_enc, data_ret=data_ret)



