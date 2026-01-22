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
from timescales_memory.behavioral import read_behav_data, process_enc_data, process_ret_data, create_behavioral_summary

# set system variables
param1 = sys.argv[1]

# print some data for the encoding phase
data_encoding = read_behav_data(sub = param1, phase = 'enc')
print(type(data_encoding))  # returns Dataframe object
print("BEHAVIORAL DATA OF ENCODING PHASE:", data_encoding.head(5))

# # print behavioral data from retrieval phase
data_retrieval = read_behav_data(sub = param1, phase = 'ret')
print("BEHAVIORAL DATA FROM RETRIEVAL:", data_retrieval.head(5))

data_enc = process_enc_data(sub = param1, data=data_encoding)
data_ret = process_ret_data(sub = param1, data=read_behav_data(phase='ret', sub=param1))

## check whether dropping NAN values has worked
assert data_enc['Enc_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"
assert data_ret['Ret_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"

print('\nCREATING BEHAVIORAL SUMMARY\n')
create_behavioral_summary(sub=param1, data_enc=data_enc, data_ret=data_ret)
