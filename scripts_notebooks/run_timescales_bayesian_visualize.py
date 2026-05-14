"""This script performs basic timescale analysis."""

# make imports 
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest

# import aABC functions
sys.path.append(os.path.join(PROJECT_DIR, 'abcTau', 'abcTau'))
import abcTau
from scipy import stats

# set system variables
sub = sys.argv[1]


# # conditions 
conditions = ['low', 'high']

# channel names (best to load from info object)
info_path = os.path.join(DERIV_DIR, 'info', f'sub-{sub}_info.fif')
info = mne.io.read_info(info_path)
ch_names = info['ch_names']

### COMPUTE POINT ESTIMATES - MAP
# load abc results
data_abc_path = os.path.join(DERIV_DIR, 'timescales', 'abc_timescales/')

for cond in conditions:
    for ch in ch_names:
        pattern = os.path.join(data_abc_path, f'sub-{sub}_abc_{cond}_{ch}_*.npy')
        files = glob.glob(pattern)

        for file in files:
            abc_results = np.load(file,  allow_pickle=True)
            fname = os.path.basename(file)
            print('FILE NAME', fname)
            
            # Find the part that contains "steps"
            step_part = [p for p in fname.split('_') if 'steps' in p][0]  

            # Remove "steps" and ".npy", convert to int
            final_step= int(step_part.replace('steps', '').replace('.npy', ''))

            print(final_step)

            # extract estimated parameters
            theta_accepted = abc_results[final_step-1]['theta accepted']

            # find MAPs
            N = 20000 # number of samples for grid search
            theta_MAP = abcTau.find_MAP(theta_accepted, N)

            # print MAP estimates 
            print(theta_MAP)