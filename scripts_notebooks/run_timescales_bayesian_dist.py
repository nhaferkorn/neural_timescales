"""This script performs basic timescale analysis."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum
from scipy import stats

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest

# import aABC functions
sys.path.append(os.path.join(PROJECT_DIR, 'abcTau', 'abcTau'))
import abcTau
from scipy import stats


# set system variables
sub = sys.argv[1]

# path for loading and saving data
datasave_path = os.path.join(DERIV_DIR, 'timescales', 'abc_timescales/')

# load cleaned data for subject
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
events[:,2] = events[:, 2] - 64512


# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# define keys and events 
keys = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target', 
        'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


event_dist = {k: EVENT_DICT[k] for k in keys}


# construct & demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = event_dist, tmin = 1.2, tmax=2.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# interpolate bad channels
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False)


# subselect epochs
epochs_high = epochs_interpolated['Encoding Stimulus Onset Distraction Right Target', 'Encoding Stimulus Onset Distraction Left Target']
epochs_low = epochs_interpolated['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


# get data
# for aABC tau we need the data in the format of a numpy array (numTrials x time-points)
epochs_high_data = epochs_high.get_data()
epochs_low_data = epochs_low.get_data()


epochs_list = [epochs_high_data, epochs_low_data]
epochs_names = ['high', 'low']

print(len(epochs_list))

# specify channel names
ch_names = epochs_high.info['ch_names']

# extract statistics from the real data
# select summary statistic metric
summStat_metric = 'comp_ac_fft'
ifNorm = True # if true, it normalizes the autocorrelation or PSD
deltaT = 1
disp = None # put the dispersion parameter if computed with grid-search
maxTimeLag = 500 # only used when using autocorrelation for summary statistics
binSize=1


# Define the prior distribution
# for a uniform prior: stats.uniform(loc=x_min,scale=x_max-x_min)
t_min = 0.0 # first timescale
t_max = 100.0
priorDist = [stats.uniform(loc= t_min, scale = t_max - t_min)]


# select generative model and distance function
generativeModel = 'oneTauOU'
distFunc = 'logarithmic_distance'


# set fitting params
epsilon_0 = 1  # initial error threshold
min_samples = 100 # min samples from the posterior
steps = 60 # max number of iterations
minAccRate = 0.01 # minimum acceptance rate to stop the iterations
parallel = True # if parallel processing
n_procs = 28 # number of processor for parallel processing (set to 1 if there is no parallel processing)


# creating model object
class MyModel(abcTau.Model):

    # This method initializes the model object.  
    def __init__(self):
        pass

    # draw samples from the prior. 
    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    # Choose the generative model (from generative_models)
    # Choose autocorrelation computation method (from basic_functions)
    def generate_data(self, theta):
        # generate synthetic data
        if disp == None:
            syn_data, numBinData =  eval('abcTau.generative_models.' + generativeModel + \
                                         '(theta, deltaT, binSize, T, numTrials, data_mean, data_var)')
        else:
            syn_data, numBinData =  eval('abcTau.generative_models.' + generativeModel + \
                                         '(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp)')
               
        # compute the summary statistics
        syn_sumStat = abcTau.summary_stats.comp_sumStat(syn_data, summStat_metric, ifNorm, deltaT, binSize, T,\
                                          numBinData, maxTimeLag)   
        return syn_sumStat

    # Computes the summary statistics
    def summary_stats(self, data):
        sum_stat = data
        return sum_stat

    # Choose the method for computing distance (from basic_functions)
    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0: # in case of all nans return large d to reject the sample
            d = 10**4
        else:
            d = eval('abcTau.distance_functions.' +distFunc + '(data, synth_data)')        
        return d



# # loop over channels, such that essentially we get one model fit per channel
for epoch_data, epoch_name in zip(epochs_list, epochs_names):

    for ch, name in zip(range(epoch_data.shape[1])[:1], ch_names[:1]):

        # first test whether statistical bias is present
        
        data_electrode = epoch_data[:, ch, :]

        # data_ac, data_mean, data_var, T, numTrials = abcTau.preprocessing.extract_stats(data_electrode, deltaT, binSize,\
        #                                                                             summStat_metric, ifNorm, maxTimeLag)

        # # fit 
        # popt, poptcov = abcTau.preprocessing.fit_oneTauExponential(data_ac, binSize, maxTimeLag)
        # tau = popt[1]

        # # check if estimated timescales with exponential fit are biased or not
        # theta = np.array([tau])
        # numTimescales = 1
        # taus_bs, taus_bs_corr, err = abcTau.preprocessing.check_expEstimates(theta, deltaT, binSize, T, numTrials,\
        #                                                                     data_mean, data_var, maxTimeLag, numTimescales,\
        #                                                                     numIter = 100, plot_it = False)

        print(f'now fitting  abcTau for channel {name}')

        data_sumStat, data_mean, data_var, T, numTrials =  abcTau.preprocessing.extract_stats(data_electrode , deltaT, binSize,\
                                                                            summStat_metric, ifNorm, maxTimeLag)
        print('MEAN', data_mean) 
        print('VARIANCE', data_var)
        print('T', T)

        # filename to save the intermediate results after running each step
        inter_save_direc = os.path.join(DERIV_DIR, 'timescales', 'abc_timescales/')
        inter_filename = f'sub-{sub}_abc_intermediate_results_{epoch_name}_{name}'

        # define filename for saving the results
        filename = f'sub-{sub}_abc_{epoch_name}_{name}'
        filenameSave = filename

        print("EPOCH:", epoch_name)
        print("FILENAME:", filenameSave)

        # fit with aABC algorithm for any generative model
        abc_results, final_step = abcTau.fit.fit_withABC(MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,\
                                            datasave_path,filenameSave, epsilon_0, min_samples, \
                                        steps, minAccRate, parallel, n_procs, disp)