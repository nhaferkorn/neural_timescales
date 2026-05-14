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

epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = 0.2, tmax=1.1, baseline = (None, None), reject_by_annotation=True, preload=True, picks = 'eeg', on_missing="ignore")

# interpolate bad channels
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False) 
print(epochs_interpolated.info)


epochs_fix_enc = epochs_interpolated['Fixation Onset Enc']
epochs_fix_ret = epochs_interpolated['Fixation Onset Ret']

# for aABC tau we need the data in the format of a numpy array (numTrials x time-points)
epochs_enc_data = epochs_fix_enc.get_data()
print('SHAPE', epochs_enc_data.shape) # shape = (n_trials, n_electrodes, n_times)


# iterate over electrodes (and channel names)
ch_names = epochs_fix_enc.info['ch_names']


# # just pick one electrode
# data_electrode = epochs_enc_data[:, 2, :]
# print('Data Electrode Shape', data_electrode.shape) # shape = (n_trials, n_times)

# # maybe start abcTau fittings on channel-averaged data 
# enc_chsavg_data = epochs_enc_data.mean(axis=1)
# print('Shape channel-averaged data', enc_chsavg_data.shape)


# extract statistics from the real data
# select summary statistic metric
summStat_metric = 'comp_ac_fft'
ifNorm = False # if true, it normalizes the autocorrelation or PSD
# deltaT = 1/epochs_fix_enc.info['sfreq']
deltaT = 1
disp = None # put the dispersion parameter if computed with grid-search
maxTimeLag = 500 # only used when using autocorrelation for summary statistics
binSize=1
#lm = round(maxTimeLag/binSize1) # maximum bin for autocorrelation computation


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
n_procs = 16 # number of processor for parallel processing (set to 1 if there is no parallel processing)


# creating model object
class MyModel(abcTau.Model):

    #This method initializes the model object.  
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



# loop over channels, such that essentially we get one model fit per channel
for ch, name in zip(range(epochs_enc_data.shape[1])[:2], ch_names[:2]):

    # first test whether statistical bias is present
    
    data_electrode = epochs_enc_data[:, ch, :]

    data_ac, data_mean, data_var, T, numTrials = abcTau.preprocessing.extract_stats(data_electrode, deltaT, binSize,\
                                                                                  summStat_metric, ifNorm, maxTimeLag)

    # fit 
    popt, poptcov = abcTau.preprocessing.fit_oneTauExponential(data_ac, binSize, maxTimeLag)
    tau = popt[1]

    # check if estimated timescales with exponential fit are biased or not
    theta = np.array([tau])
    numTimescales = 1
    taus_bs, taus_bs_corr, err = abcTau.preprocessing.check_expEstimates(theta, deltaT, binSize, T, numTrials,\
                                                                        data_mean, data_var, maxTimeLag, numTimescales,\
                                                                        numIter = 100, plot_it = False)

    print(f'now fitting  abcTau for channel {name}')

    data_sumStat, data_mean, data_var, T, numTrials =  abcTau.preprocessing.extract_stats(data_electrode , deltaT, binSize,\
                                                                        summStat_metric, ifNorm, maxTimeLag)
    print('MEAN', data_mean) 
    print('VARIANCE', data_var)
    print('T', T)

    # filename to save the intermediate results after running each step
    inter_save_direc = os.path.join(DERIV_DIR, 'timescales', 'abc_timescales/')
    inter_filename = f'abc_intermediate_results_encoding_{name}'

    # define filename for saving the results
    filename = f'abc_encoding_{name}'
    filenameSave = filename


    # fit with aABC algorithm for any generative model
    abc_results, final_step = abcTau.fit.fit_withABC(MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,\
                                        datasave_path,filenameSave, epsilon_0, min_samples, \
                                    steps, minAccRate, parallel, n_procs, disp)