"""This contains functions for timescale analysis of EEG data."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

# import functions from statsmodels, neurodsp & timescales
from statsmodels.tsa.stattools import acf as compute_acf

from neurodsp.spectral import compute_spectrum, trim_spectrum

from timescales.sim import sim_ar
from timescales.fit import ARPSD, PSD, ACF
from timescales.autoreg import compute_ar_spectrum


# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, EPOCHS_DIR, events_of_interest


## PLOT CLEANED RAW SIGNAL
def plot_reconst_raw(sub):
    """Plots the cleaned raw signal (on HPC)."""

    f_name = f'sub-{sub}-raw_cleaned.fif'
    file_path = os.path.join(RAW_CLEANED, f_name)
    
    reconst_raw = mne.io.read_raw_fif(file_path, preload=True)

    print('PLOTTING CLEANED RAW SIGNAL')
    reconst_raw.plot(block=True)


def epochs_stats():
    """Compute # of dropped epochs, based on reject by annotation."""
    pass

## SPECTRAL PARAMETERIZATION 
# TODO: check if I actually still need this function
def run_fooof(sub, reconst_raw):

    print("RUNNING FOOOF")

    psd = reconst_raw.compute_psd(method = "welch", fmin = 1, fmax=30, tmin=0, tmax=250, n_overlap=150, n_fft=300)

    spectra, freqs = psd.get_data(return_freqs=True)

    ## Fitting Power Spectrum Models

    # Initialize a FOOOFGroup object, with desired settings
    fg = fooof.FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
            peak_threshold=2., max_n_peaks=6, verbose=False)

    # Define the frequency range 
    freq_range = [1, 30]

    # Fit the power spectrum model across all channels
    fg.fit(freqs, spectra, freq_range)

    # Check the overall results of the group fits; change output directory 
    fg.plot(save_fig = True, file_name = f"fooof_sub-{sub}", file_path = os.path.join(DERIV_DIR,'timescales')) 


# FIT IN TIME DOMAIN
def timescales_acf_evoked(sub, evoked, osc: bool = False):
    """Use pre-allocation of numpy arrays to store the results of acf fit.

    Args:
        evoked (Evoked object): averaged fixation epochs (n_times, n_channels)

    Returns:
        ndarray: array of acf.fit parameters of shape (32, 3)
    """

    # get data 
    evoked_data = evoked.get_data()

    # also check dimensions of evoked data
    # TODO: turn into assert statement
    print("VIEW ON EVOKED DATA", evoked_data.shape)

    n_params = 7 if osc else 3
    n_chs = len(evoked.info["ch_names"]) 

    # pre-allocate numpy array of right size
    acf_array = np.zeros((n_chs, n_params))
    rsq = np.zeros(n_chs)

    ## then fit a timescale object for each individual channel
    for ch in range(acf_array.shape[0]):

        # initialize acf object
        acf = ACF()

        # compute autocorrelation
        acf.compute_acf(evoked_data[ch][:], fs = evoked.info["sfreq"])

        # fit function
        if osc:
            acf.fit(with_cos=True)
        else:
            acf.fit()

        # # print some output values
        # print("PARAMETER NAMES", acf.param_names)
        # print("ACF FIT PARAMETERS", acf.params) 
        
        # # print model r-squared values
        # print("ACF EXPLAINED VARIANCE (RSQ)", acf.rsq)

        acf_array[ch] = acf.params
        rsq[ch] = acf.rsq

    return acf_array, rsq



def timescales_acf_single_trials(sub, epochs, osc: bool = False):
    """Use pre-allocation of numpy arrays to store the results of acf fit.

    Args:
        Epochs object: 

    Returns:
        ndarray: array of acf.fit parameters of shape 
    """

    # get data 
    data = epochs.get_data()

    # also check dimensions of evoked data
    print("VIEW ON EPOCHS DATA", data.shape)

    n_params = 7 if osc else 3

    # specify channels
    n_chs = len(epochs.info["ch_names"]) 

    # specify number of trials
    n_epochs = len(epochs)

    # pre-allocate numpy array of right size (n_epochs, n_chs, n_params)
    acf_array = np.zeros((n_epochs, n_chs, n_params))
    rsq = np.zeros((n_epochs, n_chs, 1))

    ## then fit a timescale object for each individual channel
    for epoch in range(acf_array.shape[0]):
        for ch in range(acf_array.shape[1]):

            # initialize acf object
            acf = ACF()

            # compute autocorrelation
            acf.compute_acf(data[epoch][ch][:], fs = epochs.info["sfreq"])

            # fit function
            if osc:
                acf.fit(with_cos=True)
            else:
                acf.fit()
    
            acf_array[epoch, ch] = acf.params
            rsq[epoch, ch] = acf.rsq

    return acf_array, rsq



## COMPUTE PSD
# TODO: not sure for which purposes I still need this function
def compute_psd(sub, epochs, events_of_interest):

    print(f"\nCOMPUTE PSD FOR EPOCHS OF INTEREST FOR SUB-{sub}\n")

    epochs = epochs[events_of_interest]

    # compute spectrum object
    psd = epochs.average().compute_psd(fmin = 2, fmax = 30)

    power, freqs = psd.get_data(return_freqs = True)

    return power, freqs


## PSD timescale fits
def timescales_psd_evoked(sub, freqs, power):
        # load the epoched data
        epochs_fname = f'sub-{sub}-epo.fif'
        epochs = mne.read_epochs(os.path.join(DERIV_DIR, "epochs", epochs_fname), preload=True)
    
        epochs = epochs.copy().pick(picks=["eeg"])

        # FIXME!!
        epochs = epochs[['Fixation Onset Enc', 'Fixation Onset Ret']]

        # but this actually seems to work
        ch_names = epochs.info['ch_names'][:power.shape[0]]  # double check this!!

        print("THESE ARE THE CHANNEL NAMES: ", ch_names)
        print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
        
        psd_list = []
        timescales_tau_list = []

        # use pre-allocation of numpy array
        for ch_idx, ch_name in enumerate(ch_names):

            #Compute PSD fit
            psd_fix = PSD()

            psd_fix = PSD(freqs, power[ch_idx])

            psd_fix.fit(method='huber')

            psd_list.append(psd_fix)

            # save the parameters, aka the timescale values
            psd_timescales = psd_fix.tau
            timescales_tau_list.append(psd_timescales)

        ## save timescale parameters to list 
        print(timescales_tau_list)
        print(len(timescales_tau_list)) # >> returns list with 32 entries of type np.float , so 1 timescale per channel (because its averaged over the epochs)

        # convert list of numpy floats into np.array
        timescales_tau_array = np.array(timescales_tau_list)

        # nice, so this should have the right shape now
        print(timescales_tau_array)

        # TODO: compute fit using rsqurd

        # it also allows me to set vlim: aka lower and upper bound of the colormap
        # this plot depicts average tau for the all of the fixation periods 
        mne.viz.plot_topomap(data = timescales_tau_array, pos = epochs.info, cmap = 'viridis')

# TODO: 10.03.26 - check if this function actually works
def timescales_psd_evoked_np(sub, evoked):
    """Using FOOOF model with robust regression. Estimates the following
     params: 'offset', 'knee_freq', 'exp', 'const' """

    # get channel names
    ch_names = evoked.info['ch_names']

    n_params = 4

    evoked_data = evoked.get_data()

    # pre-allocate numpy array of right size
    psd_array = np.zeros((len(ch_names), n_params))
    rsq_psd = np.zeros(len(ch_names))
    tau_psd = np.zeros(len(ch_names))

    print(len(ch_names))
            
    for ch in range(len(ch_names)):

        # initialize PSD object
        psd_fixation = PSD()
        
        # compute autocorrelation
        psd_fixation.compute_spectrum(evoked_data[ch][:], fs = evoked.info["sfreq"])

        psd_fixation.fit(method='huber')
        psd_array[ch] = psd_fixation.params
        rsq_psd[ch] = psd_fixation.rsq
        tau_psd[ch] = psd_fixation.tau

    return psd_array, rsq_psd, tau_psd 

# FIXME!
def timescales_single_trials(sub):

    """ Computes psd for each epoch x channel pair.

    Returns a list of timescale.fit.psd.PSD objects of length (n_epochs x n_channels)
    """

    print(f"\nFITTING SINGLE-TRIAL TIMESCALES ACROSS ALL CHANNELS FOR SUB-{sub}\n") 
    # load the epoched data
    epochs_fname = f'sub-{sub}-epo.fif'
    epochs = mne.read_epochs(os.path.join(DERIV_DIR, "epochs", epochs_fname), preload=True)

    # pick only eeg channels, yess: this works but also requires that data are preloaded into memory
    epochs_eeg = epochs.copy().pick(picks=["eeg"])

    print(epochs_eeg.info)

    # select fixation period epochs
    epochs_fixation = epochs_eeg[['Fixation Onset Enc', 'Fixation Onset Ret']] 

    # compute psd per epoch; spectral representation of each epoch stored in EpochsSpectrum object
    psd_fixation = epochs_fixation.compute_psd(fmin = 2, fmax = 30)

    # now call get data method
    print(psd_fixation.get_data().shape)  # >> (430, 32, 56)

    # compute power and frequencies
    power, freqs = psd_fixation.get_data(return_freqs = True)

    print(power.shape) # >> (430, 32, 56)
    print(freqs.shape) # >> (56,)

    # initialize empty list object
    psd_list = []


    for ch_idx in range(power.shape[1]):
            
            print(ch_idx) 

            for epoch_idx in range(power.shape[0]):

                psd_fix = PSD()

                psd_fix = PSD(freqs, power[epoch_idx, ch_idx]) # index both the epoch and the channel

                # append them to list
                psd_list.append(psd_fix)

                # fit 
                psd_fix.fit(method='huber')


    # EXPECTED: 430x32 timescales PSD objects
    print('Length of PSD list', len(psd_list))  # >> returns correct dimensions: 13760 list entries


## PLOTTING FUNCTIONS
def plot_acf(sub, power, freqs):

    epochs_fname = f'sub-{sub}-epo.fif'
    epochs = mne.read_epochs(os.path.join(DERIV_DIR, "epochs", epochs_fname), preload=True)

    ch_names = epochs.info['ch_names'][:power.shape[0]]

    for ch_idx, ch_name in enumerate(ch_names):

        acf_fix = ACF()
        
        acf_fix.compute_acf(power[ch_idx], epochs.info['sfreq'])

        # plot acf
        fig, ax = plt.subplots()

        # exponential decay model
        acf_fix.fit()
        acf_fix.plot(ax=ax, title='Exponential Decay Model')

        fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"sub-{sub}_acf_fix_ch-{ch_name}.png"))
        plt.close(fig)


if __name__ == "__main__":
     pass
    

