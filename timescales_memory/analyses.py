"""This contains functions for timescale analysis of EEG data."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 


# import functions from statsmodels & timescales-methods
from statsmodels.tsa.stattools import acf as compute_acf
from timescales.fit import ARPSD, PSD, ACF


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

## SPECTRAL PARAMETERIZATION 
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

        acf_array[ch] = acf.params
        rsq[ch] = acf.rsq

    return acf_array, rsq

def timescales_acf_single_trials(sub, epochs, osc: bool = False):
    """Use pre-allocation of numpy arrays to store the results of acf fit.
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

# FIT IN FREQUENCY DOMAIN
def timescales_psd_evoked(sub, evoked):
    """Using FOOOF model with robust regression. Estimates the following
     parameters: 'offset', 'knee_freq', 'exp', 'const'. """

    # get channel names
    ch_names = evoked.info['ch_names']

    n_params = 4

    evoked_data = evoked.get_data()

    # pre-allocate numpy array of right size
    psd_array = np.zeros((len(ch_names), n_params))
    rsq_psd = np.zeros(len(ch_names))
    tau_psd = np.zeros(len(ch_names))
            
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

# TODO: single trial fit in frequency domain

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
    

