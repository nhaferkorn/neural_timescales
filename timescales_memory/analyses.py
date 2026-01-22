"""This script analyses the cleaned EEG data."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

# import functions from timescales package
from statsmodels.tsa.stattools import acf as compute_acf

from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

from timescales.sim import sim_ar
from timescales.fit import ARPSD, PSD, ACF
from timescales.autoreg import compute_ar_spectrum
from timescales.plts import set_default_rc


# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, EPOCHS_DIR, events_of_interest



## PLOT CLEANED RAW SIGNAL
def plot_reconst_raw(sub):
    reconst_fname = f'sub-{sub}-raw_cleaned.fif'
    RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)
    
    reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

    print('PLOTTING CLEANED RAW SIGNAL')
    reconst_raw.plot(block=True)


## RUN EPOCHS
def run_epochs(sub):

    # load cleaned data for subject
    print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
    reconst_fname = f'sub-{sub}-raw_cleaned.fif'
    RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

    reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

    # find events
    events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
    events[:,2] = events[:, 2] - 64512

    print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

    # input: cleaned data (post ICA + manual rejection)
    epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline = (-0.5, 0), reject_by_annotation=True, picks = 'eeg', on_missing="ignore")

    # save epoched data
    epochs_fname = reconst_fname.replace('-raw_cleaned.fif', '-epo.fif')  
    print("This is the file name: ", epochs_fname)

    epochs.save(os.path.join(DERIV_DIR, "epochs", epochs_fname), overwrite=True) 

    print("\nSUCCESSFULLY CREATED & SAVED EPOCHS")

    return epochs

def epochs_stats():
    """Compute # of dropped epochs, based on reject by annotation & bad channel criterions"""
    pass

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


## FIT IN TIME-DOMAIN (ACF)
def timescales_acf_evoked(evoked):

    """Use list apprehension to store the results of acf fit.

    Args:
        evoked (_type_): _description_

    Returns:
        list: of 32 numpy arrays (each containing the parameters of the acf fit (tau, hight & offset))
    """

    # get data (you can even specify tmin and tmax - which is super handy)
    evoked_data = evoked.get_data()

    # also check dimensions of evoked data
    print("VIEW ON EVOKED DATA", evoked_data.shape)

    # initialize empty list
    acf_list = []

    ## then fit a timescale object for each individual channel
    for ch in range(evoked_data.shape[0]):

        # initialize acf object
        acf = ACF()

        # initialize autocorrelation function
        acf.compute_acf(evoked_data[ch][:], fs = evoked.info["sfreq"])

        # fit autocorrelation function
        acf.fit()

        # print some output values
        print("PARAMETER NAMES", acf.param_names)
        print("FIRST FIVE PARAMETERS", acf.params[:5])

        acf_list.append(acf.params)

    print("ACF LIST", acf_list)
    return acf



# FIT IN TIME DOMAIN (NUMPY ARRAY ALLOCATION)
def timescales_acf_evoked_np(sub, evoked, osc: bool = False):
    """Use pre-allocation of numpy arrays to store the results of acf fit.

    Args:
        evoked (_type_): _description_

    Returns:
        ndarray: array of acf.fit parameters of shape (32, 3)
    """

    # get data (you can even specify tmin and tmax - which is super handy)
    evoked_data = evoked.get_data()

    # also check dimensions of evoked data
    print("VIEW ON EVOKED DATA", evoked_data.shape)

    ## hard coding stuff is never a good idea!
    n_params = 7 if osc else 3

    # pre-allocate numpy array of right size
    acf_array = np.zeros((32, n_params))
    rsq = np.zeros(32)

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

        # print some output values
        print("PARAMETER NAMES", acf.param_names)
        print("ACF FIT PARAMETERS", acf.params) 
        
        # print model r-squared values
        print("ACF EXPLAINED VARIANCE (RSQ)", acf.rsq)

        acf_array[ch] = acf.params
        rsq[ch] = acf.rsq

        # TODO: I think it also makes sense to save the parameter names

        # Plotting (think about moving this outside of the function!)
        fig, ax = plt.subplots()

        acf.plot(ax=ax, title=f'TIME DOMAIN FIT FOR CHANNEL:{ch}')

        fig.savefig(os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f"sub-{sub}-ch-{ch}.png"))

        plt.close(fig)

    print(acf_array)

    return acf_array, rsq



## COMPUTE PSD
def compute_psd(sub, epochs, events_of_interest):

    print(f"\nCOMPUTE PSD FOR EPOCHS OF INTEREST FOR SUB-{sub}\n")

    epochs = epochs[events_of_interest]

    # for sake of sanity check
    # print("# OF EPOCHS TO COMPUTE PSD ON", len(epochs))

    # compute spectrum object
    psd = epochs.average().compute_psd(fmin = 2, fmax = 30)

    # print('SHAPE OF PSD AFTER AVERAGING:', psd_fixation_average.get_data().shape) # >> returns: 32, 56

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

## FIXME!!
def timescales_psd_evoked_np(sub, epochs, freqs, power):

        # get channel names
        ch_names = epochs.info['ch_names']

        print("THESE ARE THE CHANNEL NAMES: ", ch_names)
        print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
        

        # TODO: Initialize numpy arrays that are of the right size to store the parameter estimates
        model_array = np.zeros(len(ch_names))
        # timescales_psd_array = np.zeros()
        
        for ch_idx, ch_name in enumerate(ch_names):

            # Compute PSD fit
            psd_fixation = PSD()

            psd_fixation = PSD(freqs, power[ch_idx])

            psd_fixation.fit(method='huber')
            
            model_array[ch_idx] = psd_fixation.tau

            # save the parameters
            # psd_timescales = psd_fixation.tau
    
        print(model_array)



def timescales_enc_ret(sub, epochs):
    """Compute timescales separately for encoding vs. retrieval phase (using PSD fit)."""

    # select epochs from enc and retrieval phase
    # in this implementation the epochs need to be already loaded (and epochs object needs to be passed as argument to the function)
    epochs_enc = epochs['Fixation Onset Enc']
    epochs_ret = epochs['Fixation Onset Ret']

    # Sanity check
    print("# OF ENCODING EPOCHS:", len(epochs_enc))
    
    print("# OF RETENTION EPOCHS:", len(epochs_ret))
    
    # Compute power & frequencies for encoding epochs
    print("\nPROCESSING ENCODING EPOCHS")

    # honestly, not sure if this is going to work
    power_enc, freqs_enc = compute_psd(sub=sub, epochs = epochs, events_of_interest = ['Fixation Onset Enc'])

    # but this actually seems to work
    ch_names = epochs.info['ch_names'][:power_enc.shape[0]]  # TODO: double check 

    print("THESE ARE THE CHANNEL NAMES: ", ch_names)
    print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
    
    psd_enc_list = []
    tau_enc_list = []
    r2_enc_list = []

    # use pre-allocation of numpy array
    for ch_idx, ch_name in enumerate(ch_names):

        #Compute PSD fit
        psd_enc = PSD()

        psd_enc = PSD(freqs_enc, power_enc[ch_idx])

        psd_enc.fit(method='fooof', fooof_init={'peak_threshold': 2.5})

        psd_enc_list.append(psd_enc)

        tau_enc = psd_enc.tau

        # compute fit
        r2_enc = psd_enc.rsq

        tau_enc_list.append(tau_enc)
        r2_enc_list.append(r2_enc)

        
    ## save timescale parameters to list 
    print(tau_enc_list)
    print(len(tau_enc_list)) # >> returns list with 32 entries of type np.float , so 1 timescale per channel (because its averaged over the epochs)

    print()

    print('R2 ENCODING', r2_enc_list)

    # convert list of numpy floats into np.array
    timescales_enc_array = np.array(tau_enc_list)

    # 
    np.savetxt(
    os.path.join(DERIV_DIR, 'timescales', 'arrays', f'Sub-{sub}_timescales_fooof_encoding.txt'),
    timescales_enc_array)


    ## RETRIEVAL PHASE
    print("\nPROCESSING RETRIEVAL EPOCHS")

    # honestly, not sure if this is going to work
    power_ret, freqs_ret = compute_psd(sub=sub, epochs = epochs, events_of_interest = ['Fixation Onset Ret'])

    # but this actually seems to work
    ch_names = epochs.info['ch_names'][:power_ret.shape[0]]  # double check this!!

    print("THESE ARE THE CHANNEL NAMES: ", ch_names)
    print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
    
    psd_ret_list = []
    tau_ret_list = []
    r2_ret_list = []

    # use pre-allocation of numpy array
    for ch_idx, ch_name in enumerate(ch_names):

        #Compute PSD fit
        psd_ret = PSD()

        psd_ret = PSD(freqs_ret, power_ret[ch_idx])

        psd_ret.fit(method='fooof', fooof_init={'peak_threshold': 2.5})

        psd_ret_list.append(psd_ret)

        # tau parameters
        tau_ret = psd_ret.tau

        # compute fit 
        r2_ret = psd_ret.rsq

        tau_ret_list.append(tau_ret)
        r2_ret_list.append(r2_ret)

        
    ## save timescale parameters to list - unit: seconds
    print(tau_ret_list)
    print(len(tau_ret_list)) # >> returns list with 32 entries of type np.float , so 1 timescale per channel (because its averaged over the epochs)

    print()

    print('R2 ENCODING', r2_enc_list)
    # convert list of numpy floats into np.array
    timescales_ret_array = np.array(tau_ret_list)

    # save as txt file
    np.savetxt(
    os.path.join(DERIV_DIR, 'timescales', 'arrays', f'Sub-{sub}_timescales_fooof_retention.txt'),
    timescales_ret_array)

    return power_enc, freqs_enc, power_ret, freqs_ret



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



## TIMESCALE ESTIMATION IN TIME DOMAIN - USING ACF OBJECTS
def timescales_acf(sub, epochs, events_of_interest):
    """Compute Timescales from the time-domain representation of epochs.
     can be fit with or without an oscillatory component - this is something I need to check again."""
    
    print("NOW ESTIMATING TIMESCALES USING ACF")
    
    epoch_avg = epochs[events_of_interest].average()  


    print('TYPE OF epoch_avg', type(epoch_avg))  # probably evoked object

    # which means I have to call get_data method
    epoch_avg_ts = epoch_avg.get_data()

    print('TYPE OF epoch_avg_ts', type(epoch_avg_ts))
    print()

    print('SHAPE OF epoch_avg_ts', epoch_avg_ts.shape) # TODO: in theory acf fitting should also work in 2D

    ch_names = epochs.info['ch_names']
    print("THESE ARE THE CHANNEL NAMES: ", ch_names)
    print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
    
    acf_enc_list = []
    acf_tau_enc_list = []

    # TODO: Numpy Array 
    for ch_idx, ch_name in enumerate(ch_names):

        # Initialize acf object
        acf_enc = ACF()

        # Compute autocorrelation
        acf_enc.compute_acf(sig = epoch_avg_ts[ch_idx], fs = epochs.info['sfreq'])

        # Fit ACF function
        acf_enc.fit()

        # evaluate acf fit
        print(acf_enc.gen_corrs_fit())  # FIXME: Return None

        # FIXME: check parameter names extract tau parameters & append them to list
        # tau_acf_enc = acf_enc.tau

        # acf_tau_enc_list.append(tau_acf_enc)



## PLOTTING FUNCTIONS
def plot_aperiodic_power(sub, freqs, power):

    epochs_fname = f'sub-{sub}-epo.fif'
    epochs = mne.read_epochs(os.path.join(DERIV_DIR, "epochs", epochs_fname), preload=True)

    ch_names = epochs.info['ch_names'][:power.shape[0]]

    psd_list = []

    for ch_idx, ch_name in enumerate(ch_names):

        #Compute PSD fit
        psd_fix = PSD()

        psd_fix = PSD(freqs, power[ch_idx])

        psd_fix.fit(method='huber')

        psd_list.append(psd_fix)
    
        # plot power spectrum - aperiodic fit for each channel
        # TODO: what does aperiodic fit mean? So power spectrum only of aperiodic components??
        fig, ax = plt.subplots()     
        psd_fix.plot(ax=ax)      
        fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"sub-{sub}_psd_fooof_ret_ch-{ch_name}.png"))
        plt.close(fig)


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



# def visualize_tau(timescales_tau_array):
#      mne.viz.plot_topomap(data = timescales_tau_array, pos = epochs_eeg.info, cmap = 'viridis')




if __name__ == "__main__":
     pass
    

