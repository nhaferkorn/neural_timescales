"""This script analyses the cleaned EEG data"""

import os
import sys
import numpy as np
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

sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, events_of_interest

########################################################
## SETUP

# set system variables
param1 = sys.argv[1]

RUN_EPOCHS = True
RUN_EVOKED = False
RUN_FOOOF = False
RUN_TIMESCALES = True


def analyses(sub):

    if RUN_EPOCHS: 

        # load cleaned data for subject
        print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
        reconst_fname = f'sub-{sub}-raw_cleaned.fif'
        RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)
        
        reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=False)

        # find events (I could also save them)
        events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
        events[:,2] = events[:, 2] - 64512

        print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

        # input: cleaned data (post ICA + manual rejection)
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject_by_annotation=True, picks = 'eeg')

        # save epoched data
        epochs_fname = reconst_fname.replace('-raw_cleaned.fif', '-epo.fif')  
        print("This is the file name: ", epochs_fname)
        epochs.save(os.path.join(DERIV_DIR, "epochs", epochs_fname), overwrite=True) 

        print("\nSUCCESSFULLY CREATED & SAVED EPOCHS")

########################################################
## EVOKED RESPONSES 

    if RUN_EVOKED:

        ## OLD CODE - TO BE FIXED

        # compute evoked response
        # epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None, reject_by_annotation = True)
        
        # evoked responses for encoding phase
        evoked_left = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Distraction Left Target"]].average().plot(titles=f"{sub} Evoked - Left Attend", picks = 'eeg')
        evoked_right = epochs[["Encoding Stimulus Onset Baseline Right", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - Right Attend", picks = 'eeg')
        evoked_base = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Baseline Right"]].average().plot(titles=f"{sub} Evoked - Low Distraction Attend", picks = 'eeg')
        evoked_dist = epochs[["Encoding Stimulus Onset Distraction Left Target", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - High Distraction Attend", picks = 'eeg')
        evoked_cue = epochs["Fixation Onset Enc"].average().plot(titles=f"{sub} Evoked - Fixation Cue", picks = 'eeg')

        # save EVOKED plots
        evoked_left.savefig(os.path.join(DERIV_DIR, f'evoked_enc_left_{sub}.png'))
        evoked_right.savefig(os.path.join(DERIV_DIR, f'evoked_enc_right_{sub}.png'))
        evoked_base.savefig(os.path.join(DERIV_DIR, f'evoked_enc_baseline_{sub}.png'))
        evoked_dist.savefig(os.path.join(DERIV_DIR, f'evoked_enc_distraction_{sub}.png'))
        evoked_cue.savefig(os.path.join(DERIV_DIR, f'evoked_enc_fixcue_{sub}.png'))

########################################################
## SPECTRAL PARAMETERIZATION 
 
    if RUN_FOOOF: 
        # check how to run fooof on epoched data; I probably don't even need this code snippet

        print("CALCULATING POWER SPECTRA")

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


########################################################
## TIMESCALE ANALYSIS

    if RUN_TIMESCALES: 

        ## FIT ON RAW DATA
        print(f"\nFITTING TIMESCALES TUTORIAL\n")
        
        # Grab the sampling rate from the data
        fs = reconst_raw.info['sfreq']

        # Settings for exploring an example channel of data
        ch_label = 'P3'
        t_start = 20000
        t_stop = int(t_start + (10 * fs))

        # print ch_names
        print(reconst_raw.info['ch_names'])

        # Extract one channel for exploration
        sig, times = reconst_raw.get_data(mne.pick_channels(reconst_raw.ch_names, [ch_label]),
                                start=t_start, stop=t_stop, return_times=True)
        
        # squeeze signal to collapse extra dimension
        print("\nSIGNAL BEFORE SQEEZING:\n", sig.shape) # (1,  10240)

        sig = np.squeeze(sig) 
    
        # check dimensions after squeezing
        print("\nSIGNAL AFTER SQEEZING:\n", sig.shape) # (10240,)

        # Fit FOOOF robustly 
        psd_robust = PSD()
        psd_robust.compute_spectrum(sig, fs)
        psd_robust.fit(method='huber')

        # okay this is interesting - so apparently psd.plot method returns an axes object (and not a figure object)
        # here we are plotting the fitted powers pectrum but for one random channel (P3) and random time period
        fig, ax = plt.subplots() 
        psd_robust.plot(ax=ax)       
        fig.savefig(f"psd_fit_sub-{sub}.png")
        plt.close(fig)

   
######################################
## FIT TIMESCALES ON EPOCHED DATA (AVERAGE ACROSS FIXATION PERIOD EPOCHS)

def timescales_average_trials(sub):
    
        print(f"\nFITTING TIMESCALES AVERAGED ACROSS ALL FIXATION PERIODS FOR SUB-{sub}\n")

        # load the epoched data
        epochs_fname = f'sub-{sub}-epo.fif'
        epochs = mne.read_epochs(os.path.join(DERIV_DIR, "epochs", epochs_fname), preload=True)

        epochs_fixation = epochs['Fixation Onset Enc']

        # compute spectrum object: AVERAGE EPOCHS BEFORE COMPUTING PSD
        psd_fixation_average = epochs_fixation.average().compute_psd(fmin = 2, fmax = 30)

        print('SHAPE OF PSD AFTER AVERAGING:', psd_fixation_average.get_data().shape) # >> returns: 32, 56

        power, freqs = psd_fixation_average.get_data(return_freqs = True)

        print(f'Power', power)
        print('Shape of power', power.shape)

        # ideally I would store them in a 2D numpy array
        
        # not sure if this works
        # psd_array = np.array(power, freqs)


      
        psd_list = []

        # for ch_idx in range(power.shape[0]):

        #     ## Compute PSD fit
        #     psd_fix = PSD()

        #     psd_fix = PSD(freqs, power[ch_idx])

        #     psd_fix.fit(method='huber')

        #     psd_list.append(psd_fix)

        #     # plot power spectrum - aperiodic fit for each channel
        #     fig, ax = plt.subplots()     
        #     psd_fix.plot(ax=ax)      
        #     fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"psd_fixation_channel-{ch_idx}-sub-{sub}.png"))
        #     plt.close(fig)

        #     # compute acf fit 

        #     # initialize ACF object (1D array of power)
        #     acf_fix = ACF()
            
        #     acf_fix.compute_acf(power[ch_idx], epochs.info['sfreq'])

        #     # plot acf
        #     fig, ax = plt.subplots()

        #     # exponential decay model
        #     acf_fix.fit()
        #     acf_fix.plot(ax=ax, title='Exponential Decay Model')

        #     fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"acf_fit_fixation_channel-{ch_idx}-sub-{sub}.png"))
        #     plt.close(fig)

    
        # print(len(psd_list)) # >> (32) PSD objects; one for each channel


        ## try with enumerate to loop over both indices and channel names
        ## now this seems to work - but this is not the best solution 

        ch_names = epochs.info['ch_names'][:power.shape[0]]  

        ch_names_from_info = epochs.info['ch_names']
        print("\nTHESE ARE THE CHANNEL NAMES FROM THE INFO OBJECT: ", ch_names_from_info)

        print("THESE ARE THE CHANNEL NAMES: ", ch_names)
        print("THESE ARE THE CHANNEL NAME SHAPE: ", len(ch_names))
        


        
        for ch_idx, ch_name in enumerate(ch_names):

            ## Compute PSD fit
            psd_fix = PSD()

            psd_fix = PSD(freqs, power[ch_idx])

            psd_fix.fit(method='huber')

            psd_list.append(psd_fix)

            # plot power spectrum - aperiodic fit for each channel
            fig, ax = plt.subplots()     
            psd_fix.plot(ax=ax)      
            fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"sub-{sub}_psd_fix_ch-{ch_name}.png"))
            plt.close(fig)

            # compute acf fit 

            # initialize ACF object (1D array of power)
            acf_fix = ACF()
            
            acf_fix.compute_acf(power[ch_idx], epochs.info['sfreq'])

            # plot acf
            fig, ax = plt.subplots()

            # exponential decay model
            acf_fix.fit()
            acf_fix.plot(ax=ax, title='Exponential Decay Model')

            fig.savefig(os.path.join(DERIV_DIR, 'timescales', f"sub-{sub}_acf_fix_ch-{ch_name}.png"))
            plt.close(fig)

    
    ## save timescale parameters to list 



   ### Try to compute topography plots of timescales
    ## okay, 



        

def timescales_single_trials(sub):
    """ Computes psd and acf subjects for each epoch x channel pair.

    Args:
        sub (_type_): subject to be processed

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
    epochs_fixation = epochs_eeg['Fixation Onset Enc'] # epochs.fif object


    # compute psd per epoch; spectral representation of each epoch stored in EpochsSpectrum object
    psd_fixation = epochs_fixation.compute_psd(fmin = 2, fmax = 30)

    # now call get data method
    print(psd_fixation.get_data().shape)  # >> (430, 32, 56)

    # compute power and frequencies
    power, freqs = psd_fixation.get_data(return_freqs = True)

    print(power.shape) # >> (430, 32, 56)
    print(freqs.shape) # >> (56,)


    # NOT IDEAL: because ideally I also want to keep track of the channel names
    # initialize empty list object
    psd_list = []


    ## Nope, this doesn't really help
    # map channel type to a list of channel indices using mne.channel_indices_by_type
    # ch_idx_type = mne.channel_indices_by_type(epochs_eeg.info, picks = "eeg")

    #print("CHANNEL INDICES BY TYPE DICTIONARY:", ch_idx_type)


    # for ch_idx in range(power.shape[1]):
            
    #         print(ch_idx) 

    #         for epoch_idx in range(power.shape[0]):

    #             psd_fix = PSD()

    #             psd_fix = PSD(freqs, power[epoch_idx, ch_idx]) # index both the epoch and the channel

    #             # append them to list
    #             psd_list.append(psd_fix)

    #             # fit 
    #             psd_fix.fit(method='huber')


    # # EXPECTED: 430x 32 timescales PSD objects
    # print('Length of PSD list', len(psd_list))  # >> returns correct dimensions: 13760 list entries









if __name__ == "__main__":
    # analyses(sub=param1)
   # timescales_single_trials(sub=param1)
    timescales_average_trials(sub=param1)

