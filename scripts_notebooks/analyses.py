"""This script analyses the cleaned EEG data"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import fooof 
import re
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
        RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, f"sub-{sub}")
        reconst_fname = os.path.join(RAW_CLEANED_SUB, f'sub-{sub}-raw_cleaned.fif')
        
        # it takes forever to load, therefore specify preload = False
        reconst_raw = mne.io.read_raw_fif(reconst_fname, preload=False)

        # find events (not sure if I have to load events again...)
        events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
        events[:,2] = events[:, 2] - 64512

        print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

        # input: cleaned data (post ICA + manual rejection)
        # Do I need to load the annotations again??
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject_by_annotation=True)

        # save the epochs
        print("SUCCESSFULLY CREATED EPOCHS")

########################################################
## EVOKED RESPONSES 

    if RUN_EVOKED:

        # compute evoked response
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None, reject_by_annotation = True)
        
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

        # to fit power spectrum models, we need to calculate power spectra
        print("CALCULATING POWER SPECTRA")

        # Run fooof on epoched data? - check documentation!
        psd = reconst_raw.compute_psd(method = "welch", fmin = 1, fmax=30, tmin=0, tmax=250, n_overlap=150, n_fft=300)

        spectra, freqs = psd.get_data(return_freqs=True)

        ## Fitting Power Spectrum Models
        # Initialize a FOOOFGroup object, with desired settings
        fg = fooof.FOOOFGroup(peak_width_limits=[1, 6], min_peak_height=0.15,
                peak_threshold=2., max_n_peaks=6, verbose=False)

        # Define the frequency range to fit
        freq_range = [1, 30]

        # Fit the power spectrum model across all channels
        fg.fit(freqs, spectra, freq_range)

        # Check the overall results of the group fits
        fg.plot(save_fig = True, file_name = f"fooof_sub-{sub}", )


########################################################
## TIMESCALE ANALYSIS

    if RUN_TIMESCALES: 
        print(f"\nFITTING TIMESCALES TUTORIAL\n")

        # Copied from Neurodsp Tutorial
        # Grab the sampling rate from the data
        fs = reconst_raw.info['sfreq']
        # Settings for exploring an example channel of data
        ch_label = 'P3'
        t_start = 20000
        t_stop = int(t_start + (10 * fs))

        # print ch_names
        print(reconst_raw.info['ch_names'])

        #vExtract an example channel to explore
        sig, times = reconst_raw.get_data(mne.pick_channels(reconst_raw.ch_names, [ch_label]),
                                start=t_start, stop=t_stop, return_times=True)
        
        # not sure why we need to squeeze the signal here
        print("\nSIGNAL BEFORE SQEEZING:\n", sig.shape) # (1,  10240)

        sig = np.squeeze(sig) 
        
        # check dimensions after squeezing
        print("\nSIGNAL AFTER SQEEZING:\n", sig.shape) # (10240,)

        # Fit FOOOF robustly 
        psd_robust = PSD()
        psd_robust.compute_spectrum(sig, fs)
        psd_robust.fit(method='huber')

        # okay this is interesting - so apparently psd.plot method returns an axes object
        fig, ax = plt.subplots() 
        psd_robust.plot(ax=ax)       
        fig.savefig(f"psd_fit_sub-{sub}.png")
        plt.close(fig)

   
########################################################
# On real epochs
        print(f"\nFITTING TIMESCALES FOR SUB-{sub}\n")

        # here we are averaging all of the epochs (after rejection bad spans) - but should I be doing this?
        epochs_fixation = epochs["Fixation Onset Enc"]

        
        print(f"\nEPOCHS LOCKED TO FIXATION ONSET", epochs_fixation, '\n')

        # plot them for all EEG channels
        # epochs_fixation.plot()

        # compute spectrum object: returns EpochsSpectrum objects -
        # and I probably need to call the get_data method
        psd_fixation = epochs_fixation.compute_psd(fmin = 2, fmax = 30)

        print(psd_fixation.ch_names)
        print(psd_fixation.freqs)

        # call the get data method to get spectrum data in NumPy array format
        print(psd_fixation.get_data())

        print(psd_fixation.get_data().shape)  # >> returns: (430, 32, 56)
        # I guess 56 refers to the frequencies?

######## Do it again, but this time average the epochs before 
        psd_fixation_average = epochs_fixation.average().compute_psd(fmin = 2, fmax = 30)
        print(psd_fixation_average.get_data())

        print(psd_fixation_average.get_data().shape) 

        power, freqs = psd_fixation_average.get_data(return_freqs = True)

        print(f'Power', power)
        print('Shape of power', power.shape)
        print('Freqs', freqs)

        print(freqs.shape)
        # What is the difference between EpochsSpectrum & EpochsSpectrumArray??


        ## Try to compute PSD fit using PSD object from timescales package
        psd_fix = PSD()

        # right now, I am just fitting it for a random channel - but maybe I should also pool across all channel
        psd_fix = PSD(freqs, power[1])

        psd_fix.fit(method='huber')

        fig, ax = plt.subplots()     
        psd_fix.plot(ax=ax)      
        fig.savefig(f"psd_fixation_window_fit_sub-{sub}.png")
        plt.close(fig)

##### Using ACF Objects
        acf_fix = ACF()
        
        # but this is for entire signal (reconst raw) after calling get_data method
        acf_fix.compute_acf(power[1], fs)

        # fitting acf & plotting 
        fig, ax = plt.subplots()

        # Exponential
        acf_fix.fit()
        acf_fix.plot(ax=ax, title='Exponential Decay Model')

        fig.savefig(f"acf_fit_fixation_window_sub-{sub}.png")
        plt.close(fig)

if __name__ == "__main__":
    analyses(sub=param1)

