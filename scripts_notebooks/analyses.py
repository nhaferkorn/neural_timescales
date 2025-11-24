"""This script analyses the cleaned EEG data"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import fooof 
import re
import pickle


# import functions from timescales package - not sure if it is actually imported
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

from timescales.sim import sim_ar
from timescales.fit import ARPSD, PSD
from timescales.autoreg import compute_ar_spectrum
from timescales.plts import set_default_rc

import importlib.util

# double check whether packages are installed
package_name = 'timescales'

if importlib.util.find_spec(package_name) is None:
    print(package_name +" is not installed")



sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, events_of_interest

########################################################
## SETUP

# set system variables
param1 = sys.argv[1]

RUN_EPOCHS = True
RUN_EVOKED = False
RUN_TIMESCALES = True


## Define Main Analysis Function
def analyses(sub):

    
########################################################
## CREATE EPOCHS

    if RUN_EPOCHS: 

        # load cleaned data for subject
        print(f"LOADING CLEANED DATA FOR SUB-{sub} \n")
        reconst_fname = os.path.join(RAW_CLEANED, f'sub-{sub}-raw_cleaned.fif')
        

        # it takes forever to load, therefore specify preload = False
        reconst_raw = mne.io.read_raw_fif(reconst_fname, preload=False)

        # find events (not sure if I have to load events again...)
        events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
        events[:,2] = events[:, 2] - 64512

        print(f"NOW RUNNING EPOCHS FOR SUB-{sub} \n")

        # input: cleaned data (post ICA + manual rejection)
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject_by_annotation=True)

        # save the epochs
        print("SUCCESSFULLY CREATED EPOCHS")

########################################################
## EVOKED RESPONSES 

    if RUN_EVOKED:

        # compute evoked response
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None)
        # evoked responses for encoding phase
        evoked_left = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Distraction Left Target"]].average().plot(titles=f"{sub} Evoked - Left Attend", picks = 'eeg')
        evoked_right = epochs[["Encoding Stimulus Onset Baseline Right", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - Right Attend", picks = 'eeg')
        evoked_base = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Baseline Right"]].average().plot(titles=f"{sub} Evoked - Low Distraction Attend", picks = 'eeg')
        evoked_dist = epochs[["Encoding Stimulus Onset Distraction Left Target", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - High Distraction Attend", picks = 'eeg')
        evoked_cue = epochs["Fixation Onset Enc"].average().plot(titles=f"{sub} Evoked - Fixation Cue", picks = 'eeg')

        # save the evoked plots 
        evoked_left.savefig(os.path.join(DERIV_DIR, f'evoked_enc_left_{sub}.png'))
        evoked_right.savefig(os.path.join(DERIV_DIR, f'evoked_enc_right_{sub}.png'))
        evoked_base.savefig(os.path.join(DERIV_DIR, f'evoked_enc_baseline_{sub}.png'))
        evoked_dist.savefig(os.path.join(DERIV_DIR, f'evoked_enc_distraction_{sub}.png'))
        evoked_cue.savefig(os.path.join(DERIV_DIR, f'evoked_enc_fixcue_{sub}.png'))

        # evoked responses for retrieval phase
        # evoked_ret_dist_left = epochs['Retrieval Stimulus Onset Distraction Left Target'].average().plot()
        # evoked_ret_base_left = epochs['Retrieval Stimulus Onset Baseline Left'].average().plot()
        


########################################################
## TIMESCALE ANALYSES 

    if RUN_TIMESCALES: 
        print(f"FITTING TIMESCALES FOR SUB-{sub}")

        # check info object as sanity check - seems okay
        print(reconst_raw.info)
    
        ## Copied from Neurodsp Tutorial
                
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
        sig = np.squeeze(sig) 

        # double check the data dimensions

        # compute psd of snippet of the raw data
        # Calculate the power spectrum, using median Welch's & extract a frequency range of interest
        # freqs, powers = compute_spectrum(sig, fs, method='welch', avg_type='median')
        # freqs, powers = trim_spectrum(freqs, powers, [3, 30])

        # # print freqs, and powers
        # print(f"These are the frequencies", freqs)
        # print(f"These are the powers", powers)

        # # # initialize power spectrum object - Okay - So compute spectra works!
        # # initialize PSD object
        # psd = timescales.fit.PSD()

        # freqs2, powers2 = compute_spectrum(sig, fs = reconst_raw.info["sfreq"])

        # psd = timescales.fit.PSD(freqs, powers)

        # # Now fit psd using fooof aperiodic model & robust regression (need to check what it means!!)
        # psd.fit(method = "huber")

        # # check psd fit
        # print(psd)
        
        # psd.plot()
        # # plot the fit
        # print(type(psd.plot())) # >> interesting: this becomes NoneType (not sure why)
        # #.savefig(f"psd_fit_sub-{sub}.png")   


        # Try again - Fit FOOOF robustly - nope, this also doesn't work...
        psd_robust = PSD()
        psd_robust.compute_spectrum(sig, fs)
        psd_robust.fit(method='huber')

        fig, ax = plt.subplots()      # create the figure
        psd_robust.plot(ax=ax)        # draw into *your* axes
        fig.savefig(f"psd_fit_sub-{sub}.png")
        plt.close(fig)
        # print(dir(psd_robust))
        # print(psd_robust.params, "\n")
        # print("Parameter names:", psd_robust.param_names)
        # psd_robust.plot().savefig(f"psd_fit_sub-{sub}.png")   

        # not sure why plot fails


        

if __name__ == "__main__":
    analyses(sub=param1)
