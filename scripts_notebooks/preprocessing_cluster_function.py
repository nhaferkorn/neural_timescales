"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import glob
import re

# think about how this translates to the cluster
sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR

########################################################
## SETUP

# set system variables
param1 = sys.argv[1]

RUN_PLOT = False
RUN_ICA = True
RUN_EPOCHS = False
RUN_EVOKED = False


def preprocessing(sub):

    # Print subject label and running
    print('\nCURRENTLY RUNNING SUBJECT: ', sub, '\n')

    # not sure if this works - do I need even need glob.glob in that case?
    rawfile = os.path.join(EEG_DIR, "%s.bdf") % sub

    print(rawfile)

    raw = mne.io.read_raw_bdf(rawfile, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS")
    
    raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    # set montage directly on raw object (and not a copy of it!) > this creates a new entry in info object
    raw.set_montage("biosemi32", on_missing = "ignore")

    # filter the data
    raw_filtered = raw.copy().filter(l_freq = 0.1, h_freq=40)

    # set average reference: 
    raw_ref = raw_filtered.copy().set_eeg_reference(ref_channels = 'average')



    # instantiate report object and specify output directory
    # report = mne.Report(title= f"Sub-{sub} Report")
    # output_path = os.path.join(REPORT_DIR, f"subject{sub}_report.html")
    
    # find events
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False)
    events[:,2] = events[:, 2] - 64512

    # keys of all the events of interest
    keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
         ,'Fixation Onset Enc', 'Cue Onset', 
         'Response Natural', 'Response Manmade', 'Response None Enc',
        'Fixation Onset Enc', 'Cue Onset', 'Rest onset', 'Rest offset', 'End Encoding',
        'Start Retrieval', 'Retrieval Stimulus Onset Baseline Left', 'Retrieval Stimulus Onset Baseline Right',
        'Retrieval Stimulus Onset Distraction Left Target', 'Retrieval Stimulus Onset Distraction Right Target',
        'Retrieval Stimulus Onset Distraction Right Distractor', 'Retrieval Stimulus Onset Distraction Left Distractor',
        'Retrieval Stimulus Onset New', 'Response Old', 'Response New', 'Response None ON',
        'Confidence Onset', 'Response Confidence 1', 'Response Confidence 2', 'Response Confidence 3', 'Response Confidence None',
        'Fixation Onset Ret'}

    # i guess i can also merge some of the events into a new events (e.g. for Responses in the Encoding Phase)



    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    # report.add_events(events=events, title='Events Plot', sfreq = raw.info['sfreq'])
    # report.save(output_path, overwrite=True)

    # find eog epochs

    # use block argument to get the plot to open on the cluster 
    # raw_filtered.plot(block=True)

    #  eog_epochs = mne.preprocessing.find_eog_events(raw_filtered, l_freq = 1.5, h_freq=10)
    #  eog_epochs.plot()



    if RUN_PLOT:
    # save events plot in derivatives directory 
        fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest)
        fig.suptitle(f"Events-{sub}", fontsize=14)

        fig.savefig(os.path.join(DERIV_DIR, f'events_{sub}.png'))


    if RUN_ICA:
            raw_ica_filtered = raw_ref.copy().filter(l_freq = 1.5, h_freq=30)
            ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
            ica.fit(raw_ica_filtered)

            # plot components
            fig = ica.plot_components(title=f"ICs for {sub}")
            fig.savefig(os.path.join(DERIV_DIR, "Fixed",  f'{sub}_ics_1.5_30.png'))

            # save subjects solution 


    # Implement check-point: only move on if raw_ica exist!! - this doesn't work straight away
    # if raw_ica: # not sure if this works...
    #      print('Blink Rejection Completed')
    # else:
    #      raise FileNotFoundError



    # Epochs based on peak-to-peak channel amplitude 
    if RUN_EPOCHS: 

        print(f"\n Now Running Epochs for {sub} \n")

        reject_criteria = dict(
            eeg=100e-6,  # 100 µV
            eog=200e-6,    # 200 µV
        ) 
        flat_criteria = dict(eeg=1e-6)  # 1 µV

        epochs = mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None, reject = reject_criteria, flat = flat_criteria)
        

        # plot how many epochs were dropped 
        epochs_after_rejection = epochs.drop_bad()

        # epochs_after_rejection.plot_droplog()

        # compute the channel stats based on a drop_log from epochs (returns total percentage of epochs dropped)
        dropped_percent = epochs_after_rejection.drop_log_stats()
        # save that information to a new file (maybe just a txt file)

        print(dropped_percent)
        print(type(dropped_percent))

        # still need to fix that it doesn't overwrite
        with open(os.path.join(DERIV_DIR, "Evokeds", 'drop_log_stats.txt', 'w')) as file:
                file.write(f"{sub}: {dropped_percent}\n")

        # plot power spectra and save figure
        psd = epochs_after_rejection.compute_psd().plot(picks="eeg")
        psd.savefig(os.path.join(DERIV_DIR, f'{sub}_psd.png'))

        # # save this figure of how many epochs were dropped to subject report
        # report.add_epochs(epochs=epochs, title='Epochs Object')
        # report.save(output_path, overwrite=True)


    ## EVOKED RESPONSES 

    if RUN_EVOKED:

        # # rejection criteria
        # reject_criteria = dict(
        #     eeg=250e-6,  # 100 µV
        #     eog=200e-6,    # 200 µV
        # ) 
        # flat_criteria = dict(eeg=1e-6)  # 1 µV

        # compute evoked response
        # epochs = mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None, reject = reject_criteria, flat = flat_criteria)
        epochs = mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None)
        # evoked responses for encoding phase
        evoked_left = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Distraction Left Target"]].average().plot(titles=f"{sub} Evoked - Left Attend", picks = 'eeg')
        evoked_right = epochs[["Encoding Stimulus Onset Baseline Right", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - Right Attend", picks = 'eeg')
        evoked_base = epochs[["Encoding Stimulus Onset Baseline Left", "Encoding Stimulus Onset Baseline Right"]].average().plot(titles=f"{sub} Evoked - Low Distraction Attend", picks = 'eeg')
        evoked_dist = epochs[["Encoding Stimulus Onset Distraction Left Target", "Encoding Stimulus Onset Distraction Right Target"]].average().plot(titles=f"{sub} Evoked - High Distraction Attend", picks = 'eeg')
        evoked_cue = epochs["Fixation Onset Enc"].average().plot(titles=f"{sub} Evoked - Fixation Cue", picks = 'eeg')

        # save the evoked plots 
        evoked_left.savefig(os.path.join(DERIV_DIR, f'evoked_enc_left_{sub}.png'))
        evoked_right.savefig(os.path.join(DERIV_DIR, f'evoked_enc_right_{sub}