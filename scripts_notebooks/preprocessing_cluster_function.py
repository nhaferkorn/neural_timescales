"""This script pre-processes the raw EEG data."""

import os
import numpy as np
import mne
import glob
import re

# think about how this translates to the cluster
from src.settings import DATA_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, REPORT_DIR, DERIV_DIR


########################################################
## SETUP

# specify directories 
# or maybe it actually works if I import it...

RUN_ICA = True
RUN_EPOCHS = False

# i somehow need to specify jobs for the individual subjects; I guess I would also need to run eeg2bids first

def preprocessing(subj_id):

    # Print subject label and running
    print('\nCURRENTLY RUNNING SUBJECT: ', subj_id, '\n')

    rawfile = glob.glob(os.path.join(EEG_DIR, "*.bdf"))

    raw = mne.io.read_raw_bdf(rawfile, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS")
    raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    # okay, so  I have to set montage directly on raw object (and not a copy of it!) > this creates a new entry in the info object
    raw.set_montage("biosemi32", on_missing = "ignore")

    # filter the data
    raw_filtered = raw.copy().filter(l_freq = 0.1, h_freq=45)

    # instantiate report object and specify output directory
    report = mne.Report(title= f"Sub-{subj_id} Report")
    # yeah, I don't think it is possible to save it as a pdf
    output_path = os.path.join(REPORT_DIR, f"subject{subj_id}_report.html")
    
    # find events
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False)
    events[:,2] = events[:, 2] - 64512

    keys = {'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
         ,'Fixation Onset Enc', 'Cue Onset'}

    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    report.add_events(events=events, title='Events Plot', sfreq = raw.info['sfreq'])
    report.save(output_path, overwrite=True)

    # save events plot in derivatives directory 
    fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest)
    fig.suptitle(f"Events-{subj_id}", fontsize=14)
    fig.savefig(os.path.join(DERIV_DIR, f'events_{subj_id}.png'))

    if RUN_ICA:
            raw_ica_filtered = raw.copy().filter(l_freq = 1, h_freq=40)
            ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
            ica.fit(raw_ica_filtered)

            # plot components
            fig = ica.plot_components(title=f"ICs for {subj_id}")
            fig.savefig(os.path.join(DERIV_DIR, f'ics_{subj_id}.png'))

            # it might also makes sense to save the subjects ICA solution 
    else:
        continue




    # Implement check-point: only move on if raw_ica exist!! - this doesn't work straight away
    # if raw_ica: # not sure if this works...
    #      print('Blink Rejection Completed')
    # else:
    #      raise FileNotFoundError

    ## set average reference: 
    # reconst_raw_ref = reconst_raw.copy().set_eeg_reference(ref_channels = 'average')

    # Epochs based on peak-to-peak channel amplitude 
    if RUN_EPOCHS: 
        epochs = mne.Epochs(raw_filtered, events = events, event_id = events_of_interest, tmin = -2, tmax=2, baseline=None, reject = reject_criteria, flat = flat_criteria)
        
        reject_criteria = dict(
            eeg=200e-6,  # 100 µV
            eog=200e-6,    # 200 µV
        ) 
        flat_criteria = dict(eeg=1e-6)  # 1 µV

        # plot how many epochs were dropped 
        epochs_dropped = epochs.drop_bad()

        epochs_dropped.plot_drop_log()

        # compute the channel stats based on a drop_log from epochs
        epochs.drop_log_stats()
        # save that information to a new file (maybe just a txt file)

        # plot power spectra
        epochs["Encoding Stimulus Onset Baseline Left"].compute_psd().plot(picks="eeg")

        # # save this figure of how many epochs were dropped to subject report
        # report.add_epochs(epochs=epochs, title='Epochs Object')
        # report.save(output_path, overwrite=True)

    
    
    
    
    
    ## EVOKED RESPONSES 

    # compute evoked response; epochs are averaged in a channel-wise fashion
    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average()

    evoked_enc_base_left.plot(titles="left encoding")


    ## plot evoked response relative to Fixation Onset Encoding - we see a decay in power over time (which makes sense, but is also interesting)
    evoked_enc_fix = epochs["Fixation Onset Enc"].average()
    evoked_enc_fix.plot()

    # Compare left vs. right presented stimuli in encoding phase 
    # pick occipital electrodes
    picks = mne.pick_channels(raw.info["ch_names"], ['O1', 'Oz' ,'O2'])
    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average(picks)
    evoked_enc_base_left.plot(titles="left")

    ## Interesting: so, we definitely see a positive deflection centered at around 0.1 Hz, and then the signal plateus afterwards
    evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average(picks)
    evoked_enc_base_right.plot();

    # I guess it make sense to pool over targets and baseline stimuli and then check for left and right side separately 
    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average().plot()
    evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average().plot()

    # I don't understand why 
    evoked_enc_dist_left = epochs['Encoding Stimulus Onset Distraction Left Target'].average().plot()

    # now, we get a much flatter power spectrum
    enc_pooled_left = epochs[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']].average().plot()

    # Evoked estimates for retieval phase
    evoked_ret_dist_left = epochs['Retrieval Stimulus Onset Distraction Left Target'].average().plot()
    evoked_ret_base_left = epochs['Retrieval Stimulus Onset Baseline Left'].average().plot()