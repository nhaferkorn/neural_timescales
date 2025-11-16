"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import glob
import re

sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR

########################################################
## SETUP

# set system variables
param1 = sys.argv[1]

# Toggle ON or OFF
RUN_PLOT_EVENTS = False
RUN_ANNOTATIONS = False
FIND_EOGs = True
RUN_ICA = True
RUN_EPOCHS = False
RUN_EVOKED = False


def preprocessing(sub):

    # Print subject label and running
    print('\nCURRENTLY RUNNING SUBJECT: ', sub, '\n')

    rawfile = os.path.join(EEG_DIR, "%s.bdf") % sub

    print(rawfile)

    raw = mne.io.read_raw_bdf(rawfile, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS", infer_types=True)
    
    raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    #  create VEOG and HEOG channels
    #  heogl (horizontal EOG left): EXG1
    #  heogr (horizontal EOG right): EXG2
    #  veogt (vertical EOG; top): EXG3
    #  veogb (vertical EOG; bottom): EXG4

    raw = mne.set_bipolar_reference(raw, 'EXG3', 'EXG4', ch_name='VEOG')
    raw = mne.set_bipolar_reference(raw, 'EXG1', 'EXG2', ch_name='HEOG')

    # notch filter the data to remove 50Hz noise
    # mne.filter.notch_filter
    raw.info['line_freq'] = 50.

    # set montage directly on raw object (and not a copy of it!) > this creates a new entry in info object
    raw.set_montage("biosemi32", on_missing = "ignore")
    
    # set average reference: 
    raw_ref = raw.copy().set_eeg_reference(ref_channels = 'average')

    # filter the data (choose same filter settings as orig paper)
    raw_ref = raw_ref.copy().filter(l_freq = 0.5, h_freq=30)
    
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


    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    if RUN_PLOT_EVENTS:      
        fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest)
        fig.suptitle(f"Events-{sub}", fontsize=14)

        fig.savefig(os.path.join(DERIV_DIR, f'events_{sub}.png'))

    if RUN_ANNOTATIONS:
         
        # check if annotations.fif file already exists
        if os.path.exists(os.path.join(DERIV_DIR, "Raw_Annotations", f"{sub}-annotations.fif")):
            print(f"ANNOTATION FILE FOR SUB-{sub} ALREADY EXISTS\n")
            print(" READING ANNOTATIONS FROM .FIF FILE")
            annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "Raw_Annotations", f"{sub}-annotations.fif"))
            print(annot_from_file)

            # now, if we want to set new annotations in interactive plot mode
            raw_ref.set_annotations(annot_from_file, emit_warning=False)
            raw_ref.plot(block=True)

            print("SAVING NEWLY ADDED ANNOTATIONS TO FILE")
            raw_ref.annotations.save(os.path.join(DERIV_DIR, "Raw_Annotations", f"{sub}-annotations.fif"), overwrite=True)
        
        else:
            print(f"ANNOTATION FILE FOR SUB-{sub} DOESN'T EXIST\n")
            # load raw lot to create annotations and save them
            raw_ref.plot(block=True)
            # save initially created annotations to file
            raw_ref.annotations.save(os.path.join(DERIV_DIR, "Raw_Annotations", f"{sub}-annotations.fif"), overwrite=True)


    if FIND_EOGs: 
        # load saved annotations 
        annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "Raw_Annotations", f"{sub}-annotations.fif"))

        # # set annotations
        raw_ref.set_annotations(annot_from_file)

        # # double check that they are really added to raw_ref object - seems like they are
        print("SANITY CHECK: ANNOTATIONS ARE SET:",  raw_ref.annotations)

        # pass reject_by_annotation parameter (but I don't think that they are recognized) - maybe I need to set them more explicitly
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_ref.filter(1, 10), reject_by_annotation = True)
        eog_plot = eog_epochs.average().plot_joint(title= f"AFTER: EOG Epochs Plot - Sub-{sub}")
        eog_plot.savefig(os.path.join(DERIV_DIR, 'Fixed', 'After', 'EOG_epochs', f'{sub}_eog_epochs_bad_rejected_onlyend.png'))



    if RUN_ICA:
            print(raw_ref.info)
            raw_ica_filtered = raw_ref.copy().filter(l_freq = 1., h_freq=30)
            ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
            ica.fit(raw_ica_filtered, reject_by_annotation=True)

            # plot components
            fig = ica.plot_components(title=f"AFTER: ICs for Sub-{sub}")
            fig.savefig(os.path.join(DERIV_DIR, "Fixed", 'After', 'ICA_results', f'{sub}_avgref_filt_1_30_onlyend.png'))

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

        # maybe also try rejection without Fp1 and Fp2 electrodes 

        epochs = mne.Epochs(raw_ref, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject = reject_criteria, flat = flat_criteria)
        

        # plot how many epochs were dropped 
        epochs_after_rejection = epochs.drop_bad()

        plot_dropped =  epochs_after_rejection.plot_drop_log()
        plot_dropped.savefig(os.path.join(DERIV_DIR, 'Fixed', 'Dropped_veog_eog_300_microvolt', f'{sub}_drop_log_eeo_veog_heog_reject_shorter_epochs.png'))


        # compute the channel stats based on a drop_log from epochs (returns total percentage of epochs dropped)
        dropped_percent = epochs_after_rejection.drop_log_stats()
        # save that information to a new file (maybe just a txt file)

        print(dropped_percent)
        print(type(dropped_percent))

        # # still need to fix that it doesn't overwrite
        # with open(os.path.join(DERIV_DIR, "Evokeds", 'drop_log_stats_only_eog_reject.txt', 'w')) as file:
        #         file.write(f"{sub}: {dropped_percent}\n")

        # plot power spectra and save figure
        # psd = epochs_after_rejection.compute_psd().plot(picks="eeg")
        # psd.savefig(os.path.join(DERIV_DIR, f'{sub}_psd.png'))

        # # save this figure of how many epochs were dropped to subject report
        # report.add_epochs(epochs=epochs, title='Epochs Object')
        # report.save(output_path, overwrite=True)

        # plot the dropped epochs


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
        evoked_right.savefig(os.path.join(DERIV_DIR, f'evoked_enc_right_{sub}.png'))
        evoked_base.savefig(os.path.join(DERIV_DIR, f'evoked_enc_baseline_{sub}.png'))
        evoked_dist.savefig(os.path.join(DERIV_DIR, f'evoked_enc_distraction_{sub}.png'))
        evoked_cue.savefig(os.path.join(DERIV_DIR, f'evoked_enc_fixcue_{sub}.png'))

        # evokeds = dict(low_dist=l_aud, high_dist=l_vis)
        # fig_evoked_compared mne.viz.plot_compare_evokeds(evokeds,  combine="mean")
        # fig_evoked_compared.savefig()


        # evoked responses for retrieval phase
        # evoked_ret_dist_left = epochs['Retrieval Stimulus Onset Distraction Left Target'].average().plot()
        # evoked_ret_base_left = epochs['Retrieval Stimulus Onset Baseline Left'].average().plot()


if __name__ == "__main__":
    preprocessing(sub=param1)
