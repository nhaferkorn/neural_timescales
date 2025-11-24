"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import fooof 
import glob
import re
import pickle

sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED

########################################################
## SETUP

# set system variables
param1 = sys.argv[1]

# Toggle ON or OFF - To BE FIXED: they depend on one another if I dont save and load intermediate results
RUN_PLOT_EVENTS = False
RUN_ANNOTATIONS = True
FIND_EOGs = False
RUN_ICA = True # you only need to run this once per subject, because it then saves the ica solution
APPLY_ICA = True 
RUN_EPOCHS = False


def preprocessing(sub):

    # Print subject label 
    print('\nCURRENTLY RUNNING SUBJECT: ', sub, '\n')

    rawfile = os.path.join(EEG_DIR, "%s.bdf") % sub

    raw = mne.io.read_raw_bdf(rawfile, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS", infer_types=True)
    
    raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    # create VEOG and HEOG channels
    raw = mne.set_bipolar_reference(raw, 'EXG3', 'EXG4', ch_name='VEOG')
    raw = mne.set_bipolar_reference(raw, 'EXG1', 'EXG2', ch_name='HEOG')

    # specify line noise frequency
    # mne.filter.notch_filter
    raw.info['line_freq'] = 50.

    # set montage directly on raw object (and not a copy of it!) > this creates a new entry in info object
    raw.set_montage("biosemi32", on_missing = "ignore")
    
    # set average reference: 
    raw_ref = raw.copy().set_eeg_reference(ref_channels = 'average')

    # filter the data (choose same filter settings as orig paper)
    raw_ref = raw_ref.copy().filter(l_freq = 0.5, h_freq=30)
    
    # find events
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
    events[:,2] = events[:, 2] - 64512

    # keys of all the events of interest (move into settings file)
    keys = {'Start Practice Trial',
        'Start Encoding','Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
         ,'Fixation Onset Enc', 'Cue Onset', 
         'Response Natural', 'Response Manmade', 'Response None Enc',
        'Fixation Onset Enc', 'Cue Onset', 'Rest onset', 'Rest offset', 'End Encoding',
        'Start Retrieval', 'Retrieval Stimulus Onset Baseline Left', 'Retrieval Stimulus Onset Baseline Right',
        'Retrieval Stimulus Onset Distraction Left Target', 'Retrieval Stimulus Onset Distraction Right Target',
        'Retrieval Stimulus Onset Distraction Right Distractor', 'Retrieval Stimulus Onset Distraction Left Distractor',
        'Retrieval Stimulus Onset New', 'Response Old', 'Response New', 'Response None ON',
        'Confidence Onset', 'Response Confidence 1', 'Response Confidence 2', 'Response Confidence 3', 'Response Confidence None',
        'Fixation Onset Ret', 'End Retrieval'}


    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    if RUN_PLOT_EVENTS:      
        fig = mne.viz.plot_events(
        events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest, on_missing='ignore')
        fig.suptitle(f"Events-{sub}", fontsize=14)

        fig.savefig(os.path.join(DERIV_DIR, 'Events', f'Events_Sub-{sub}.png'))


    if RUN_ANNOTATIONS:

        ##  Crop the signal before Practice Trial 1 and End of Retrieval
        events_mapping_blocks = mne.pick_events(events, include = [99, 93])
        print(events_mapping_blocks)

        # events dict
        mapping_blocks = {99:"Start Practice Trial", 93:"End Retrieval"}

        # convert events array into annotation
        annotations_mapping_blocks = mne.annotations_from_events(events, sfreq = raw_ref.info["sfreq"], event_desc = mapping_blocks)

        # this should yield 3 annotations for most subjects
        for ann in annotations_mapping_blocks:
            print(ann["onset"])
        
        # plot those two events for sake of sanity check
        raw_ref.plot(events_mapping_blocks, event_id = mapping_blocks, block = True)

        # crop the data 
        print(f"Onset Practice Trial 1", annotations_mapping_blocks[0]["onset"])
        print(f"End Retrieval", annotations_mapping_blocks[2]["onset"])

        print("\nNOW CREATING THE CROPPED SIGNAL\n")
        raw_crop = raw_ref.copy().crop(tmin = annotations_mapping_blocks[0]["onset"], tmax = annotations_mapping_blocks[2]["onset"])


        ## Annotate the rest breaks (Double check if this works as intended)
        
        # Annotate rest periods - ADD some output print messages
        mapping_rests = {90:"Rest onset", 91:"Rest offset"}

        annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw_crop.info["sfreq"], orig_time=raw_crop.info["meas_date"],)

        # this should reset the annotations - but is this something I want?
        raw_crop.set_annotations(annot_from_events_rests)

        print("These are the onsets of the annotations for the rest periods:", raw_crop.annotations.onset)
        
        onsets_rests = []
        durations_rests = []

        # okay, this works; but might not be the most elegant solution
        for i in range(0, len(raw_crop.annotations.onset)-1, 2):
            
            # set onset of rest
            onsets_rest = raw_crop.annotations.onset[i]
            onsets_rests.append(onsets_rest)

            # set duration of rest
            durations_rest = raw_crop.annotations.onset[i+1] - raw_crop.annotations.onset[i]
            durations_rests.append(durations_rest)


        print(onsets_rests)
        print(durations_rests)

        descriptions_rest = ["BAD_Rest Period"] * len(onsets_rests)
        orig_times = raw_crop.info["meas_date"]


        rest_annots = mne.Annotations(
            onset=onsets_rests, 
            duration=durations_rests,
            description=descriptions_rest,
            orig_time=orig_times)


        # are these then saved or do they need to be reloaded?
        raw_crop.set_annotations(annot_from_events_rests + rest_annots)

        ## Annotate bad spans of data
        # check if annotations.fif file already exists 
        if os.path.exists(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif")):
            print(f"\nANNOTATION FILE FOR SUB-{sub} ALREADY EXISTS\n")
            print("\nREADING ANNOTATIONS FROM .FIF FILE")
            annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))
            print(annot_from_file)

            # now, if we want to set new annotations in interactive plot mode
            raw_crop.set_annotations(annot_from_file, emit_warning=False)
            raw_crop.plot(block=True)

            print("\nSAVING NEWLY ADDED ANNOTATIONS TO FILE \n")
            raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"), overwrite=True)
        
        else:
            print(f"\nANNOTATION FILE FOR SUB-{sub} DOESN'T EXIST\n")
            # load raw lot to create annotations and save them
            raw_crop.plot(block=True)
            # save initially created annotations to file
            raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"), overwrite=True)


    if FIND_EOGs: 
        # load saved annotations 
        annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        # # set annotations
        raw_crop.set_annotations(annot_from_file)

        # find eog events & plot them on top of raw data
        eog_events = mne.preprocessing.find_eog_events(raw_crop, reject_by_annotation = True)
        raw_crop.plot(events=eog_events, block = True)

        # print("SANITY CHECK: ANNOTATIONS ARE SET:",  raw_crop.annotations)

        # # create eog epochs pass reject_by_annotation parameter 
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_crop.filter(1, 10), reject_by_annotation = True)
        eog_plot = eog_epochs.average().plot_joint(title= f"EOG Epochs Plot - Sub-{sub}")
        eog_plot.savefig(os.path.join(DERIV_DIR, 'Fixed', 'After', 'EOG_epochs', f'{sub}_eog_epochs.png'))

        eog_evoked = mne.preprocessing.create_eog_epochs(raw_crop.filter(1, 10), reject_by_annotation = True).average(picks="all")
        eog_evoked.apply_baseline((None, None))
        eog_evoked.plot()  

        # compute the SSP Projections based on the EOG epochs
        eog_projs, _ = mne.preprocessing.compute_proj_eog(
        raw_crop, reject=None)

        # visualize the projections
        mne.viz.plot_projs_topomap(eog_projs, info=raw_crop.info)


    if RUN_ICA:

        raw_ica_filtered = raw_crop.copy().filter(l_freq = 1., h_freq=30)
        ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
        ica.fit(raw_ica_filtered, reject_by_annotation=True)

        # Save ICA solution
        ica.save(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))

        # plot components
        fig_sources = ica.plot_sources(raw_crop, title=f"AFTER: ICs Timecourses for Sub-{sub}", show_scrollbars=False) 
        fig_components = ica.plot_components(title=f"ICs for Sub-{sub}")
        fig_components.savefig(os.path.join(DERIV_DIR, "ica", f'{sub}_ics.png'))

   
    if APPLY_ICA:

        # but I guess it also should be possible to save the ICA solution in the previous step & load it again
        ica = mne.preprocessing.read_ica(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))
        
        if not os.path.isfile(os.path.join(DERIV_DIR, 'ica', f'{sub}_components')):

            # ask for user input to indicate the components
            components = [int(x) for x in input("ENTER INDICES OF COMPONENTS TO REMOVE: ").split()]
            
            # save subject solution as pickle object
            with open(os.path.join(DERIV_DIR, 'ica', f'{sub}_components'), 'wb') as fp:
                pickle.dump(components, fp)
                components_loaded = pickle.load(fp)

        else:
            print(f"\nCOMPONENTS FOR SUB-{sub} EXIST & ARE BEING LOADED\n")

            # load subject solution using pickle
            with open(os.path.join(DERIV_DIR, 'ica', f'{sub}_components'), 'rb') as fp:
                components_loaded = pickle.load(fp)

        print("\nLOADED THE FOLLOWING COMPONENTS:", components_loaded, "\n")
            
        # select which components to exclude
        ica.exclude = components_loaded

        # ica.apply changes Raw object in-place, so let's make a copy first
        reconst_raw = raw_crop.copy()
        ica.apply(reconst_raw)

        # plot the signal with blinks removed
        raw_crop.plot(block=True)

        # plot the signal after applying ICA and removal of ocular components 
        reconst_raw.plot(block=True)

        ## save the cleaned raw data (i.e. reconst_raw) in a new .fif file for each subject, in subject-specific folder
        # Create subject-specific folder 
        RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, f"sub-{sub}")

        if not os.path.isdir(RAW_CLEANED_SUB):
            os.mkdir(RAW_CLEANED_SUB)

        print("\nSAVING PREPROCESSED DATA\n ")

        reconst_fname = os.path.join(RAW_CLEANED_SUB, f'sub-{sub}-raw_cleaned.fif')
        reconst_raw.save(reconst_fname)


########################################################
## EPOCHING 

    if RUN_EPOCHS: 

        # load subjects ICA solution
        print(f"\nLOADING CLEANED DATA FOR SUB-{sub} \n")
        reconst_fname = os.path.join(RAW_CLEANED, f'sub-{sub}-raw_cleaned.fif')
        reconst_raw = mne.io.read_raw_fif(reconst_fname)

        # sanity check 
        print("INFO OBJECT OF:", reconst_raw.info)


        print(f"NOW RUNNING EPOCHS FOR SUB-{sub}\n")

        # input: cleaned data (after ICA and manual rejection of bad spans)
        epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject_by_annotation=True)
        
        # plot how many epochs were dropped 
        epochs_after_rejection = epochs.drop_bad()

        plot_dropped = epochs_after_rejection.plot_drop_log()

        # adjust output directory
        # plot_dropped.savefig(os.path.join())

        # compute the channel stats based on a drop_log from epochs (returns total percentage of epochs dropped)
        dropped_percent = epochs_after_rejection.drop_log_stats()
        print(dropped_percent)

        # save the epochs


if __name__ == "__main__":
    preprocessing(sub=param1)
