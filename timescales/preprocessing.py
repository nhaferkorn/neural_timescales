"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import fooof 
import pickle

sys.path.append('/project/4180000.57/neural_timescales/timescales')

# import variables and paths
from settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys


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
    raw.info['line_freq'] = 50

    # set montage directly on raw object (and not a copy of it!) > this creates a new entry in info object
    raw.set_montage("biosemi32", on_missing = "ignore")
    
    # set average reference: 
    raw.set_eeg_reference(ref_channels = 'average')

    # bandpass filter the data 
    raw.filter(l_freq = 0.5, h_freq=30)
    
    # find events & subtract marker offset
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
    events[:,2] = events[:, 2] - 64512

    return raw, events


def plot_events(raw, sub):  
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
    events[:,2] = events[:, 2] - 64512

    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    fig = mne.viz.plot_events(
    events, sfreq=raw.info["sfreq"], first_samp=raw.first_samp, event_id=events_of_interest, on_missing='ignore')
    fig.suptitle(f"Events-{sub}")

    fig.savefig(os.path.join(DERIV_DIR, 'Events', f'Sub-{sub}_Events.png'))


def annotate_rest(raw, events, sub):
        ##  Crop the signal before Practice Trial 1 and End of Retrieval
        events_mapping_blocks = mne.pick_events(events, include = [10, 93])

        # events dict
        mapping_blocks = {10:"Start Encoding", 93:"End Retrieval"}

        # convert events array into annotation
        annotations_mapping_blocks = mne.annotations_from_events(events, sfreq = raw.info["sfreq"], event_desc = mapping_blocks)

        # should result 3 annotations for most subjects
        for ann in annotations_mapping_blocks:
            print(ann["onset"])
        
        assert len(annotations_mapping_blocks) == 2, 'not the right dimensions'
     
        # sanity check: plot those two events 
        raw.plot(events_mapping_blocks, event_id = mapping_blocks, block = True)

        # crop the data and cut-off segments before the start of encoding & after the end of retrieval
        print(f"Start Encoding", annotations_mapping_blocks[0]["onset"])
        print(f"End Retrieval", annotations_mapping_blocks[1]["onset"])

        print("\nNOW CREATING THE CROPPED SIGNAL\n")
        raw_crop = raw.copy()
        raw_crop.crop(tmin = annotations_mapping_blocks[0]["onset"], tmax = annotations_mapping_blocks[1]["onset"])
        

        ## Annotate rest periods 
        mapping_rests = {90:"Rest onset", 91:"Rest offset"}

        annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw_crop.info["sfreq"], orig_time=raw_crop.info["meas_date"],)
        
        raw_crop.set_annotations(raw_crop.annotations + annot_from_events_rests)  # this line is crucial

        print("\nONSETS OF ANNOTATIONS FOR REST PERIODS:", raw_crop.annotations.onset)
        
        onsets_rests = []
        durations_rests = []

        for i in range(0, len(raw_crop.annotations.onset)-1, 2):
            
            # set onset of rest
            onsets_rest = raw_crop.annotations.onset[i]
            onsets_rests.append(onsets_rest)

            # set duration of rest
            durations_rest = raw_crop.annotations.onset[i+1] - raw_crop.annotations.onset[i]
            durations_rests.append(durations_rest)

        print('ONSET RESTS', onsets_rests)
        print('DURATIONS RESTS', durations_rests)

        descriptions_rest = ["BAD_Rest_Period"] * len(onsets_rests)
        orig_times = raw_crop.info["meas_date"]

        rest_annots = mne.Annotations(
            onset=onsets_rests, 
            duration=durations_rests,
            description=descriptions_rest,
            orig_time=orig_times)
        
        raw_crop.set_annotations(annot_from_events_rests + rest_annots)

        # save rest period annotations & markers
        if not os.path.exists(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif")):
            raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"))

        return raw_crop

def annotate_bad_spans(sub, raw_crop):

    if os.path.exists(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif")):
        ## Annotate bad spans of data
        print("\nREADING ANNOTATIONS FROM .FIF FILE")

        rest_annotations_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"))

        annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        print('THESE ARE THE ANNOTATIONS ALREADY SET', annot_from_file, rest_annotations_file)

        # annotate in interactive plot mode
        raw_crop.set_annotations(annot_from_file + rest_annotations_file, emit_warning=False)

        raw_crop.plot(block=True)

        print("\nSAVING NEWLY ADDED ANNOTATIONS OF BAD SEGMENTS TO FILE \n")
        raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"), overwrite=True)
    
    else:
        rest_annotations_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"))
        raw_crop.set_annotations(rest_annotations_file)

        raw_crop.plot(block=True)

        print("\nSAVING NEWLY ADDED ANNOTATIONS OF BAD SEGMENTS TO FILE \n")
        raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))
    




def find_eogs(raw_crop, sub):
        
        # load saved annotations 
        annot_from_file = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        # set annotations
        raw_crop.set_annotations(annot_from_file)

        # find eog events & plot them on top of raw data
        eog_events = mne.preprocessing.find_eog_events(raw_crop, reject_by_annotation = True)
        raw_crop.plot(events=eog_events, block = True)

        # # create eog epochs pass reject_by_annotation parameter 
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_crop.filter(1, 10), reject_by_annotation = True)
        eog_plot = eog_epochs.average().plot_joint(title= f"EOG Epochs Plot - Sub-{sub}")
        # eog_plot.savefig(os.path.join(DERIV_DIR, 'eog', f'{sub}_eog_epochs.png'))

        eog_evoked = mne.preprocessing.create_eog_epochs(raw_crop.filter(1, 10), reject_by_annotation = True).average(picks="all")
        eog_evoked.apply_baseline((None, None))
        eog_evoked.plot()  

        # compute the SSP Projections based on the EOG epochs
        eog_projs, _ = mne.preprocessing.compute_proj_eog(
        raw_crop, reject=None)

        # visualize the projections
        mne.viz.plot_projs_topomap(eog_projs, info=raw_crop.info)



def fit_ica(sub, raw_crop):

        raw_ica_filtered = raw_crop.copy().filter(l_freq = 1., h_freq=30)
        ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
        ica.fit(raw_ica_filtered, reject_by_annotation=True)

        # Save ICA solution
        ica.save(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))

        # plot components
        fig_sources = ica.plot_sources(raw_crop, title=f"AFTER: ICs Timecourses for Sub-{sub}", show_scrollbars=False) 
        fig_components = ica.plot_components(title=f"ICs for Sub-{sub}")
        fig_components.savefig(os.path.join(DERIV_DIR, "ica", f'{sub}_ics.png'))

   
def apply_ica(sub, raw_crop):
    # check if already cleaned data exists
    if os.path.isfile(os.path.join(DERIV_DIR, 'raw_cleaned', f'sub-{sub}-raw_cleaned.fif')):
        raise FileExistsError

    else:
        # load subjeccts ICA solution
        ica = mne.preprocessing.read_ica(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))
        
        if not os.path.isfile(os.path.join(DERIV_DIR, 'ica', f'{sub}_components')):

            # ask for user input to indicate the components (separated by spaces)
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

        print('PLOTTING SIGNAL AFTER BLINK REMOVAL')
        reconst_raw.plot(block=True)

        print("\nSAVING CLEANED DATA\n ")
        reconst_fname = f'sub-{sub}-raw_cleaned.fif'
        reconst_raw.save(os.path.join(RAW_CLEANED, reconst_fname))
    

def run_epochs(sub, events):

    # load subjects ICA solution
    print(f"\nLOADING CLEANED DATA FOR SUB-{sub} \n")
    reconst_fname = os.path.join(RAW_CLEANED, f'sub-{sub}-raw_cleaned.fif')
    reconst_raw = mne.io.read_raw_fif(reconst_fname)

    # define events of interest
    events_of_interest = {k: EVENT_DICT[k] for k in keys}

    # sanity check 
    print("INFO OBJECT OF:", reconst_raw.info)

    print(f"NOW RUNNING EPOCHS FOR SUB-{sub}\n")

    # input: cleaned data (after ICA and manual rejection of bad spans)
    epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1, tmax=1, baseline=(-0.5, 0), reject_by_annotation=True)
    
    # plot how many epochs were dropped 
    epochs_after_rejection = epochs.drop_bad()

    plot_dropped = epochs_after_rejection.plot_drop_log()

    # compute the channel stats based on a drop_log from epochs (returns total percentage of epochs dropped)
    dropped_percent = epochs_after_rejection.drop_log_stats()
    
    # print(dropped_percent)


if __name__ == "__main__":
    pass