"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import fooof 
import pickle

# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys

# TODO: change function name
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

    # set montage directly on raw object 
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

    fig.savefig(os.path.join(DERIV_DIR, 'events', f'Sub-{sub}_Events.png'))


def annotate_rest(raw, events, sub):
    """Creates a cropped version of signal and annotates rest periods."""

    # events dict
    mapping_start_end = {10:"Start Encoding", 93:"End Retrieval"}

    # convert events array into annotations
    annotations_start_end = mne.annotations_from_events(events, sfreq = raw.info["sfreq"], event_desc = mapping_start_end)

    # print for sanity check
    print('ANNOTATIONS START_END ', annotations_start_end)

    # should result 2 annotations for most subjects
    for ann in annotations_start_end:
        print(ann["onset"])
    assert len(annotations_start_end) == 2, f'Not the right dimensions - expecting two markers but got {len(annotations_start_end)}'
    
    # # crop the data and cut-off segments before the start of encoding & after the end of retrieval
    print(f"Start Encoding", annotations_start_end[0]["onset"])
    print(f"End Retrieval", annotations_start_end[1]["onset"])

    print("\nNOW CREATING THE CROPPED SIGNAL\n")
    raw_crop = raw.copy()
    raw_crop.crop(tmin = annotations_start_end[0]["onset"], tmax = annotations_start_end[1]["onset"])
    
    # assert that raw_crop is shorter than raw, else the cropping didn't work
    assert raw_crop.duration <= raw.duration

    ## Annotate rest periods 
    mapping_rests = {90:"Rest onset", 91:"Rest offset"}

    annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw_crop.info["sfreq"], orig_time=raw_crop.info["meas_date"])
    
    raw_crop.set_annotations(raw_crop.annotations + annot_from_events_rests)

    print("\nONSETS OF ANNOTATIONS FOR REST PERIODS: ", raw_crop.annotations.onset)

    # set rest periods
    onsets_rests = []
    durations_rests = []

    for i in range(0, len(raw_crop.annotations.onset)-1, 2):
        
        # set onset of rest
        onsets_rest = raw_crop.annotations.onset[i]
        onsets_rests.append(onsets_rest)

        # set duration of rest
        durations_rest = raw_crop.annotations.onset[i+1] - raw_crop.annotations.onset[i]
        durations_rests.append(durations_rest)

    # print('ONSET RESTS', onsets_rests)
    # print('DURATIONS RESTS', durations_rests)

    descriptions_rest = ["BAD_Rest_Period"] * len(onsets_rests)
    orig_times = raw_crop.info["meas_date"]

    rest_annots = mne.Annotations(
        onset=onsets_rests, 
        duration=durations_rests,
        description=descriptions_rest,
        orig_time=orig_times)
    
    raw_crop.set_annotations(annot_from_events_rests + rest_annots)

    ##  save rest period annotations & markers
    # if not os.path.exists(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif")):
    
    raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"), overwrite = True)

    return raw_crop

def annotate_bad_spans(sub, raw_crop):

    if os.path.exists(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif")):
    
        # 1. read annotations of rest periods
        print("\nREADING ANNOTATIONS FOR REST PERIODS FROM REST_ANNOTATIONS FILE")
        rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"))

        # 2. read annotations of bad segments 
        print("\nREADING ANNOTATIONS FOR BAD SEGMENTS FROM BAD_ANNOTATIONS FILE")
        bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        # set annotations 
        raw_crop.set_annotations(bad_annotations + rest_annotations, emit_warning=False)

        # tell user that both rest annotations & bad segment annotations were set & ask if they want to review it
        resp = input("Would you like to review / add more bad annotations  ? (y/n)")
        if resp.lower() != "y":
            return 
        else:  
            print('PLOTTING SIGNAL IN ANNOTATION MODE')
            raw_crop.plot(block=True)

            print("\nSAVING NEWLY ADDED ANNOTATIONS OF BAD SEGMENTS TO FILE \n")
            raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"), overwrite=True)
        
            # print info object (and also check if bad channels have been marked)
            print('BAD CHANNELS', raw_crop.info['bads'])
    else:
        rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-rest_annotations.fif"))
        raw_crop.set_annotations(rest_annotations)

        raw_crop.plot(block=True)

        print("\nSAVING NEWLY ADDED ANNOTATIONS OF BAD SEGMENTS TO FILE\n")
        raw_crop.annotations.save(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        print('BAD CHANNELS', raw_crop.info['bads'])

    return raw_crop



def find_eogs(raw_crop, sub):
        
        # load saved annotations 
        bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "raw_annotations", f"{sub}-annotations.fif"))

        # set annotations
        raw_crop.set_annotations(bad_annotations)

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


## TODO: set seed to make it reproducible!!
def fit_ica(sub, raw_crop):

        raw_ica_filtered = raw_crop.copy().filter(l_freq = 1., h_freq=30)
        ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
        ica.fit(raw_ica_filtered, reject_by_annotation=True)

        # Save ICA solution
        ica.save(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'), overwrite=True)

        ## FIXME: why is the plotted signal so incredibly noisy??
        # plot components
        fig_sources = ica.plot_sources(raw_crop, title=f"AFTER: ICs Timecourses for Sub-{sub}", show_scrollbars=False) 
        fig_components = ica.plot_components(title=f"ICs for Sub-{sub}")
        fig_components.savefig(os.path.join(DERIV_DIR, "ica", f'{sub}_ics.png'))



## TODO: fix the ask for components (doesn't work)
def apply_ica(sub, raw_crop):

    # check if already cleaned data exists
    if os.path.isfile(os.path.join(DERIV_DIR, 'raw_cleaned', f'sub-{sub}-raw_cleaned.fif')):
        raise FileExistsError

    else:
        # load subject's ICA solution
        ica = mne.preprocessing.read_ica(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))
        
        if not os.path.isfile(os.path.join(DERIV_DIR, 'ica', f'{sub}_components')):

            # ask for user input to indicate the components (separated by spaces)
            components = [int(x) for x in input("ENTER INDICES OF COMPONENTS TO REMOVE: ").split()]
            
            # FIXME!! this also fails
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

        # ica.apply changes raw object in-place, so let's make a copy first
        reconst_raw = raw_crop.copy()
        ica.apply(reconst_raw)

        print('PLOTTING SIGNAL AFTER BLINK REMOVAL')
        reconst_raw.plot(block=True)

        print("\nSAVING CLEANED DATA\n ")
        reconst_fname = f'sub-{sub}-raw_cleaned.fif'
        reconst_raw.save(os.path.join(RAW_CLEANED, reconst_fname))
    

def run_epochs(sub, events):

    # load subjects ICA solution
    print(f"\nLOADING CLEANED DATA FOR SUB-{sub}\n")
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
    
    print(dropped_percent)


if __name__ == "__main__":
    pass