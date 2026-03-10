"""This script pre-processes the raw EEG data."""

# make imports
import os
import sys
import numpy as np
import mne
import pickle

# import variables and paths
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys

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
    """Annotates blocks & rest periods."""

    # events dict for start and end of task
    mapping_start_end = {10:"Start Encoding", 93:"End Retrieval"}

    # convert events array into annotations
    annotations_start_end = mne.annotations_from_events(events, sfreq = raw.info["sfreq"], event_desc = mapping_start_end)

    # print for sanity check
    print('ANNOTATIONS START_END ', annotations_start_end)

    # should result 2 annotations for most subjects
    assert len(annotations_start_end) == 2, f'Not the right dimensions - expecting two markers but got {len(annotations_start_end)}'
    
    ## annotate segments before start of encoding & after the end of retrieval
    print(f"Start Encoding", annotations_start_end[0]["onset"])
    print(f"End Retrieval", annotations_start_end[1]["onset"])

    start_enc = annotations_start_end[0]["onset"]
    # annotations_start_end[annotations_start_end.description == "Start Encoding"].onset[0]

    end_ret = annotations_start_end[1]["onset"]
    # annotations_start_end[annotations_start_end.description == "End Retrieval"].onset[0]

    print('NEW START ENCODING', start_enc)
    print('NEW END ENCODING', end_ret)

    onsets = [raw.times[0], end_ret]
    durations = [
    start_enc,
    raw.times[-1] - end_ret
    ]

    start_end_annots = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=["BAD_PRE_EXP", "BAD_POST_EXP"],
    orig_time=raw.info['meas_date']
)
    
    # save start_end annotations
    raw.set_annotations(start_end_annots)
    print('LENGTH ONLY START-END ANNOTATIONS', len(raw.annotations))

    raw.annotations.save(os.path.join(DERIV_DIR, "annotations", f"{sub}-start_end_annotations.fif"), overwrite = True)

    # plot start_end annotations
    print('NOW PLOTTING START_END ANNOTATIONS')
    raw.plot(block=True)

    ## Annotate rest periods 
    mapping_rests = {90:"Rest onset", 91:"Rest offset"}

    annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw.info["sfreq"], orig_time=raw.info['meas_date'])
    
    raw.set_annotations(annot_from_events_rests)

    print("\nONSETS OF ANNOTATIONS FOR REST PERIODS: ", raw.annotations.onset)

    # set rest periods
    onsets_rests = []
    durations_rests = []

    for i in range(0, len(raw.annotations.onset)-1, 2):
        # set onset of rest
        onsets_rest = raw.annotations.onset[i]
        onsets_rests.append(onsets_rest)

        # set duration of rest
        durations_rest = raw.annotations.onset[i+1] - raw.annotations.onset[i]
        durations_rests.append(durations_rest)

    descriptions_rest = ["BAD_Rest_Period"] * len(onsets_rests)

    rest_annots = mne.Annotations(
        onset=onsets_rests, 
        duration=durations_rests,
        description=descriptions_rest, 
        orig_time=raw.info['meas_date'])
    
    print('# REST BLOCKS', len(rest_annots))

    raw.set_annotations(annot_from_events_rests + rest_annots)
    
    raw.annotations.save(os.path.join(DERIV_DIR, "annotations", f"{sub}-rest_annotations.fif"), overwrite = True)

    print('PLOTTING ONLY REST ANNOTATIONS')
    raw.plot(block=True)

    return raw


def annotate_bad_spans(sub, raw):

    # read annotations of rest periods
    rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-rest_annotations.fif"))

    # read annotations of start-end
    start_end_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-start_end_annotations.fif"))

    # read annotations of bad spans
    if os.path.exists(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif")):
        bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif"))

    # check whether bad annotations file already exists 
    if os.path.exists(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif")):
        # ask user if they want to add more annotations
        resp = input("Would you like to review / add more bad annotations  ? (y/n)")
        if resp.lower() != "y":
            raw.set_annotations(rest_annotations + start_end_annotations + bad_annotations, emit_warning=False)
            return raw 
        
        else:
            
            # set annotations 
            raw.set_annotations(rest_annotations + start_end_annotations + bad_annotations, emit_warning=False)

            # plot in interactive mode
            raw.plot(block=True)

            # save newly added annotations to file
            interactive_annot = raw.annotations

            bad_only = interactive_annot[
            [d.startswith("BAD_SPAN") for d in interactive_annot.description]]

            bad_only.save(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif"), overwrite=True)

            return raw
  
    else:  

        raw.set_annotations(rest_annotations + start_end_annotations)
   
        print('PLOTTING SIGNAL IN INTERACTIVE ANNOTATION MODE')
        
        raw.plot(block=True)

        print("\nSAVING ANNOTATIONS OF BAD SEGMENTS TO FILE \n")

        interactive_annot = raw.annotations

        bad_only = interactive_annot[
            [d.startswith("BAD_SPAN") for d in interactive_annot.description]]

        bad_only.save(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif"), overwrite=True)

        return raw


def add_bad_channels(sub, raw):
    # ask user for input which electrodes are bad and save those!
    print(raw.info['bads'])

    bad_chs = input('Indicate which channels to mark as bad: ')

    bad_chs = [ch.strip() for ch in bad_chs.split(',')]

    
    # TODO: save bad channels in raw info object & also as list for each subject
    with open(os.path.join(DERIV_DIR, 'annotations', f'{sub}_bad_chs'), 'wb') as fp:
        pickle.dump(bad_chs, fp)

    raw.info['bads'] = bad_chs

    print(raw.info['bads'])

    # check type
    print('TYPE BAD ', type(raw.info['bads']))

    return raw 


def fit_ica(sub, raw):
    
    # check whether ICA solution already exists
    if os.path.isfile(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif')):
        raise FileExistsError

    else:
        raw_ica_filtered = raw.copy().filter(l_freq = 1., h_freq=30)

        # sanity check
        print('INFO OBJECT OF RAW FILTERED PRIOR TO ICA\n')
        print(raw_ica_filtered.info)

        ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=95)
        ica.fit(raw_ica_filtered, reject_by_annotation=True)

        # Save ICA solution
        ica.save(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'), overwrite=True)
   
        # Plot components
        fig_sources = ica.plot_sources(raw_ica_filtered, title=f"AFTER: ICs Timecourses for Sub-{sub}", show_scrollbars=False) 
        fig_components = ica.plot_components(title=f"ICs for Sub-{sub}")
        fig_components.savefig(os.path.join(DERIV_DIR, "ica", f'{sub}_ics.png'))


def apply_ica(sub, raw):

    # check if already cleaned data exists
    if os.path.isfile(os.path.join(DERIV_DIR, 'raw_cleaned', f'sub-{sub}-raw_cleaned.fif')):
        raise FileExistsError

    else:
        # load subject's ICA solution
        ica = mne.preprocessing.read_ica(os.path.join(DERIV_DIR, 'ica', f'{sub}' + '-ica.fif'))
        
        if not os.path.isfile(os.path.join(DERIV_DIR, 'ica', f'{sub}_components')):

            # ask for user input to indicate the components (separated by spaces)
            components = [int(x) for x in input("ENTER INDICES OF COMPONENTS TO REMOVE: ").split()]
            
            # FIXME!! this fails
            # save subject solution as pickle object
            with open(os.path.join(DERIV_DIR, 'ica', f'{sub}_components'), 'wb') as fp:
                pickle.dump(components, fp)

        else:
            print(f"\nCOMPONENTS FOR SUB-{sub} EXIST & ARE BEING LOADED\n")

            # load subject solution
            with open(os.path.join(DERIV_DIR, 'ica', f'{sub}_components'), 'rb') as fp:
                components_loaded = pickle.load(fp)


        with open(os.path.join(DERIV_DIR, 'ica', f'{sub}_components'), 'rb') as fp:
            components_loaded = pickle.load(fp)
        print("\nLOADED THE FOLLOWING COMPONENTS:", components_loaded, "\n")
            
        # select which components to exclude
        ica.exclude = components_loaded

        # ica.apply changes raw object in-place, so let's make a copy first
        ## FIXME: look into if I should apply ICA to the raw_ica_filtered data or the raw data directly!!

        ## MNE DOCUMENTATION: However, because filtering is a linear operation, the ICA solution found from the filtered signal 
        # can be applied to the unfiltered signal (see [2] for more information), so we’ll keep a copy of 
        # the unfiltered Raw object around so we can apply the ICA solution to it later.
        reconst_raw = raw.copy()
        ica.apply(reconst_raw)

        print('PLOTTING SIGNAL AFTER BLINK REMOVAL')
        reconst_raw.plot(block=True)

        print("\nSAVING CLEANED DATA\n ")
        reconst_fname = f'sub-{sub}-raw_cleaned.fif'

        # think about adding a date to the file name
        reconst_raw.save(os.path.join(RAW_CLEANED, reconst_fname))
    
if __name__ == "__main__":
    pass