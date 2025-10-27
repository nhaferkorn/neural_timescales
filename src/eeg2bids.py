"""Converts BDF data to BIDS format."""

import os
import glob

import re

import mne

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids


# Set-up directory structure
root_dir = os.path.expanduser('~/OneDrive - Radboud Universiteit/Documents/CNS_Master/Master_Thesis/neural_timescales') # to be replaced with respective directory
data_dir = os.path.normpath(os.path.join(root_dir, 'data'))
eeg_dir = os.path.join(data_dir,  'eeg')
behavior_dir = os.path.join(data_dir,  'behavior')

# Path to BIDS root directory
bids_root = os.path.join(data_dir, 'data_bids')

def data2bids():
    """Provide docstring for data2bids function!
    """
    # check if eeg and behavioral subdirectories exist in data_dir
    for file in os.scandir(data_dir):
        print(f"{file} - {'dir' if file.is_dir() else 'file'}")

    if os.path.exists(eeg_dir):
        print("EEG directory exists")

    else:
        print("Creating EEG directory")
        os.mkdir(eeg_dir)


    task = "dseeg"

    event_dict = dict(zip(['Start Practice Trial',
                        'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 
                        'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target',
                        'Response Natural', 'Response Manmade', 'Response None Enc',
                        'Fixation Onset Enc', 'Cue Onset', 'Rest onset', 'Rest offset', 'End Encoding',
                        'Start Retrieval', 'Retrieval Stimulus Onset Baseline Left', 'Retrieval Stimulus Onset Baseline Right',
                        'Retrieval Stimulus Onset Distraction Left Target', 'Retrieval Stimulus Onset Distraction Right Target',
                        'Retrieval Stimulus Onset Distraction Right Distractor', 'Retrieval Stimulus Onset Distraction Left Distractor',
                        'Retrieval Stimulus Onset New', 'Response Old', 'Response New', 'Response None ON',
                        'Confidence Onset', 'Response Confidence 1', 'Response Confidence 2', 'Response Confidence 3', 'Response Confidence None',
                        'Fixation Onset Ret', 'End Retrieval'],[99,
                        21,22,
                        23,24,
                        33,35,38,
                        40,45,90,91,13,
                        50,51,52,
                        53,54,
                        55,56,
                        57,63,65,68,
                        70,73,75,77,78,
                        80,93,
                        ],
    ))

# but this is definitely not a long term solution - mne python just doesn't really like it when 
    event_dict.update({
    'stimulus_10': 10,
    'stimulus_65536': 65536,
    })


# def eeg2bids():

# initialize empty lists
    raws = list()
    events = list()
    subject_ids = list()

    for (root, dirs, files)  in os.walk(eeg_dir):
        print(files)
        for file in files:
            if file.endswith('.bdf'):
                file_path = os.path.join(root, file)
                print(f"Loading: {file_path}")
                raw = mne.io.read_raw_bdf(file_path, preload=False) # make sure to include overwrite checkpoint
                raws.append(raw)
                # obtain events 
                event = mne.find_events(raw, stim_channel = "Status", initial_event=False)
                event[:,2] = event[:, 2] - 64512
                events.append(event)

                # obtain subject ID
                subject_id = file.split('.')[0]
                print(subject_id)
                subject_ids.append(subject_id)

    print(raws)
    print(events)
    print(subject_ids)

    # set up
    if os.path.exists(bids_root):
        print("BIDS root directory exists")

    else:
        print("Creating BIDS root directory")
        os.mkdir(bids_root)


    # extract subject ID from file names - use re module here
    bids_list = list()

    for subject_id in subject_ids: 
        bids_path = BIDSPath(
        subject=f"{subject_id[-2:]}",
        task=task,
        root=bids_root,
       )
        bids_list.append(bids_path)

    for raw, event, bids_path in zip(raws, events, bids_list):
        write_raw_bids(
        raw,
        bids_path,
        events=event,
        event_id=event_dict,
        overwrite=True  
        )

    return raws, events, subject_ids

if __name__  == "__main__":
    data2bids()