"""Converts BDF data to BIDS format."""

import os
import glob

import re

import mne

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids

from src.settings import ROOT_DIR, DATA_DIR, EEG_DIR, BIDS_ROOT, TASK, EVENT_DICT

def data2bids():
    """Provide docstring for data2bids function!
    """
    # check if eeg and behavioral subdirectories exist in data_dir
    for file in os.scandir(DATA_DIR):
        print(f"{file} - {'dir' if file.is_dir() else 'file'}")

    if os.path.exists(EEG_DIR):
        print("EEG directory exists")

    else:
        print("Creating EEG directory")
        os.mkdir(EEG_DIR)

# initialize empty lists
    raws = list()
    events = list()
    subject_ids = list()

    for (root, dirs, files)  in os.walk(EEG_DIR):
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
    if os.path.exists(BIDS_ROOT):
        print("BIDS root directory exists")

    else:
        print("Creating BIDS root directory")
        os.mkdir(BIDS_ROOT)

    # extract subject ID from file names - use re module here
    bids_list = list()

    for subject_id in subject_ids: 
        bids_path = BIDSPath(
        subject=f"{subject_id[-2:]}",
        task=TASK,
        root=BIDS_ROOT,
       )
        bids_list.append(bids_path)

    for raw, event, bids_path in zip(raws, events, bids_list):
        write_raw_bids(
        raw,
        bids_path,
        events=event,
        event_id=EVENT_DICT,
        overwrite=True  
        )

    return raws, events, subject_ids, bids_list

if __name__  == "__main__":
    data2bids()