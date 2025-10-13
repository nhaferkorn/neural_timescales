########## Converts EDF data to BIDS format ##################
import os
import mne

from pathlib import Path   

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids

from src.setup import *

# Get the data
print(data_dir)

# initialize as a directory - this is not really necessary
raws = {}
events = {}

for file in eeg_dir.glob('*.bdf'):
    raws[file] = mne.io.read_raw_bdf(file, preload=False)
    
print(raws)

# this actually works
for file in eeg_dir.glob('*.bdf'):
    events[file] = mne.find_events(raws[file], stim_channel="Status", initial_event=False)
    mne.write_events(eeg_dir / f"events._{file.stem}.tsv", events[file])


# for file in os.listdir(data_dir):
#     filename = os.fsdecode(file)
#     if filename.endswith(".bdf"):
#         print(file)
#         mne.io.read_raw_bdf(file, preload=False)
#     else: 
#          continue

# raws = [mne.io.read_raw_bdf(file, preload=False) for file in os.listdir(data_dir)]
# events = mne.find_events(raw, stim_channel="Status", initial_event=False)


# Subtract the marker offset value from all elements in the third column
# for event in range(len(events)):
#       events[event][2] =  events[event][2] - 64512
# print(events)


# # define a task name 
task = "dseeg"

# add control flow to check whether this directory already consists, if not create it
bids_root = data_dir / "data_bids"

if bids_root.exists():
    print("BIDS root directory exists")
    
else:
    print("Creating BIDS root directory")
    bids_root.mkdir()


# let's write the BIDS data
# bids_path = BIDSPath(subject = subject_id, task = task, root=bids_root)

# # check how to add the events file to the BIDS path
# bids_path.update(subject="01", task="dseeg", suffix="events", extension=".tsv"
# )
# write_raw_bids(raw, bids_path, overwrite=True)

# print BIDS directory - nice, this seems to work
# print_dir_tree(bids_root)
