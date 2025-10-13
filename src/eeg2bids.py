########## Converts EDF data to BIDS format ##################

# make relevant imports
import shutil
from pathlib import Path

import mne
import os

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
<<<<<<< HEAD

from src.setup_analysis import *

# Get the data
print(data_dir)

# initialize as a directory - this is not really necessary; but appending it to a list should also work
raws = list()
events = list()

for file in eeg_dir.glob('*.bdf'):
    raw = mne.io.read_raw_bdf(file, preload=False)
    raws.append(raw)

print(raws)

# this actually works
for raw in raws:
    event = mne.find_events(raw, stim_channel="Status", initial_event=False)
    events.append(mne.write_events(eeg_dir / f"events_{raw}.tsv", event))

print(events)
=======
from mne_bids.stats import count_events
>>>>>>> parent of bac3d5b (reading multiple bdf files & event files)


# Get the data 
data_directory = "C:/Users/neleh/OneDrive - Radboud Universiteit/Documents/CNS_Master/Master_Thesis/neural_timescales/data"
# data_dir = Path(mne.get_config("C:/Users/neleh/OneDrive - Radboud Universiteit/Documents/CNS_Master/Master_Thesis/neural_timescales/data"))
raw = mne.io.read_raw_bdf(os.path.join(data_directory,'101.bdf'), preload=False)

print(raw)
events = mne.find_events(raw, stim_channel="Status", initial_event=False)

#write events to tsv file (I need to adjust path)
mne.write_events("events.tsv", events)
# write_raw_bids() tries to extract as much meta data as possible from the raw data and then formats it in a BIDS compatible way
subject_id = "01"

# define a tasl name and a directory where to save the data to
task = "dseeg"

## this is not good practice - I need to change this!
## add control flow statement - and check whether this directory already consists, if not create it
bids_root = "C:\\Users\\neleh\\OneDrive - Radboud Universiteit\\Documents\\CNS_Master\\Master_Thesis\\neural_timescales\\data\\data_bids"

# extract subject ID from file names

# subject_id = list(range(1,61))
subject_ids = [1, 2, 3]


# for subject_id in subject_ids:









# let's write the BIDS data
bids_path = BIDSPath(subject = subject_id, task = task, root=bids_root)

# check how to add the events file to the BIDS path
bids_path.update(subject="01", task="dseeg", suffix="events", extension=".tsv"
)
write_raw_bids(raw, bids_path, overwrite=True)

# and print fresh BIDS directory - nice, this seems to work
print_dir_tree(bids_root)

# get an overview of the events on the whole dataset - this doesn't work, because events.tsv file is not created automatically from bdf file
# counts = count_events(bids_root)
# counts