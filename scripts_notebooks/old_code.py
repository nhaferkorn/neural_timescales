"""Code dump for old code that I might want to use again"""

# load files for multiple subjects - code as it is rn is not gonna work

# raws = list()
# subjects = ["01", "02", "03"]

# for file, subject in eeg_dir.glob('*.bdf'):
#         raw_subject = mne.io.read_raw_bdf(file, preload=False)
#         raws.append(raw_subject)

# print(raws)

# Read multiple bdf files; kinda works but not the most elegant solution
# raws = []

# for file in eeg_dir.glob('*.bdf'):
#     print(file)
#     raw = mne.io.read_raw_bdf(file, preload=False)
#     raws.append(raw)

# print(raws)

# %% Regression approach: Identification of Blinks and Ocular artifacts

# epoch the data - wow, this looks truly horrible!! (not sure what I am doing here!!)

# event_id = {'Encoding Stimulus Onset Baseline Left': 21}
# epochs = mne.Epochs(raw_filtered, events, event_id=event_id, preload=True)

# # we'll try to keep a consistent ylim across figures
# plot_kwargs = dict(picks="all", ylim=dict(eeg=(-10, 10), eog=(-5, 15)))

# # plot the evoked for the EEG and the EOG sensors
# fig = epochs.average("all").plot(**plot_kwargs)
# fig.set_size_inches(6, 6)



### Old EEG BIDS conversion script:
"""Converts EDF data to BIDS format."""

import mne

from pathlib import Path

from mne_bids import BIDSPath, print_dir_tree, write_raw_bids

from src.settings import *

# define a task name 
task = "dseeg"

bids_root = data_dir / "data_bids"

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


def eeg2bids():
    # Get the data
    print(data_dir)

    # initialize empty lists
    raws = list()
    events = list()

    for file in eeg_dir.glob('*.bdf'):
        raw = mne.io.read_raw_bdf(file, preload=False)
        raws.append(raw)
        event = mne.find_events(raw, stim_channel = "Status", initial_event=False)
        event[:,2] = event[:, 2] - 64512
        events.append(event)

    print(raws)
    print(events)

# set up correct folder structure
    if bids_root.exists():
        print("BIDS root directory exists")
        
    else:
        print("Creating BIDS root directory")
        bids_root.mkdir()

    # extract subject ID from file names - use re module here
    subject_ids = [1, 2, 3]

    bids_list = list()

    for subject_id in subject_ids: 
        bids_path = BIDSPath(
                subject=f"{subject_id:02}",
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

    return raws, events


if __name__  == "__main__":
    eeg2bids()



    # but this is definitely not a long term solution - mne python just doesn't really like it when 
    # event_dict.update({
    # 'stimulus_10': 10,
    # 'stimulus_65536': 65536,
    # })



#  Power line noise
fig = raw.compute_psd(tmax=np.inf, fmax=250).plot(
    average=True, amplitude=False, picks="data", exclude="bads"
)

#  Compute TFRs and calculate the difference between alpha power for attending left vs. attending right trials;
# so I guess that's pooled over baseline & distraction (not sure if that actually makes sense)

