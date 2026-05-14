import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle
import pandas as pd
from datetime import datetime


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw


# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")


# set system variables
sub = sys.argv[1]

# raw = mne.io.read_raw_fif(os.path.join(RAW_CLEANED, f"sub-{sub}-raw_cleaned.fif"))
# info =  raw.info

# print('Before', info['ch_names'])

# # # save info object
# # info_eeg = mne.pick_info(info, mne.pick_types(info, eeg=True))
# # info_eeg.save(os.path.join(DERIV_DIR, 'info', f'sub-{sub}_info.fif'), overwrite=True)

# # find events
# events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
# events[:,2] = events[:, 2] - 64512

# # run epochs 
# print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# # demean epochs
# epochs = mne.Epochs(raw, events = events, event_id = events_of_interest, tmin = 0.2, tmax=1.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# # interpolate bad channels, reset_bads: If True, remove the bads from info.
# epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False) 


# # info_eeg = mne.pick_info(info, mne.pick_types(info, eeg=True))
# epochs_interpolated.info.save(os.path.join(DERIV_DIR, 'info', f'sub-{sub}_info.fif'), overwrite=True)

# # load and check
# info = mne.io.read_info(os.path.join(DERIV_DIR, 'info', f'sub-{sub}_info.fif'))
# print('Updated ch_names', len(info['ch_names']))

# print('PLOTTING CLEANED SIGNAL AFTER ICA (BLINK REMOVAL)')

# # plot_reconst_raw(sub=sub)

# # # construct epochs 
# events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
# events[:,2] = events[:, 2] - 64512

# print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# # input: cleaned data (post ICA + manual rejection)
# epochs = mne.Epochs(raw, events = events, event_id = events_of_interest, tmin = 0, tmax=1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=False)

# epochs_fixation = epochs[['Fixation Onset Enc', 'Fixation Onset Ret']]


# # # how many epochs were dropped - add titles 
# # epochs.drop_bad()

# # # save as figure 
# # fig_droplog = epochs.plot_drop_log()
# # fig_droplog.suptitle(f"Sub-{param1} Drop Log (All Epochs)")

# # fig_droplog.savefig(os.path.join(DERIV_DIR, 'epochs', f'Sub-{param1}_droplog.png'))

# # # plot evoked objects
# fig_enc = epochs_fixation['Fixation Onset Enc'].average().plot_joint()
# fig_enc.suptitle(f"Sub-{sub} Evoked Fixation Encoding")
# fig_enc.savefig(os.path.join(DERIV_DIR, 'evokeds', f'Sub-{sub}_evoked_enc_new_epoching_{date}_01.png'))


# fig_ret = epochs['Fixation Onset Ret'].average().plot_joint()
# fig_ret.suptitle(f"Sub-{sub} Evoked Fixation Retention")
# fig_ret.savefig(os.path.join(DERIV_DIR, 'evokeds', f'Sub-{sub}_evoked_ret_new_epoching_{date}_01.png'))



# print number of bad channels in csv file:
# load pickle file
# bad_chs = pd.read_pickle(
#     os.path.join(DERIV_DIR, "annotations", f"{sub}_bad_chs")
# )

# # inspect contents
# print(bad_chs)

# # if pickle contains a list
# if isinstance(bad_chs, list):

#     # save directly as one-column csv
#     pd.DataFrame(
#         {"bad_channel": bad_chs}
#     ).to_csv(
#         os.path.join(
#             DERIV_DIR,
#             "annotations",
#             f"{sub}_bad_chs.csv"
#         ),
#         index=False
#     )

# # if pickle already contains dataframe
# else:
#     bad_chs.to_csv(
#         os.path.join(
#             DERIV_DIR,
#             "annotations",
#             f"{sub}_bad_chs",
#             f"{sub}_bad_chs.csv"
#         ),
#         index=False
#     )


# load pickle file
bad_chs = pd.read_pickle(
    os.path.join(DERIV_DIR, "annotations", f"{sub}_bad_chs")
)

# inspect contents
print(bad_chs)

# if pickle contains a list
if isinstance(bad_chs, list):

    # create dataframe with counts
    bad_chs_df = (
        pd.Series(bad_chs)
        .value_counts()
        .reset_index()
    )

    # rename columns
    bad_chs_df.columns = ["bad_channel", "count"]

# if pickle already contains dataframe
else:
    bad_chs_df = bad_chs

# save csv
bad_chs_df.to_csv(
    os.path.join(
        DERIV_DIR,
        "annotations",
        f"{sub}_bad_chs.csv"
    ),
    index=False
)