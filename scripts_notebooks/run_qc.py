import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw, run_epochs, plot_epochs,  compute_psd

# set system variables
param1 = sys.argv[1]

raw = mne.io.read_raw_fif(os.path.join(RAW_CLEANED, f"sub-{param1}-raw_cleaned.fif"))

print('INFO OF RAW CLEANED', raw.info)

print('PLOTTING CLEANED SIGNAL AFTER ICA (BLINK REMOVAL)')

plot_reconst_raw(sub=param1)


# # construct epochs 
# events = mne.find_events(raw, stim_channel = "Status", initial_event=False, shortest_event=1)
# events[:,2] = events[:, 2] - 64512

# print(f"\nNOW RUNNING EPOCHS FOR SUB-{param1}\n")

# # input: cleaned data (post ICA + manual rejection)
# epochs = mne.Epochs(raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (-0.5, 0), reject_by_annotation=True, picks = 'eeg', on_missing="ignore")


# epochs = epochs[['Fixation Onset Enc', 'Fixation Onset Ret']]

# print(epochs)

# # how many epochs were dropped - add titles 
# epochs.drop_bad()

# # save as figure 
# fig_droplog = epochs.plot_drop_log()
# fig_droplog.suptitle(f"Sub-{param1} Drop Log (All Epochs)")

# fig_droplog.savefig(os.path.join(DERIV_DIR, 'epochs', f'Sub-{param1}_droplog.png'))

# # plot evoked objects
# fig_enc = epochs['Fixation Onset Enc'].average().plot_joint()

# fig_enc.suptitle(f"Sub-{param1} Evoked Fixation Encoding")

# fig_enc.savefig(os.path.join(DERIV_DIR, 'evokeds', f'Sub-{param1}_evoked_enc.png'))



# epochs['Fixation Onset Ret'].average().plot_joint()

# fig_enc.suptitle(f"Sub-{param1} Evoked Fixation Retention")

# fig_enc.savefig(os.path.join(DERIV_DIR, 'evokeds', f'Sub-{param1}_evoked_ret.png'))


