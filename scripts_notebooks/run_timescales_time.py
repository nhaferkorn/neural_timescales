"""This script performs timescale analysis in the frequency domain."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 

# import functions from timescales package
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

# make imports from timescales methods
from timescales.conversions import knee_to_tau

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw, run_epochs, compute_psd, timescales_acf_evoked_np, timescales_psd_evoked_np, timescales_acf_single_trials

# set system variables
sub = sys.argv[1]


# load cleaned data for subject
print(f"LOADING CLEANED DATA FOR SUB-{sub}\n")
reconst_fname = f'sub-{sub}-raw_cleaned.fif'
RAW_CLEANED_SUB = os.path.join(RAW_CLEANED, reconst_fname)

reconst_raw = mne.io.read_raw_fif(RAW_CLEANED_SUB, preload=True)

# find events
events = mne.find_events(reconst_raw, stim_channel = "Status", initial_event=False, shortest_event=1)
events[:,2] = events[:, 2] - 64512

# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# crop epochs 
epochs_crop = epochs.copy().crop(tmin=0.2, tmax=1.1)


# check bad channels
print("INFO OBJECT OF EPOCHS CROPPED INFO:", epochs_crop.info["bads"])


# exclude bad channels
chs_cleaned = list(set(epochs_crop.info['ch_names']) - set(epochs_crop.info['bads']))


epochs_crop = epochs_crop.copy().pick_channels(chs_cleaned)
print(epochs_crop.info)

# TODO: compute drop log & save that information
epochs_fix_enc = epochs_crop['Fixation Onset Enc']
epochs_fix_ret = epochs_crop['Fixation Onset Ret']

# # evoked_fix_enc = epochs_fix_enc.average()
# # evoked_fix_ret = epochs_fix_ret.average()


# # # compute grand averages (i.e. average of evoked encoding & retrieval)
# # grand_average = mne.grand_average([evoked_fix_enc, evoked_fix_ret])
# # # plot grand average
# # grand_average.plot(titles=f"{sub} Grand Avg Evoked Both").savefig(os.path.join(DERIV_DIR, "timescales", "plots", f'sub-{sub}_grand_avg.png'))


# ## FIT TIMESCALES ON EVOKED OBJECTS (NO OSC)
# acf_fitted_wo, rsq_fitted_wo = timescales_acf_evoked_np(sub, evoked_fix_enc, osc = False)  # returns numpy array

# # convert to pd.DataFrame
# df_wo = pd.DataFrame(acf_fitted_wo, columns = ["tau", "height", "offset"])

# # add channel names
# df_wo['ch_names'] = ch_names

# # add explained variance per channel
# df_wo['RSQ'] = rsq_fitted_wo
# print(df_wo.head(5))

# # save as csv file
# df_wo.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc_without_osc.csv'), sep = ',', header = True, index = False)

# # ## FIT TIMESCALES ON EVOKED OBJECTs WITH OSCILLATIONS
# # print('NOW FITTING WITH OSCILLATIONS\n')

# # # fit timescales on evoked objects 
# # acf_fitted, rsq_fitted = timescales_acf_evoked_np(sub, evoked_fix_enc, osc = True)

# # # convert to pd.DataFrame
# # df_osc = pd.DataFrame(acf_fitted, columns = ["exp_tau", "osc_tau", "osc_gamma", "osc_freq", "amp_ratio", "height", "offset"])

# # # add channel names
# # df_osc['ch_names'] = ch_names

# # # add explained variance per channel
# # df_osc['RSQ'] = rsq_fitted
# # print(df_osc.head(5))

# # # save as csv file
# # df_osc.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc.csv'), sep = ',', header = True, index = False)


# FIT TIMESCALES ON SINGLE TRIALS (NO OSC)
acf_trials, rsq_trial = timescales_acf_single_trials(sub, epochs = epochs_fix_enc)
print(acf_trials.shape) # (389, 31, 3)
print(rsq_trial.shape)


# add channel names & epochs index



# reshape into 2D array
acf_trials_reshaped = acf_trials.reshape(31 * 389, 3)
print(acf_trials_reshaped.shape)

print(acf_trials_reshaped)


# TODO: ideally I would also add a column with the trial number!


## Alternatively - use concept of 3D pandas dataframe
# Convert the numpy array into a pandas DataFrame - no, this doesn't really
# df_3d = pd.DataFrame(acf_trials.reshape(31, -1), columns=[f'Col{j+1}' for j in range(389 * 3)])
# print(df_3d)

# acf_trials_flattened = acf_trials.flatten()
# print(acf_trials_flattened)


# # convert to pd.DataFrame
df_trials = pd.DataFrame(acf_trials_reshaped, columns = ["tau", "height", "offset"])

# add channel names
df_trials['chs'] = np.tile(epochs_fix_enc.info['ch_names'], acf_trials.shape[0])


# # add explained variance per channel
# df_trials['rsq'] = rsq_trial 

# df_trials['sub'] = sub 


print(df_trials.head(50))

# save as csv file
# df_osc.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_evoked_enc.csv'), sep = ',', header = True, index = False)



# # subsetting epochs object - okay so this does seem to work
# # but I need to find criterion that is empirically motivated!
# epochs_firstthird = epochs_fix_enc_cropped[0:130]
# epochs_secondthird = epochs_fix_enc_cropped[130:260]
# epochs_thirdthird = epochs_fix_enc_cropped[260:]

# print("LENGTH EPOCHS FIRST THIRD", len(epochs_firstthird)) # 130
# print("LENGTH EPOCHS SECOND THIRD", len(epochs_secondthird)) # 130
# print("LENGTH EPOCHS FINAL THIRD", len(epochs_thirdthird)) # 129

# ####### Check if Timescales change across the epochs (but not sure if it makes sense to compute them on very short segments)
# # crop epochs into three segments 
# epochs_seg1 = epochs.copy().crop(tmin=0, tmax=0.3)
# epochs_seg2 = epochs.copy().crop(tmin=0.3, tmax=0.6)
# epochs_seg3 = epochs.copy().crop(tmin=0.6, tmax=1.0)

# # plot the evoked for encoding phase
# epochs_seg1['Fixation Onset Enc'].average().plot(titles=f"{sub} Enc Evoked - First 300ms")
# epochs_seg2['Fixation Onset Enc'].average().plot(titles=f"{sub} Enc Evoked - 300-600ms")
# epochs_seg3['Fixation Onset Enc'].average().plot(titles=f"{sub} Enc Evoked - 600-1000ms")

# epochs_enc_seg1 = epochs_seg1['Fixation Onset Enc'].average()
# epochs_enc_seg2 = epochs_seg2['Fixation Onset Enc'].average()
# epochs_enc_seg3 = epochs_seg3['Fixation Onset Enc'].average()

# list_epochs_segments = [epochs_enc_seg1, epochs_enc_seg2, epochs_enc_seg3]

# for epoch in list_epochs_segments:

#     # set ch_names
#     ch_names = epochs.info['ch_names']

#     ## FIT WITHOUT OSCILLATIONS: fit timescales on evoked objects 
#     acf_fitted_wo, rsq_fitted_wo = timescales_acf_evoked_np(sub, epoch, osc = False)  

#     df_wo = pd.DataFrame(acf_fitted_wo, columns = ["tau", "height", "offset"])

#     # add channel names
#     df_wo['ch_names'] = ch_names

#     # add explained variance per channel
#     df_wo['RSQ'] = rsq_fitted_wo
#     print(df_wo.head(5))

#     # save as csv file
#     df_wo.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'acf_timescales', f'sub-{sub}_acf_params_{epoch}_without_osc.csv'), sep = ',', header = True, index = False)
