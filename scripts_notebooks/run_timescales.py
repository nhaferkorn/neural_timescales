"""This script performs timescale analysis."""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import fooof 
import pickle

# import functions from timescales package
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

from timescales.sim import sim_ar
from timescales.fit import ARPSD, PSD, ACF
from timescales.autoreg import compute_ar_spectrum
from timescales.plts import set_default_rc 

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.analyses import plot_reconst_raw, run_epochs, compute_psd, timescales_enc_ret, timescales_acf_evoked, timescales_acf_evoked_np, timescales_psd_evoked_np

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

# demean - entire epoch!
epochs = mne.Epochs(reconst_raw, events = events, event_id = events_of_interest, tmin = -1.5, tmax=1.5, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)

# crop epochs 
epochs_crop = epochs.copy().crop(tmin=0, tmax=1)

print("INFO OBJECT OF EPOCHS CROPPED INFO:", epochs_crop.info)

# TODO: compute drop log & save that information
epochs_fix_enc = epochs_crop['Fixation Onset Enc']
epochs_fix_ret = epochs_crop['Fixation Onset Ret']

evoked_fix_enc = epochs_fix_enc.average()
evoked_fix_ret = epochs_fix_ret.average()

# plot the epochs to identify event-locked increases in power 
# evoked_fix_enc.plot(titles=f"{sub} Evoked - Encoding (0s-1s)").savefig(os.path.join(DERIV_DIR, "evokeds", f'sub-{sub}_encoding.png'))

## plots show that there are power deflections that are time-locked to around 200ms, which means I need to crop the epochs even more
epochs_fix_enc_cropped = epochs_fix_enc.crop(tmin=0.2, tmax=1)
evoked_fix_enc_cropped = epochs_fix_enc.average()

# compute power and frequencies
power, freqs = compute_psd(sub=sub, epochs = epochs_crop, events_of_interest = 'Fixation Onset Enc')

print('POWER', power)
print('FREQUENCY', freqs)

print()

print('SHAPE OF POWER', power.shape, '\n') # (31, 28); probably cause there's one bad channel!

print('SHAPE OF FREQS', freqs.shape, '\n') # (28,)

## FIXME: there is still something wrong with the function, the results don't really make sense (too similar across channels)
# try running timescale analysis in frequency domain
psd_fitted, rsq_fitted, tau_fitted = timescales_psd_evoked_np(sub=sub, evoked = evoked_fix_enc_cropped, freqs = freqs, power = power)

# convert to pd.DataFrame
df_psd = pd.DataFrame(psd_fitted, columns = ["offset", "knee_freq", "exp", "const"])

# add channel names
ch_names = epochs.info['ch_names']
df_psd['ch_names'] = ch_names

# add tau column
df_psd['tau'] = tau_fitted

# add explained variance per channel
df_psd['RSQ'] = rsq_fitted

print(df_psd.head(10))

# now I also need to convert the knee frequency into tau



# # and plot again
# evoked_fix_enc_cropped.plot(titles=f"{sub} Evoked - Encoding (0.2s-1s)").savefig(os.path.join(DERIV_DIR, "evokeds", f'sub-{sub}_encoding_cropped.png'))

# # set ch_names
# ch_names = epochs.info['ch_names']

# ## FIT WITHOUT OSCILLATIONS: fit timescales on evoked objects 
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

# # ## FIT WITH OSCILLATIONS
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

# Fit timescales for batches of epochs (progression of the task!)
## first figure out how I loop through epochs object and subselect them; I guess they are intrinsically ordered

# print("LENGTH OF EPOCHS", len(epochs_fix_enc_cropped))

# # for epoch in range(10):
# #     print(epochs_fix_enc_cropped[epoch])


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
