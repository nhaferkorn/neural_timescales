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
from neurodsp.spectral import compute_spectrum, trim_spectrum

# make imports from timescales methods
from timescales.conversions import knee_to_tau

# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED
from timescales_memory.analyses import timescales_acf_evoked,  timescales_psd_evoked

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

# define keys and events 
keys = ['Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target', 
        'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


event_dist = {k: EVENT_DICT[k] for k in keys}


# run epochs 
print(f"\nNOW RUNNING EPOCHS FOR SUB-{sub}\n")

# demean epochs
epochs = mne.Epochs(reconst_raw, events = events, event_id = event_dist, tmin = 1.2, tmax=2.1, baseline = (None, None), reject_by_annotation=True, picks = 'eeg', on_missing="ignore", preload=True)


# interpolate bad channels
epochs_interpolated = epochs.copy().interpolate_bads(reset_bads=True)


# subselect epochs
epochs_high = epochs_interpolated['Encoding Stimulus Onset Distraction Right Target', 'Encoding Stimulus Onset Distraction Left Target']
epochs_low = epochs_interpolated['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right']


# construct Evoked objects
evoked_high = epochs_high.average()
evoked_low = epochs_low.average()


# RUN FOOOF BEFORE COMPUTING TIMESCALES
# 1. calculate power spectrum across epoched data
psd = evoked_high.compute_psd(method = "welch", fmin=1, fmax=30)

spectra, freqs = psd.get_data(return_freqs=True)

## 2. fit fooof
# Initialize a FOOOFGroup object
fg = fooof.FOOOFGroup(peak_width_limits=[2, 12], 
                peak_threshold=2.5, max_n_peaks=4, verbose=False, aperiodic_mode = 'knee')

# Define the frequency range to fit
freq_range = [1, 30]

# Fit the power spectrum model across all channels
fg.fit(freqs, spectra, freq_range)

# plot for multiple indices
indices = range(1,30)

chs = evoked_high.info['ch_names']
print('# of chs: ', len(chs))

# FIXME: 1 channel is being dropped
# not sure if the order of channels is preserved when fitting psd in fooof
for idx, ch in enumerate(chs):
    fm = fg.get_fooof(ind=idx, regenerate=True)
    fm.plot(file_path = os.path.join(DERIV_DIR, 'timescales', 'psd_timescales', 'evoked', 'fooof'), save_fig=True, file_name = f'sub-{sub}_fooof_{ch}', plt_log = True)


# Print out results
fg.print_results()


# Extract knee frequencies
knee = fg.get_params('aperiodic_params', 'knee')
print('These are the knee frequencies', knee)

# Convert into tau parameters
tau_from_knee = knee_to_tau(knee)

# save important psd_timescale params in csv file
df_fooof_params = pd.DataFrame({
    'knee': knee,
    'tau_from_knee (s)': tau_from_knee
})


df_fooof_params['tau_ms'] = df_fooof_params['tau_from_knee (s)'] * 1000

df_fooof_params['rsq'] = fg.get_params('r_squared')

# mean rsq
df_fooof_params['rsq_avg'] = df_fooof_params['rsq'].mean()

df_fooof_params['sub'] = sub

print(df_fooof_params.head())

# save psd_timescale parameters as csv
df_fooof_params.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'psd_timescales', f'sub-{sub}_fooof_params_evoked_high.csv'), sep = ',', header = True, index = False)



## Estimate timescales from psd
# High vs. Low Distraction Split
evoked_list = [evoked_high, evoked_low]
evoked_names = ['high', 'low']

# loop over epoch objects and names
for name, evoked in zip(evoked_names, evoked_list):
        
        psd_array, rsq_psd, tau_psd  = timescales_psd_evoked(sub=sub, evoked=evoked)

        # print their shape
        print(psd_array.shape)
        print(rsq_psd.shape)
        print(tau_psd.shape)

        # set channel index to match channel order of info object
        chs_idx = evoked.info['ch_names']

        # # convert to pd.DataFrame
        df = pd.DataFrame(psd_array, columns = [f"tau_{name}", f"height_{name}", f"offset_{name}"])

        df["chs"] = chs_idx
        df["sub"] = sub
        df[f'rsq_{name}'] = rsq_psd
        df[f'tau_{name}'] = tau_psd

        # # save as csv file
        # df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'timescales', 'psd_timescales', 'evoked', 'distraction', f'sub-{sub}_acf_params_evoked_{name}.csv'), sep = ',', header = True, index = False)


