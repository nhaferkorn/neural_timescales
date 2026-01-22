"""This script pre-processes the raw EEG data."""

# make imports
import os
import sys
import numpy as np
import mne
import fooof 
import pickle

from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys
from timescales_memory.preprocessing import preprocessing, plot_events, add_bad_channels, annotate_rest, annotate_bad_spans, fit_ica, apply_ica

# set system variable
param1 = sys.argv[1]

# run preprocessing
raw, events = preprocessing(sub=param1)

# read annotations of rest periods
rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-rest_annotations.fif"))

# read annotations of start-end
start_end_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-start_end_annotations.fif"))

# read bad annotations
bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-bad_annotations.fif"))

raw.set_annotations(rest_annotations + start_end_annotations + bad_annotations)

print('COUNT', raw.annotations.count())

# read bad channels
# TODO: implement try switch logic to see whether bad channel list is not empty
if os.path.exists(os.path.join(DERIV_DIR, "annotations", f"{param1}_bad_chs")):
    with open(os.path.join(DERIV_DIR, 'annotations', f'{param1}_bad_chs'), 'rb') as fp:
        bad_chs = pickle.load(fp)

    print(bad_chs)

    raw.info['bads'] = bad_chs

print('RAW INFO', raw.info)

# fit ica
print('NOW RUNNING ICA')
fit_ica(sub=param1, raw=raw)

# apply ica
print('NOW FITTING ICA')
apply_ica(sub=param1, raw = raw)

# remove the raw file (to save memory)
print('REMOVING RAW FILE')

os.remove(os.path.join(EEG_DIR, f"{param1}.bdf"))

raw = mne.io.read_raw_fif(os.path.join(DERIV_DIR, 'raw_cleaned', f"sub-{param1}-raw_cleaned.fif"))

print('INFO OF RAW CLEANED', raw.info)



