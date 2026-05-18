"""This script pre-processes the raw EEG data.
It expects the subject ID (e.g. "101") as a command-line argument (sys.argv[1])."""

# make imports
import os
import sys
import numpy as np
import mne
import fooof 
import pickle

# import paths and custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys
from timescales_memory.preprocessing import preprocessing, plot_events, add_bad_channels, annotate_rest, annotate_bad_spans, fit_ica, apply_ica

# set system variable
sub = sys.argv[1]

# run preprocessing
raw, events = preprocessing(sub=sub)

# annotate rest periods
raw = annotate_rest(raw, events=events, sub=sub)

# # annotate bad spans
raw = annotate_bad_spans(sub=sub, raw=raw)

# read annotations of rest periods
rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-rest_annotations.fif"))

# read annotations of start-end
start_end_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-start_end_annotations.fif"))

# read bad annotations
bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{sub}-bad_annotations.fif"))

# set all annotations
raw.set_annotations(rest_annotations + start_end_annotations + bad_annotations)


raw_all = add_bad_channels(sub=sub, raw=raw)

# fit ica
print('NOW RUNNING ICA')
fit_ica(sub=sub, raw=raw)

# apply ica
apply_ica(sub=sub, raw=raw)

