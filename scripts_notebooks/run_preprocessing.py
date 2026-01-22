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

# annotate rest periods
raw = annotate_rest(raw, events = events, sub = param1)

# # annotate bad spans
raw = annotate_bad_spans(sub = param1, raw=raw)


# read annotations of rest periods
rest_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-rest_annotations.fif"))

# read annotations of start-end
start_end_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-start_end_annotations.fif"))

# read bad annotations
bad_annotations = mne.read_annotations(os.path.join(DERIV_DIR, "annotations", f"{param1}-bad_annotations.fif"))

raw.set_annotations(rest_annotations + start_end_annotations + bad_annotations)

# print('REST ANNOT COUNT ', rest_annotations.count())
# print('START_END ANNOT COUNT ', start_end_annotations.count())
# print('BAD ANNOT COUNT ', bad_annotations.count())

# print(raw.annotations)
print('COUNT', raw.annotations.count())


raw_all = add_bad_channels(sub=param1, raw=raw)
# fit ica
# print('NOW RUNNING ICA')
# fit_ica(sub=param1, raw=raw)

# apply_ica(sub=param1, raw = raw)

## Save some of the intermediate ouputs to file to create subjects report
# include: # picture with components, % of bad annotated segments