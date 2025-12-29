"""This script pre-processes the raw EEG data."""

import os
import sys
import numpy as np
import mne
import fooof 
import pickle

# make imports 
from timescales.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, EVENT_DICT_CLEAN, DERIV_DIR, RAW_CLEANED, keys
from timescales.preprocessing import preprocessing, annotate_rest, annotate_bad_spans, fit_ica, apply_ica

# set system variables
param1 = sys.argv[1]


# run preprocessing
raw, events = preprocessing(sub=param1)


## FIXME: crosstalk between these two functions doesn't yet work properly
# annotate rest periods
raw_crop = annotate_rest(raw, events = events, sub = param1)

# annotate bad spans
annotate_bad_spans(sub = param1, raw_crop=raw_crop)

# apply ica
# apply_ica(sub = param1, raw_crop = raw_crop)