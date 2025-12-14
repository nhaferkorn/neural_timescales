"""Set up of all experimental variables and Paths."""

import os
import re
import glob

# Set-up directory structure
PROJECT_DIR = '/project/4180000.57/neural_timescales/' # to be replaced with respective directory
DATA_DIR = os.path.join(os.path.join(PROJECT_DIR, 'data'))
EEG_DIR = os.path.join(DATA_DIR,  'eeg')
BEHAV_DIR = os.path.join(DATA_DIR,  'behavioral')
DERIV_DIR = os.path.join(PROJECT_DIR, "derivatives")
RAW_CLEANED = os.path.join(DERIV_DIR, "raw_cleaned") # make sure if this works


# # Path to BIDS root directory
# BIDS_ROOT = os.path.join(DATA_DIR, 'data_bids')

# SUBJ_DIR = os.path.join(BIDS_ROOT, "eeg")

# for subj in glob.glob(os.path.join(BIDS_ROOT, "sub-*")):
#     EEG_BIDS_DIR = os.path.join(subj, "eeg")
#     print(EEG_BIDS_DIR)

# Set up experimental variables
TASK = "dseeg"


DESCRIPTION = ['Misc_Offset', 'Start Practice Trial',
'Start Encoding', 'Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 
'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target',
'Response Natural', 'Response Manmade', 'Response None Enc',
'Fixation Onset Enc', 'Cue Onset', 'Rest onset', 'Rest offset', 'End Encoding',
'Start Retrieval', 'Retrieval Stimulus Onset Baseline Left', 'Retrieval Stimulus Onset Baseline Right',
'Retrieval Stimulus Onset Distraction Left Target', 'Retrieval Stimulus Onset Distraction Right Target',
'Retrieval Stimulus Onset Distraction Right Distractor', 'Retrieval Stimulus Onset Distraction Left Distractor',
'Retrieval Stimulus Onset New', 'Response Old', 'Response New', 'Response None ON',
'Confidence Onset', 'Response Confidence 1', 'Response Confidence 2', 'Response Confidence 3', 'Response Confidence None',
'Fixation Onset Ret', 'End Retrieval',
'Begin Localizer/End Localizer', 'Centre', 'Bottom Right', 'Bottom Left',
'Middle Left', 'Middle Right', 'Bottom Middle',
'2/3 Left', '2/3 Right']


# not sure about the 65536 value - ask Britta
ORIGINAL_MARKER = [65536, 99,
10,21,22,
23,24,
33,35,38,
40,45,90,91,13,
50,51,52,
53,54,
55,56,
57,63,65,68,
70,73,75,77,78,
80,93,
30,1,2,3,
4,5,6,
7,8]


EVENT_DICT = dict(zip(DESCRIPTION, ORIGINAL_MARKER))


EVENT_DICT_CLEAN = dict(zip(['Start Practice Trial',
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


#  create VEOG and HEOG channels
#  heogl (horizontal EOG left): EXG1
#  heogr (horizontal EOG right): EXG2
#  veogt (vertical EOG; top): EXG3
#  veogb (vertical EOG; bottom): EXG4

# keys of all the events of interest 
keys = {'Start Practice Trial',
    'Start Encoding','Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Baseline Right', 'Encoding Stimulus Onset Distraction Left Target', 'Encoding Stimulus Onset Distraction Right Target'
    ,'Fixation Onset Enc', 'Cue Onset', 
    'Response Natural', 'Response Manmade', 'Response None Enc',
    'Fixation Onset Enc', 'Cue Onset', 'Rest onset', 'Rest offset', 'End Encoding',
    'Start Retrieval', 'Retrieval Stimulus Onset Baseline Left', 'Retrieval Stimulus Onset Baseline Right',
    'Retrieval Stimulus Onset Distraction Left Target', 'Retrieval Stimulus Onset Distraction Right Target',
    'Retrieval Stimulus Onset Distraction Right Distractor', 'Retrieval Stimulus Onset Distraction Left Distractor',
    'Retrieval Stimulus Onset New', 'Response Old', 'Response New', 'Response None ON',
    'Confidence Onset', 'Response Confidence 1', 'Response Confidence 2', 'Response Confidence 3', 'Response Confidence None',
    'Fixation Onset Ret', 'End Retrieval'}


events_of_interest = {k: EVENT_DICT[k] for k in keys}