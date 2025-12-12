"""This script analyses the reaction time data and computes memory performance metrics.
"""

# make imports
import sys
import os
import scipy 
import numpy as np
import pandas as pd
from pprint import pprint

sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, BEHAV_DIR, DERIV_DIR


# define helper functions
def read_behav_data(sub, phase):
    """Reads the .xlsx files for each subject as pandas data frame object.

    Args:
        sub (_type_): _description_
        phase (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    if phase == 'enc':
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Enc_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Enc_trialtype', 'Enc_response', 'Enc_RT'])
    elif phase == "ret":
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Ret_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Ret_trialtype', 'Ret_response', 'Ret_RT', 'Conf_response', "Conf_RT"])
    else: 
        raise ValueError("This is not a valid experimental phase, please specify either 'enc' or 'ret'")
    


def process_enc_data(sub, data):
    """

    Args:
        sub (_int_): subject ID 
        data (_type_): raw pandas dataframe

    Returns:
        _type_: processed pandas dataframe
    """

    # replace 0 values with NAN
    data = data.replace(0, np.nan)

    # use value counts method 
    print('VALUES IN:', data['Enc_trialtype'].value_counts())
    print('VALUES IN:', data['Enc_response'].value_counts())

    # add ground truth label of pictures 
    data['GroundTruth_Left'] = data['Pic_left'].apply(
    lambda x: 'manmade' if x[0].isupper() else 'nature')

    data['GroundTruth_Right'] = data['Pic_right'].apply(
    lambda x: 'manmade' if x[0].isupper() else 'nature')

    
    # TODO: double check if my implementation is correct!!!
    # add column for correct encoding (but only of targets)
    conditions_correct = [

    # correct semantic classification
    (data['GroundTruth_Left'] == 'nature') & ((data['Enc_trialtype'] ==  1) | (data['Enc_trialtype'] == 3)) & (data['Enc_response'] == 37),  # 1 & 3 indicate that targets were presented left, Enc_response: 37 indicates nature pictures

    (data['GroundTruth_Right'] == 'nature') & ((data['Enc_trialtype'] ==  2) | (data['Enc_trialtype'] == 4)) & (data['Enc_response'] == 37)

    ]

    values_correct = ['correct', 'correct']

    data['Enc_accuracy'] = np.select(conditions_correct, values_correct,  default='incorrect') # default describes all of the values that are not true


    # drop NA values
    return data.dropna()



def process_ret_data(sub, data):
    print(data.info())

    data['Ret_RT'] = data['Ret_RT'].replace(0, np.nan)

    # Add column that encodes hits & misses (based on the trial type and the subjects respose)
    conditions = [

    # Hits 
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  1) | (data['Ret_trialtype'] == 2)), # hit low distraction target

    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  3) | (data['Ret_trialtype'] == 4)), # hit high distraction target

    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  30) | (data['Ret_trialtype'] == 40)),  # hit distractor


    # Misses
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  1) | (data['Ret_trialtype'] == 2)),  # miss low distraction target

    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  3) | (data['Ret_trialtype'] == 4)), # miss high distraction targets

    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  30) | (data['Ret_trialtype'] == 40)),  # miss distractor 

    # False Alarms
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  91) | (data['Ret_trialtype'] == 92)), 

    # True Negatives. aka pictures judged as new, which were actually new
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  91) | (data['Ret_trialtype'] == 92))

    ]

    # create list of values to assign to each condition
    values = ['hit_low_dist_tar', 'hit_high_dist_tar', 'hit_distractor', 'miss_low_dist_tar', 'miss_high_dist_tar', 'miss_distractor', 'false_alarms', 'correct_new']

    # create a new column and use np.select
    data['hit_category'] = np.select(conditions, values,  default=None) # default describes all of the values that are not true

    ## Create column for confidence judgements
    conditions_confidence = [

        ((data['Ret_response'] == 37) & (data['Conf_response'] ==  37)), # not confident old

        ((data['Ret_response'] == 37) & (data['Conf_response'] ==  40)), # medium confident old

        ((data['Ret_response'] == 37) & (data['Conf_response'] ==  39)), # very confident old

        ((data['Ret_response'] == 39) & (data['Conf_response'] ==  37)),  # not confident new

        ((data['Ret_response'] == 37) & (data['Conf_response'] ==  40)),  # medium confident new

        ((data['Ret_response'] == 37) & (data['Conf_response'] ==  39))  # very confident new
    ]

    values_confidence = ['not_conf_old', 'med_conf_old', 'high_conf_old', 'not_conf_new', 'med_conf_new', 'high_conf_new']

    data['confidence_category'] = np.select(conditions_confidence, values_confidence, default = None) # TODO: check whether None is treates as NaN value

    # dropn NaN values
    data = data.dropna()

    # drop unnecessary columns
    data = data.drop(columns = ['Gender',  'Edu', 'TrialNr', 'Pic_left', 'Pic_right'])

    print(data.head(5))
 
    return data


def calculate_hitrate(data, trial_wise = False):

    """Calculates hitrate, by taking the ratio of number 
    of correct responses and number of total responses

    Args:
        data (pd.Dataframe): a pandas data frame of the retrieval data that contains column of hits
    """

    # this seems to work - but test it
    if trial_wise == True:

        #  hits for low distraction targets
        hits_low_dist = data[data['hit_category'].str.contains('hit_low_dist_tar')]
        
        # hits for high distraction targets
        hits_high_dist = data[data['hit_category'].str.contains('hit_high_dist_tar')]

        # hits for distractors
        hits_distractor = data[data['hit_category'].str.contains('hit_distractor')]

        return (len(hits_low_dist) / len(data)), (len(hits_high_dist) / len(data)), (len(hits_distractor) / len(data))
    
    else:

        hits = data[data['hit_category'].str.contains('hit')]

        # number of observations
        print('# OF HITS:', len(hits))
        print('# OF OBSERVATIONS', len(data))

        # return a ratio, aka hitrate 
        return (len(hits) / len(data))



def calculate_fa_rate(data):
    # TODO: Implement some tests to check if this actually works and matches Syanah's implementation
    """Calculate false alarm rate as: False Positives / (False Positives + True Negatives).

    Args:
        data (pd.Dataframe): _description_
    """

    # define false positives
    false_positives = data[data['hit_category'].str.contains('false_alarms')]

    # define true negatives 
    true_negatives = false_positives = data[data['hit_category'].str.contains('correct_new')]

    # return false alarm rate
    return len(false_positives) / (len(false_positives) + len(true_negatives))

# FIXME: okay, so this won't work since I am only providing a single argument
def calculate_d_prime(hitrate, fa_rate):
    
    # standardize hitrate & false alarm rate
    
    z_score_hitrate = scipy.stats.zscore(hitrate)
    z_score_farate = scipy.stats.zscore(fa_rate)

    print(z_score_hitrate)
    print(z_score_farate)

    return (z_score_hitrate - z_score_farate)


def classify_age(data):

    if (data['Age'] > 40).any():
        return 'old'
    else:
        return 'young'


def create_behavioral_summary(data):
    # input pre-processed data
    # output summary information to txt file or excel file 
    pass



def main():

    # set system variables
    param1 = sys.argv[1]

    # print some data for the encoding phase
    data_encoding = read_behav_data(sub = param1, phase = 'enc')
    print(type(data_encoding))  # returns Dataframe object
    print("BEHAVIORAL DATA OF ENCODING PHASE:", data_encoding.head(5))

    # # print behavioral data from retrieval phase
    data_retrieval = read_behav_data(sub = param1, phase = 'ret')
    print("BEHAVIORAL DATA FROM RETRIEVAL:", data_retrieval.head(5))

    data_enc = process_enc_data(sub = param1, data=data_encoding)
    data_ret = process_ret_data(sub = param1, data=read_behav_data(phase='ret', sub=param1))

    ## check whether dropping NAN values has worked
    assert data_enc['Enc_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"
    assert data_ret['Ret_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"

    hitrate_low, hitrate_high, hit_rate_dist = calculate_hitrate(data_ret, trial_wise=True)
    farate = calculate_fa_rate(data_ret)

    print('THIS IS THE HITRATE:', hitrate_low, hitrate_high, hit_rate_dist)

    print("THIS IS THE FALSE ALARM RATE:", farate)

    # d_prime_low = calculate_d_prime(hitrate_low, farate)
    age = classify_age(data_enc)

    print('AGE CATEGORY:', age)



    

if __name__ == "__main__":
    main()
