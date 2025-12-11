"""This script analyses the reaction time data and computes memory performance metrics.

"""

# make imports
import sys
import os
import numpy as np
import pandas as pd
from pprint import pprint

sys.path.append('/project/4180000.57/neural_timescales/src')

# import variables and paths
from settings import PROJECT_DIR, BEHAV_DIR, DERIV_DIR


######################## Define functions here ###################################################################################

def read_behav_data(sub, phase):
    # read the xlsx files for each subject as pandas data frame object 
    if phase == 'enc':
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Enc_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Enc_trialtype', 'Enc_response', 'Enc_RT'])
    elif phase == "ret":
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Ret_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Ret_trialtype', 'Ret_response', 'Ret_RT', 'Conf_response', "Conf_RT"])
    else: 
        raise ValueError("This is not a valid experimental phase, please specify either 'enc' or 'ret'")
    


def process_enc_data(sub, sub_data):
    """

    Should ideally take as input the subject data in format of pandas dataframe & Return Data, where non-responses are excluded
    
    :param sub: Description
    :param sub_data: Description
    """
    # basic descriptive stats for each column
    print(sub_data.describe())

    print(sub_data.query('Enc_RT == 0'))

    # replace 0 values with NAN
    sub_data = sub_data.replace(0, np.nan)

    # check again
    print(sub_data.query('Enc_RT == 0')) # >> yes, this is empty now

    # print again the number of NAN values
    print('COUNT OF NAN VALUES IN DATA: ', sub_data.isna().sum())

    # use value counts method 
    print('VALUES IN:', sub_data['Enc_trialtype'].value_counts())
    print('VALUES IN:', sub_data['Enc_response'].value_counts())

    # TODO: Add a new column classifying whether target picture was manmade or nature image

    ## apply mapping to every element in column of data frame and store these into a new column (not sure if this would works)

    # FIXME: Not sure if I actually need all of this or whether, I only have access to the correct encoded trials

    sub_data['GroundTruth_Left'] = sub_data['Pic_left'].apply(
    lambda x: 'manmade_left' if x[0].isupper() else 'nature_left')

    sub_data['GroundTruth_Right'] = sub_data['Pic_right'].apply(
    lambda x: 'manmade_right' if x[0].isupper() else 'nature_right')

    # drop NA values
    return sub_data.dropna(axis = 0, inplace = False)

def process_ret_data(sub, sub_data):
    # FIXME: change name to data and make sure that the data is loaded!! 
    print(sub_data.info())

    sub_data['Ret_RT'] = sub_data['Ret_RT'].replace(0, np.nan)

    # # print again the number of NAN values
    print('NUMBER OF NAN VALUES IN DATA: ', sub_data.isna().sum())

    # use value counts method 
    print('VALUES IN:', sub_data['Ret_trialtype'].value_counts())

    # drop NA values
    # sub_data.dropna(axis = 0, inplace = False)

    # Add column that encodes hits & misses (based on the trial type and the subjects respose)
    
    conditions = [

    # Hits 
    (sub_data['Ret_response'] == 37) & ((sub_data['Ret_trialtype'] ==  1) | (sub_data['Ret_trialtype'] == 2)), # hit low distraction target

    (sub_data['Ret_response'] == 37) & ((sub_data['Ret_trialtype'] ==  3) | (sub_data['Ret_trialtype'] == 4)), # hit high distraction target

    (sub_data['Ret_response'] == 37) & ((sub_data['Ret_trialtype'] ==  30) | (sub_data['Ret_trialtype'] == 40)),  # hit distractor


    # Misses
    (sub_data['Ret_response'] == 39) & ((sub_data['Ret_trialtype'] ==  1) | (sub_data['Ret_trialtype'] == 2)),  # miss low distraction target

    (sub_data['Ret_response'] == 39) & ((sub_data['Ret_trialtype'] ==  3) | (sub_data['Ret_trialtype'] == 4)), # miss high distraction targets

    (sub_data['Ret_response'] == 39) & ((sub_data['Ret_trialtype'] ==  30) | (sub_data['Ret_trialtype'] == 40)),  # miss distractor 


    # False Alarms
    (sub_data['Ret_response'] == 37) & ((sub_data['Ret_trialtype'] ==  91) | (sub_data['Ret_trialtype'] == 92)), 

    # True Negatives. aka pictures judged as new, which were new
    (sub_data['Ret_response'] == 39) & ((sub_data['Ret_trialtype'] ==  91) | (sub_data['Ret_trialtype'] == 92))

    ]

    # create list of values to assign to each condition
    values = ['hit_tld', 'hit_thd', 'hit_distractor', 'miss_tld', 'miss_thd', 'miss_distractor', 'false_alarms', 'correct_new']

    # create a new column and use np.select
    sub_data['hit_categories'] = np.select(conditions, values,  default='NONE') # then default describes all of the values that are not true

    return sub_data


def calculate_hitrate(data):

    # TODO: Implement  calculation of hitrate for different trialtypes; aka 
    # have separate hit rates for: low distraction taregts, high distraction targets & distractors
    # By setting it as a function argument

    """Calculates the overall hitrate, by taking the ratio of number 
    of correct responses and number of total responses

    Args:
        data (pd.Dataframe): a pandas data frame of the retrieval data that contains column of hits
    """
    hits = data[data['hit_categories'].str.contains('hit')]

    # number of observations
    print('# OF HITS:', len(hits))
    print('# OF OBSERVATIONS', len(data))

    # return a ratio, aka hitrate 
    return (len(hits) / len(data))


def calculate_fa_rate(data):
    # TODO: Implement some tests to check if this actually works and matches Syanah's implementation
    """Calculates false alarm rate as: specify formula
    calculated as False Positives / (False Positives + True Negatives), 
    representing the proportion of actual negatives wrongly identified as positive

    Args:
        data (pd.Dataframe): _description_
    """

    # define false positives
    false_positives = data[data['hit_categories'].str.contains('false_alarms')]

    # define true negatives 
    true_negatives = false_positives = data[data['hit_categories'].str.contains('correct_new')]

    # return false alarm rate
    return len(false_positives) / (len(false_positives) + len(true_negatives))


def main():

    # set system variables
    param1 = sys.argv[1]


    # print some data for the encoding phase
    sub_data_encoding = read_behav_data(sub = param1, phase = 'enc')
    print(type(sub_data_encoding))  # returns Dataframe object
    print("BEHAVIORAL DATA OF ENCODING PHASE:", sub_data_encoding.head(5))


    # # print behavioral data from retrieval phase
    sub_data_retrieval = read_behav_data(sub = param1, phase = 'ret')
    print("BEHAVIORAL DATA FROM RETRIEVAL:", sub_data_retrieval.head(5))

    sub_data_enc = process_enc_data(sub = param1, sub_data=sub_data_encoding)
    sub_data_ret = process_ret_data(sub = param1, sub_data=read_behav_data(phase='ret', sub=param1))

    ## check whether dropping NAN values has worked FIXME!!! - the dropping of NAN values doesn't work properly (also think about whether I would like to have that)
    # assert sub_data_enc['Enc_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"
    # assert sub_data_ret['Ret_RT'].isna().sum() == 0, "Not all NAN values were successfully dropped"

    hitrate = calculate_hitrate(sub_data_ret)
    farate = calculate_fa_rate(sub_data_ret)

    print('THIS IS THE HITRATE:', hitrate)

    print("THIS IS THE FALSE ALARM RATE:", farate)

if __name__ == "__main__":
    main()
