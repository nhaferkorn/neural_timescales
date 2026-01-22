"""This script analyses the reaction time data and computes memory performance metrics.
"""

# make imports
import sys
import os
import scipy 
import numpy as np
import pandas as pd

# import variables and paths
from timescales_memory.settings import PROJECT_DIR, BEHAV_DIR, DERIV_DIR


def read_behav_data(sub, phase):
    """Reads the .xlsx files for each subject as pd DataFrame.
    """

    if phase == 'enc':
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Enc_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Enc_trialtype', 'Enc_response', 'Enc_RT'])
    elif phase == "ret":
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Ret_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Ret_trialtype', 'Ret_response', 'Ret_RT', 'Conf_response', "Conf_RT"])
    else: 
        raise ValueError("This is not a valid experimental phase, please specify either 'enc' or 'ret'")
    

def process_enc_data(sub, data):
    """Excludes NaN values, and adds columns for correct encoding."""

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
    

    # add column to indicate whether target location of trial (i.e. left or right)
    data['Target_Loc'] = data['Enc_trialtype'].apply(
        lambda x: 'left' if (x == 1 or x == 3) else 'right'
    )

    # add column for correct encoding
    conditions_correct = [

    # correct nature classification
    (data['GroundTruth_Left'] == 'nature') & ((data['Enc_trialtype'] ==  1) | (data['Enc_trialtype'] == 3)) & (data['Enc_response'] == 37),  # 1 & 3 indicate that targets were presented left, Enc_response: 37 indicates nature pictures

    (data['GroundTruth_Right'] == 'nature') & ((data['Enc_trialtype'] ==  2) | (data['Enc_trialtype'] == 4))& (data['Enc_response'] == 37),
    
    # correct manmade classification
    (data['GroundTruth_Left'] == 'manmade') & ((data['Enc_trialtype'] ==  1) | (data['Enc_trialtype'] == 3)) & (data['Enc_response'] == 39),

    (data['GroundTruth_Right'] == 'manmade') & ((data['Enc_trialtype'] ==  2) | (data['Enc_trialtype'] == 4)) & (data['Enc_response'] == 39)

    ]

    values_correct = ['correct', 'correct', 'correct', 'correct']

    data['Enc_accuracy'] = np.select(conditions_correct, values_correct,  default='incorrect') 

    # drop NA values
    data = data.dropna()

    data.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral', 'processed', f'sub-{sub}-enc_processed.csv'), sep = ',', header = True, index = False)

    return data 


def process_ret_data(sub, data):

    # encode NaN values
    data['Ret_RT'] = data['Ret_RT'].replace(0, np.nan)


    # Add column that encodes hits & misses (based on the trial type and the subjects respose)
    conditions = [

    # Hits 
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  1) | (data['Ret_trialtype'] == 2)), # hit low distraction target

    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  3) | (data['Ret_trialtype'] == 4)), # hit high distraction target

    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  30) | (data['Ret_trialtype'] == 40)),  # hit distractor

    ## FIXME: I don't get how this can classified as a miss?? Because this is just the participants judgement old/new in combination with 
    ## the target trial type 
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
    # TODO: rename to better names!!
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

    # drop NaN values
    data = data.dropna()

    # drop unnecessary columns
    data = data.drop(columns = ['Gender',  'Edu', 'TrialNr'])

    # save data as new csv file
    data.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral', 'processed', f'sub-{sub}-ret_processed.csv'), sep = ',', header = True, index = False)
 
    return data


def calculate_encoding_accuracy(data):
    """Computes count of correctly encoded pictures during encoding phase.
    """
    count_correct = (data['Enc_accuracy'] == 'correct').sum()
    count_total = data['Enc_accuracy'].count()

    print('# CORRECT', count_correct)
    print('# TOTAL', count_total)

    return count_correct / count_total


# FIXME!  I need to divide by the len(misses) and for the trial wise hitrate by the total number of possible
# observation (e.g. for hitrate_lowdist = hits_low_dist / hits_low_dist + miss_low_dist)
def calculate_hitrate(data, trial_wise = False):

    """Calculate hitrate, by taking the ratio of number 
    of correct responses and number of total responses (or the misses??)
    """
    
    misses = data[data['hit_category'].str.contains('miss')]
        
    if trial_wise:
        
        #  hits & misses for low distraction targets
        hits_low_dist = data[data['hit_category'].str.contains('hit_low_dist_tar')]
        miss_low_dist = data[data['hit_category'].str.contains('miss_low_dist_tar')]
        
        # hits & misses for high distraction targets
        hits_high_dist = data[data['hit_category'].str.contains('hit_high_dist_tar')]
        miss_high_dist = data[data['hit_category'].str.contains('miss_high_dist_tar')]

        # hits & misses for distractors
        hits_distractor = data[data['hit_category'].str.contains('hit_distractor')]
        miss_distractor = data[data['hit_category'].str.contains('miss_distractor')]

        return (len(hits_low_dist) / (len(hits_low_dist) + len(miss_low_dist))), (len(hits_high_dist) / (len(hits_high_dist) + len(miss_high_dist))), (len(hits_distractor) / (len(hits_distractor) + len(miss_distractor)))
    
    else:

        hits = data[data['hit_category'].str.contains('hit')]

        # number of observations
        print('# OF HITS:', len(hits))
        print('# OF MISSES', len(misses))
        print('# OF OBSERVATIONS', len(data))

        return (len(hits) / (len(hits) + len(misses)))



def calculate_fa_rate(data): 
    """Calculate FA Rate: False Positives / (False Positives + True Negatives).
    """
    # define false positives
    false_positives = data[data['hit_category'] == 'false_alarms']

    print('LENGTH FALSE POSITIVES', len(false_positives))

    # define true negatives 
    true_negatives = data[data['hit_category'] == 'correct_new']

    print('LENGTH TRUE NEGATIVES', len(true_negatives)) 

    # return false alarm rate
    return len(false_positives) / (len(false_positives) + len(true_negatives))



# TODO: Check if this is actually correct 
def calculate_d_prime(hitrate, fa_rate):
    
    # standardize hitrate & false alarm rate
    z_score_hitrate = scipy.stats.norm.ppf(hitrate)
    z_score_farate = scipy.stats.norm.ppf(fa_rate)

    return (z_score_hitrate - z_score_farate)


def classify_age(data):
    """Classifies subject as OA or YA."""

    if (data['Age'] > 40).any():
        return 'old'
    else:
        return 'young'


def create_behavioral_summary(sub, data_enc, data_ret):
    """Creates a summary pd.DataFrame of the behavioral data, 
    including: Sub-ID, Age, Encoding Accuracy & Memory Performance (D' metrics).

    Args:
        sub (_type_): _description_
        data (_type_): _description_
    """

    Sub_ID = sub 

    Age = classify_age(data_enc)

    Enc_Accuracy = calculate_encoding_accuracy(data_enc)

    # calculate false alarm rate
    farate = calculate_fa_rate(data_ret)

    # calculate hitrates
    hitrate_global = calculate_hitrate(data_ret)
    hitrate_low, hitrate_high, hitrate_dist = calculate_hitrate(data_ret, trial_wise=True)

    # calculate both global and local memory performance
    D_Prime_Global = calculate_d_prime(hitrate=hitrate_global, fa_rate=farate)
    D_Prime_Low = calculate_d_prime(hitrate=hitrate_low, fa_rate=farate)
    D_Prime_High = calculate_d_prime(hitrate=hitrate_high, fa_rate=farate)
    D_Prime_Dist = calculate_d_prime(hitrate=hitrate_dist, fa_rate=farate)
    D_Prime_Targets_Avg = (D_Prime_High + D_Prime_Low) / 2 # d prime averaged over low and high distractor targets

    df = pd.DataFrame({
    'ID': [Sub_ID],
    'AGE': [Age],
    'ENC_ACCURACY': [Enc_Accuracy],
    'D_PRIME_GLOBAL': [D_Prime_Global],
    'D_PRIME_HighDist': [D_Prime_High],
    'D_PRIME_LowDist': [D_Prime_Low],
    'D_PRIME_Dist': [D_Prime_Dist],
    'D_PRIME_TARGETS': [D_Prime_Targets_Avg]

})
    print(df)

    # save pandas dataframe to file
    df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral','summary_stats', f'sub-{sub}_summary.csv'), sep = ',', header = True, index = False)

    return df


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

    print('\nCREATING BEHAVIORAL SUMMARY\n')
    create_behavioral_summary(sub=param1, data_enc=data_enc, data_ret=data_ret)


if __name__ == "__main__":
    main()
