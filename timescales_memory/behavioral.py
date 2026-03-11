"""This script includes functions to process the behavioral data."""

# make imports
import sys
import os
import scipy 
import numpy as np
import pandas as pd

# import paths
from timescales_memory.settings import PROJECT_DIR, BEHAV_DIR, DERIV_DIR

def read_behav_data(sub, phase):
    """Reads the .xlsx files for each subject as pd.DataFrame.
    """
    if phase == 'enc':
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Enc_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Enc_trialtype', 'Enc_response', 'Enc_RT'])
    elif phase == "ret":
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Ret_{sub}.xlsx"),  names = ['ID', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Ret_trialtype', 'Ret_response', 'Ret_RT', 'Conf_response', "Conf_RT"])
    else: 
        raise ValueError("This is not a valid experimental phase, please specify either 'enc' or 'ret'")
  
def process_enc_data(sub, data):
    """Excludes NaN values, and adds columns for ground truth stimulus identity.
    
    Coding of Enc_trialtype column:
    1 = low distraction left target
    2 = low distraction right target
    3 = high distraction left target
    4 = high distraction right target

    Coding of Enc_response column:
    37 = nature
    39 = manmade
    99 = no response 
    
    """

    # replace Enc_Ret that have value 0 with NAN
    data['Enc_RT'] = data['Enc_RT'].replace(0, np.nan)

    # use value counts method - shows twice as much high distraction than low distraction trials
    print('VALUES IN:', data['Enc_trialtype'].value_counts())
    print('VALUES IN:', data['Enc_response'].value_counts())

    # count NAN values 
    print('NAN VALUES', data['Enc_RT'].isna().sum()) 

    # add ground truth label of pictures, depending on whether their letter is capitalized
    data['Pic_label_left'] = data['Pic_left'].apply(
    lambda x: 'manmade' if x[0].isupper() else 'nature')

    data['Pic_label_right'] = data['Pic_right'].apply(
    lambda x: 'manmade' if x[0].isupper() else 'nature')
    
    # add column to indicate whether target stimulus was left or right
    data['Target'] = data['Enc_trialtype'].apply(
        lambda x: 'left' if (x == 1 or x == 3) else 'right'
    )

    conditions_correct = [
    # correct nature classification
    (data['Pic_label_left'] == 'nature') & ((data['Enc_trialtype'] ==  1) | (data['Enc_trialtype'] == 3)) & (data['Enc_response'] == 37),  

    (data['Pic_label_right'] == 'nature') & ((data['Enc_trialtype'] ==  2) | (data['Enc_trialtype'] == 4))& (data['Enc_response'] == 37),
    
    # correct manmade classification
    (data['Pic_label_left'] == 'manmade') & ((data['Enc_trialtype'] ==  1) | (data['Enc_trialtype'] == 3)) & (data['Enc_response'] == 39),

    (data['Pic_label_right'] == 'manmade') & ((data['Enc_trialtype'] ==  2) | (data['Enc_trialtype'] == 4)) & (data['Enc_response'] == 39)

    ]

    values_correct = ['correct']*4

    # create a new column to reflect semantic classification
    data['Enc_accuracy'] = np.select(conditions_correct, values_correct, default='incorrect')

    # classify 99 values in Enc_response as no_response
    data.loc[data['Enc_response'] == 99, 'Enc_accuracy'] = 'no_response'

    data.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral', 'processed', f'sub-{sub}-enc_processed.csv'), sep = ',', header = True, index = False)

    return data 


def process_ret_data(sub, data):

    """Process data from retrieval phase.
    
    Coding of Ret_trialtype column:
    1 = low distraction target left
    2 = low distraction target right
    3 = high distraction target left
    4 = high distraction target right
    40 = distractor stimulus
    92 = novel stimulus

    Coding of Ret_response column:
    37 = old
    39 = new
    99 = no response 

    Coding of Conf_response column:
    37 = not very confident
    40 = medium confident
    39 = very confident
    
    """

    # replace 0 Ret_RT values & Conf_Ret values with NAN
    data['Ret_RT'] = data['Ret_RT'].replace(0, np.nan)
    data['Conf_RT'] = data['Conf_RT'].replace(0, np.nan)

    # Encode hits & misses 
    conditions = [

    ## Hits 
    # hit low distraction target
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  1) | (data['Ret_trialtype'] == 2)), 

    # hit high distraction target
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  3) | (data['Ret_trialtype'] == 4)), 

    # hit distractor
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  30) | (data['Ret_trialtype'] == 40)),  

    ## Misses
    # miss low distraction target
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  1) | (data['Ret_trialtype'] == 2)),  

    # miss high distraction targets
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  3) | (data['Ret_trialtype'] == 4)), 

    # miss distractor 
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  30) | (data['Ret_trialtype'] == 40)), 

    ## false alarms
    # new items that were incorrectly classified as old
    (data['Ret_response'] == 37) & ((data['Ret_trialtype'] ==  91) | (data['Ret_trialtype'] == 92)), 

    ## correct new
    # new items that were correctly classified as new
    (data['Ret_response'] == 39) & ((data['Ret_trialtype'] ==  91) | (data['Ret_trialtype'] == 92)),

    # no responses
    (data['Ret_response'] == 99)

    ]

    # create list of values to assign to each condition
    values = ['hit_low_dist_target', 'hit_high_dist_target', 'hit_distractor', 'miss_low_dist_target', 'miss_high_dist_target', 'miss_distractor', 'false_alarms', 'correct_new', 'no_response']


    # create column for hit_category
    data['hit_category'] = np.select(conditions, values,  default=None)


    # create confidence judgements
    conditions_confidence = [

        # not confident
        (data['Conf_response'] ==  37), 

        # medium confident
        (data['Conf_response'] ==  40), 

        # very confident 
        (data['Conf_response'] ==  39),

        # no response
        (data['Conf_response'] ==  99),

    ]

    values_confidence = ['low', 'medium', 'high', 'no_response']

    data['confidence'] = np.select(conditions_confidence, values_confidence, default=None) 

    # drop unnecessary columns
    data = data.drop(columns = ['Gender',  'Edu', 'TrialNr'])

    # save data as new csv file
    data.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral', 'processed', f'sub-{sub}-ret_processed.csv'), sep = ',', header = True, index = False)
 
    return data


def calculate_encoding_task_performance(data):
    """Computes ratio of: count correctly classified pictures / count total pictures
    during encoding phase.
    """
    # count correct classified
    count_correct = count_total = data['Enc_accuracy'].isin(['correct']).sum()

    # count total with no responses excluded
    count_total = data['Enc_accuracy'].isin(['correct', 'incorrect']).sum()

    return count_correct / count_total


def calculate_retrieval_task_performance(data):
    """Computes ratio of correct / total responses during retrieval phase.
    Excluding non-responses.

    Responses correct: 
    - hits (targets & distractors)
    - correct new

    Responses incorrect:
    - misses (targets & distractors)
    - false alarms 
    """

    # count correct responses
    count_hits = data['hit_category'].str.contains('hit').sum() 
    count_hits_new = data['hit_category'].str.contains('correct_new').sum()

    count_correct = count_hits + count_hits_new
                     
    # count incorrect responses
    count_misses = data['hit_category'].str.contains('miss').sum()
    count_false_alarms = data['hit_category'].str.contains('false_alarms').sum()

    count_incorrect = count_misses + count_false_alarms


    # sanity checks
    count_no_response = data['hit_category'].str.contains('no_response').sum()
    total_count = data['hit_category'].count()
    assert count_correct + count_incorrect + count_no_response == total_count

    return count_correct / (count_correct + count_incorrect)


def calculate_hitrate(data, trial_wise=False):

    """Calculate hitrate: success in recognizing old items (targets & distractors) only. Ignores new items entirely.

    Formula:

    hit rate = Correct old responses to old items / total old items presented

    """

    # count correctly identified old items (targets & distractors)
    count_old_hits = data['hit_category'].str.contains('hit').sum()

    # count number of missed old items
    count_old_misses =  data['hit_category'].str.contains('miss').sum()

        
    if trial_wise:
        
        #  hits & misses for low distraction targets
        hits_low_dist = data['hit_category'].str.contains('hit_low_dist_tar').sum()
        miss_low_dist = data['hit_category'].str.contains('miss_low_dist_tar').sum()
        
        # hits & misses for high distraction targets
        hits_high_dist = data['hit_category'].str.contains('hit_high_dist_tar').sum()
        miss_high_dist = data['hit_category'].str.contains('miss_high_dist_tar').sum()

        # hits & misses for distractors
        hits_distractor = data['hit_category'].str.contains('hit_distractor').sum()
        miss_distractor = data['hit_category'].str.contains('miss_distractor').sum()

        assert (hits_low_dist + hits_high_dist + hits_distractor) == count_old_hits

        return (hits_low_dist / (hits_low_dist + miss_low_dist)), (hits_high_dist / (hits_high_dist + miss_high_dist)), (hits_distractor / (hits_distractor + miss_distractor))
    
    else:
        return count_old_hits / (count_old_hits + count_old_misses)


def calculate_fa_rate(data): 
    """Calculates FA rate: False Positives / (False Positives + True Negatives). 
    """
    # define false positives
    false_positives = data['hit_category'].str.contains('false_alarms').sum()

    # define true negatives 
    true_negatives = data['hit_category'].str.contains('correct_new').sum()

    # return false alarm rate
    return false_positives / (false_positives + true_negatives)


def calculate_d_prime(hitrate, fa_rate):
    # standardize hitrate & false alarm rate
    z_score_hitrate = scipy.stats.norm.ppf(hitrate)
    z_score_farate = scipy.stats.norm.ppf(fa_rate)

    return (z_score_hitrate - z_score_farate)


def classify_age(data):
    """Classifies subject as OA (> 40) or YA."""

    if (data['Age'] > 40).any():
        return 'old'
    else:
        return 'young'


def create_behavioral_summary(sub, data_enc, data_ret):
    """Creates a summary pd.DataFrame of the behavioral data, 
    including: Sub-ID, Age, Task Performance, Hit Rates, FA Rate, Mean RTs.
    """

    Sub_ID = sub 

    Age = classify_age(data_enc)

    Enc_performance = calculate_encoding_task_performance(data_enc)
    Ret_performance = calculate_retrieval_task_performance(data_ret)

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


    # calculate count of non-responses during enc & ret
    NAN_RT_enc = data_enc['Enc_RT'].isna().sum()
    NAN_RT_ret = data_ret['Ret_RT'].isna().sum()

    # calculate mean reaction time for enc and ret data
    mean_RT_enc = data_enc['Enc_RT'].mean()
    mean_RT_ret = data_ret['Ret_RT'].mean()

    # TODO: I should maybe also include the average hitrate for targets
    df = pd.DataFrame({
    'sub': [Sub_ID],
    'age': [Age],
    'enc_performance': [Enc_performance],
    'ret_performance': [Ret_performance],
    'enc_non_responses': [NAN_RT_enc],
    'ret_non_responses': [NAN_RT_ret],
    'hitrate_global': [hitrate_global],
    'hitrate_lowdist': [hitrate_low],
    'hitrate_highdist': [hitrate_high],
    'hitrate_distractors': [hitrate_dist],
    'd_prime_global': [D_Prime_Global],
    'd_prime_highdist': [D_Prime_High],
    'd_prime_lowdist': [D_Prime_Low],
    'd_prime_distractors': [D_Prime_Dist],
    'd_prime_targets': [D_Prime_Targets_Avg],
    'RT_enc_mean': [mean_RT_enc],
    'RT_ret_mean': [mean_RT_ret]

})

    # save pandas dataframe to file
    df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral','summary_stats', f'sub-{sub}_summary.csv'), sep = ',', header = True, index = False)

    return df


if __name__ == "__main__":
    pass