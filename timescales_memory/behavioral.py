"""This script includes functions to process the behavioral data."""

# make imports
import sys
import os
import scipy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import cmocean
from scipy.stats import norm

# import paths
from timescales_memory.settings import PROJECT_DIR, BEHAV_DIR, DERIV_DIR

def read_behav_data(sub, phase):
    """Reads the .xlsx files for each subject as pd.DataFrame.
    """
    if phase == 'enc':
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Enc_{sub}.xlsx"),  names = ['sub', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Enc_trialtype', 'Enc_response', 'Enc_RT'])
    elif phase == "ret":
        return pd.read_excel(os.path.join(BEHAV_DIR, f"Ret_{sub}.xlsx"),  names = ['sub', 'Gender', 'Age', 'Edu', 'TrialNr', 'Pic_left', 'Pic_right', 'Ret_trialtype', 'Ret_response', 'Ret_RT', 'Conf_response', "Conf_RT"])
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
    count_correct = data['Enc_accuracy'].isin(['correct']).sum()

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


# def calculate_hitrate(data, trial_wise=False):

#     """Calculate hitrate: success in recognizing old items (targets & distractors). Ignores new items.

#     Formula:

#     hit rate = correct old responses to old items / total old items presented

#     """

#     # count correctly identified old items (targets & distractors)
#     count_old_hits = data['hit_category'].str.contains('hit').sum()

#     # count number of missed old items
#     count_old_misses =  data['hit_category'].str.contains('miss').sum()

    
#     if trial_wise:
        
#         #  hits & misses for low distraction targets
#         hits_low_dist = data['hit_category'].str.contains('hit_low_dist_target').sum()
#         miss_low_dist = data['hit_category'].str.contains('miss_low_dist_target').sum()
        
#         # hits & misses for high distraction targets
#         hits_high_dist = data['hit_category'].str.contains('hit_high_dist_target').sum()
#         miss_high_dist = data['hit_category'].str.contains('miss_high_dist_target').sum()

#         # hit rate for targets
#         hits_targets = hits_low_dist + hits_high_dist
#         miss_targets = miss_low_dist + miss_high_dist

#         # hits & misses for distractors
#         hits_distractor = data['hit_category'].str.contains('hit_distractor').sum() 
#         miss_distractor = data['hit_category'].str.contains('miss_distractor').sum()

#         assert (hits_low_dist + hits_high_dist + hits_distractor) == count_old_hits

#         return (hits_low_dist / (hits_low_dist + miss_low_dist)), (hits_high_dist / (hits_high_dist + miss_high_dist)), (hits_targets / (hits_targets + miss_targets)), (hits_distractor / (hits_distractor + miss_distractor))
    
#     else:
#         return count_old_hits / (count_old_hits + count_old_misses)


def calculate_hitrate(data, trial_wise=False):
    """Calculate hitrate: success in recognizing old items (targets & distractors). Ignores new items.

    Formula:
    hit rate = correct old responses to old items / total old items presented
    """

    # explicitly define categories
    hit_labels = {
        'hit_low_dist_target',
        'hit_high_dist_target',
        'hit_distractor'
    }

    miss_labels = {
        'miss_low_dist_target',
        'miss_high_dist_target',
        'miss_distractor'
    }

    # total old hits & misses
    count_old_hits = data['hit_category'].isin(hit_labels).sum()
    count_old_misses = data['hit_category'].isin(miss_labels).sum()

    if trial_wise:

        # low distraction targets
        hits_low = data['hit_category'].eq('hit_low_dist_target').sum()
        miss_low = data['hit_category'].eq('miss_low_dist_target').sum()

        # high distraction targets
        hits_high = data['hit_category'].eq('hit_high_dist_target').sum()
        miss_high = data['hit_category'].eq('miss_high_dist_target').sum()

        # distractors
        hits_dist = data['hit_category'].eq('hit_distractor').sum()
        miss_dist = data['hit_category'].eq('miss_distractor').sum()

        # targets combined
        hits_targets = hits_low + hits_high
        miss_targets = miss_low + miss_high

        # strict sanity checks (fail if data is missing)
        assert (hits_low + miss_low) > 0, "No low-dist target trials"
        assert (hits_high + miss_high) > 0, "No high-dist target trials"
        assert (hits_targets + miss_targets) > 0, "No target trials"
        assert (hits_dist + miss_dist) > 0, "No distractor trials"

        # global sanity check
        assert (hits_low + hits_high + hits_dist) == count_old_hits

        return (
            hits_low / (hits_low + miss_low),
            hits_high / (hits_high + miss_high),
            hits_targets / (hits_targets + miss_targets),
            hits_dist / (hits_dist + miss_dist)
        )

    else:
        # also guard global case
        assert (count_old_hits + count_old_misses) > 0, "No old-item trials"

        return count_old_hits / (count_old_hits + count_old_misses)
    

# also calculate hitrate for only high confidence targets!
# def calculate_hitrate_confidence(data):
        
#         # first filter data to keep only high confidence rows
#         data = data[data['confidence'].str.contains('high')]

#         print(data.head(10))
        
#         # hits & misses for low distraction targets
#         hits_low_dist = data['hit_category'].str.contains('hit_low_dist_target').sum()
#         miss_low_dist = data['hit_category'].str.contains('miss_low_dist_target').sum()
        
#         # hits & misses for high distraction targets
#         hits_high_dist = data['hit_category'].str.contains('hit_high_dist_target').sum()
#         miss_high_dist = data['hit_category'].str.contains('miss_high_dist_target').sum()

#         # hit rate for targets
#         hits_targets = hits_low_dist + hits_high_dist
#         miss_targets = miss_low_dist + miss_high_dist

#         return (hits_targets / (hits_targets + miss_targets))
    
   
def calculate_hitrate_confidence(data):
    """High-confidence hitrate:
    proportion of old items correctly recognized with high confidence.
    """

    # define old items
    old_labels = {
        'hit_low_dist_target', 'miss_low_dist_target',
        'hit_high_dist_target', 'miss_high_dist_target',
        'hit_distractor', 'miss_distractor'
    }

    # define hits
    hit_labels = {
        'hit_low_dist_target',
        'hit_high_dist_target',
        'hit_distractor'
    }

    # mask old items
    old_mask = data['hit_category'].isin(old_labels)

    # numerator: high-confidence hits
    highconf_hits = (
        old_mask &
        data['hit_category'].isin(hit_labels) &
        (data['confidence'] == 'high')
    ).sum()

    # denominator: ALL old items
    total_old = old_mask.sum()

    assert total_old > 0, "No old-item trials"

    return highconf_hits / total_old

def calculate_fa_rate(data): 
    """Calculates FA rate: False Positives / (False Positives + True Negatives). 
    """
    # define false positives
    false_positives = data['hit_category'].str.contains('false_alarms',na=False).sum()

    # define true negatives 
    true_negatives = data['hit_category'].str.contains('correct_new',na=False).sum()

    # return false alarm rate
    return false_positives / (false_positives + true_negatives)


def calculate_accuracy(hitrate, fa_rate):
    """memory performance operationalized as hitrate - false alarmrate"""
    return hitrate - fa_rate

def classify_age(data):
    """Classifies subject as OA (> 40) or YA."""

    if (data['Age'] > 40).any():
        return 'old'
    else:
        return 'young'
    



### not sure if this actually works, but I can try
def compute_dprime_targets_only(data):
    """
    Computes target-only d' for retrieval phase.

    Signal = targets (low + high distraction)
    Noise  = novel items only (correct_new + false alarms context)
    Distractors are excluded for cleaner memory sensitivity estimate.
    """

    # targets
    hit_targets = data['hit_category'].isin([
        'hit_low_dist_target',
        'hit_high_dist_target'
    ]).sum()

    miss_targets = data['hit_category'].isin([
        'miss_low_dist_target',
        'miss_high_dist_target'
    ]).sum()

    # new items
    fa = data['hit_category'].eq('false_alarms').sum()
    cr = data['hit_category'].eq('correct_new').sum()

    # hit rate and fa rate
    H = hit_targets / (hit_targets + miss_targets)
    F = fa / (fa + cr)

    dprime = norm.ppf(H) - norm.ppf(F)

    return dprime


def create_behavioral_summary(sub, data_enc, data_ret):
    """Creates a summary pd.DataFrame of the behavioral data, 
    including: Sub-ID, Age, Task Performance, Hit Rates, FA Rate, Mean RTs.
    """

    Sub_ID = sub 

    Age_numeric = data_enc["Age"].iloc[0]

    Age = classify_age(data_enc)

    Enc_performance = calculate_encoding_task_performance(data_enc)
    Ret_performance = calculate_retrieval_task_performance(data_ret)

    # calculate false alarm rate
    farate = calculate_fa_rate(data_ret)

    # calculate hitrates
    hitrate_global = calculate_hitrate(data_ret)
    hitrate_low, hitrate_high, hitrate_targets, hitrate_dist = calculate_hitrate(data_ret, trial_wise=True)
    hitrate_highconf_targets = calculate_hitrate_confidence(data_ret)


    # difference between hitrate targets and hitrate distractors
    hitrate_diff = hitrate_targets - hitrate_dist

    # calculate count of non-responses during enc & ret
    NAN_RT_enc = data_enc['Enc_RT'].isna().sum()
    NAN_RT_ret = data_ret['Ret_RT'].isna().sum()

    # calculate total trials
    trial_count_enc = len(data_enc['Enc_RT'])
    trial_count_ret = len(data_ret['Ret_RT'])

    # percentage non-responses
    NAN_percent_enc = NAN_RT_enc / trial_count_enc
    NAN_percent_ret = NAN_RT_ret /  trial_count_ret

    # calculate mean reaction time for enc and ret data
    mean_RT_enc = data_enc['Enc_RT'].mean()
    mean_RT_ret = data_ret['Ret_RT'].mean()

    # als calculate median (cause the mean is a biased estimator for skewed distributions)
    median_RT_enc = data_enc['Enc_RT'].median()
    median_RT_ret = data_ret['Ret_RT'].median()

    # also calculate log-transformed RTs
    mean_RT_enc_log = np.log10(data_enc['Enc_RT']).mean()
    mean_RT_ret_log = np.log10(data_ret['Ret_RT']).mean()

    # also calculate RT_ret split by distraction load
    mean_RT_ret_high = data_ret[data_ret['hit_category'].str.contains('high_dist')]['Ret_RT'].mean() # for both hits and misses
    mean_RT_ret_low = data_ret[data_ret['hit_category'].str.contains('low_dist')]['Ret_RT'].mean() # for both hits and misses

    # calculate d prime
    d_prime = compute_dprime_targets_only(data_ret)

    df = pd.DataFrame({
    'sub': [Sub_ID],
    'age_numeric':[Age_numeric],
    'age': [Age],
    'enc_performance': [Enc_performance],
    'ret_performance': [Ret_performance],
    'enc_trial_count': [trial_count_enc],
    'enc_non_responses': [NAN_RT_enc],
    'enc_percent_non_responses': [NAN_percent_enc],
    'ret_trial_count':[trial_count_ret],
    'ret_non_responses': [NAN_RT_ret],
    'ret_percent_non_responses':[NAN_percent_ret],
    'hitrate_global': [hitrate_global],
    'hitrate_targets':[hitrate_targets],
    'hitrate_targets_highconf':[hitrate_highconf_targets],
    'hitrate_lowdist': [hitrate_low],
    'hitrate_highdist': [hitrate_high],
    'hitrate_distractors': [hitrate_dist],
    'hitrate_diff': [hitrate_diff],
    'RT_enc_mean': [mean_RT_enc],
    'RT_enc_median':[median_RT_enc],
    'RT_ret_mean': [mean_RT_ret],
    'RT_ret_median':[median_RT_ret],
    'RT_enc_log_mean':[mean_RT_enc_log],
    'RT_ret_log_mean':[mean_RT_ret_log],
    'RT_ret_mean_highdist':[mean_RT_ret_high],
    'RT_ret_mean_lowdist':[mean_RT_ret_low],
    'fa_rate': [farate],
    'd_prime':[d_prime]


})

    # save pandas dataframe to file
    df.to_csv(path_or_buf = os.path.join(DERIV_DIR, 'behavioral','summary_stats', f'sub-{sub}_summary.csv'), sep = ',', header = True, index = False)

    return df



def corr_heatmap_with_pval(df, method = 'pearson', figsize=(20, 10), title=None, path=None, labels=None):
  """
  df: dataframe to be used. Ensured the dataframe has been sliced to contain only the column you need. It accepts only numerical columns
  method: default uses the pearson method. It overall permits 3 methods; 'pearson', 'spearman' and 'kendall'
  figsize: default is (20, 10) but you can change it based on your preference
  title: Specify the title for your chart, default is None
  """
  # Make a copy of the df
  data = df.copy()
  
  if labels is not None:
     data = data.rename(columns=labels)

  # Check features correlation
  corr = data.corr(method = method)

  # Create a mask to hide the upper triangle
  mask = np.zeros_like(corr, dtype=bool)
  mask[np.triu_indices_from(mask)] = True

  # Set the diagonal elements of the mask to False to display self-correlation
  np.fill_diagonal(mask, False)

  fig, ax = plt.subplots(figsize=figsize)
  plt.title(title, fontsize=30)

  # Create the heatmap with the custom mask
  heatmap = sns.heatmap(corr,
                        annot=True,
                        annot_kws={"fontsize": 16},  # Adjust annotation font size
                        fmt='.2f',
                        linewidths=0.5,
                        cmap=cmocean.cm.tempo,
                        mask=mask,
                        ax=ax)

  # Create a function to calculate and format p-values
  p_values = np.full((corr.shape[0], corr.shape[1]), np.nan)
  for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[1]):
      x = data.iloc[:, i]
      y = data.iloc[:, j]
      mask = ~np.logical_or(np.isnan(x), np.isnan(y))
      if np.sum(mask) > 0:
        if method == 'pearson':
          p_values[i, j] = pearsonr(x[mask], y[mask])[1] #Changes based on the method chosen in the function
        elif method == 'kendall':
          p_values[i, j] = kendalltau(x[mask], y[mask])[1]
        elif method == 'spearman':
          p_values[i, j] = spearmanr(x[mask], y[mask])[1]
  
  p_values = pd.DataFrame(p_values, columns=corr.columns, index=corr.index)

  # Create a mask for the p-values heatmap
  mask_pvalues = np.triu(np.ones_like(p_values), k=1)

  # Calculate the highest and lowest correlation coefficients
  max_corr = np.max(corr.max())
  min_corr = np.min(corr.min())
  
  # Get colormap + normalization from heatmap
  cmap = ax.collections[0].cmap
  norm = ax.collections[0].norm

  # Annotate the heatmap with p-values and change text color based on correlation value
  for i in range(p_values.shape[0]):
    for j in range(p_values.shape[1]):
      if mask_pvalues[i, j]:
        p_value = p_values.iloc[i, j]
        if not np.isnan(p_value):
          correlation_value = corr.iloc[i, j]
          # ---- LUMINANCE-BASED TEXT COLOR ----
          rgba = cmap(norm(correlation_value))
          r, g, b, _ = rgba
          luminance = 0.299*r + 0.587*g + 0.114*b
          text_color = 'white' if luminance < 0.5 else 'black'

          # text_color = 'white' if correlation_value >= (max_corr - 0.4) or correlation_value <= (min_corr + 0.4) else 'black'
          if p_value <= 0.01:
            #include double asterisks for p-value <= 0.01
            ax.text(i + 0.5, j + 0.8, f'(p <= {p_value:.2f})**',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color=text_color)
          elif p_value <= 0.05:
            #include single asterisks for p-value <= 0.05
            ax.text(i + 0.5, j + 0.8, f'(p <= {p_value:.2f})*',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color=text_color)
          else:
            ax.text(i + 0.5, j + 0.8, f'(p = {p_value:.2f})',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    color=text_color)

  # Customize x-axis labels
  x_labels = [textwrap.fill(label.get_text(), 15) for label in ax.get_xticklabels()]
  ax.set_xticklabels(x_labels, rotation=0, ha="center", fontsize=14)

  # Customize y-axis labels
  y_labels = [textwrap.fill(label.get_text(), 15) for label in ax.get_yticklabels()]
  ax.set_yticklabels(y_labels, rotation=0, ha="right", fontsize=14)
  ax.grid(False)
  plt.savefig(path)



if __name__ == "__main__":
   pass