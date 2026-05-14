"""This script processes the behavioral data at the group level"""

# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import cmocean
import textwrap
import scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import ttest_ind
from datetime import datetime
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib.gridspec import GridSpec
from statannotations.Annotator import Annotator



# specify date
now = datetime.now()
date  =  now.strftime("%d-%m-%Y")


# import custom functions
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest
from timescales_memory.behavioral import read_behav_data, process_enc_data, process_ret_data, calculate_encoding_task_performance, calculate_retrieval_task_performance, create_behavioral_summary, calculate_hitrate, calculate_fa_rate

# specify subjects & list to exclude
exclude = [102, 105, 110, 115, 116, 123, 133]
subjects = [f"sub-{i}" for i in range(101, 162) if i not in exclude]

# create empty list to store subjects csv files
dfs = []

for sub in subjects:

    # read summary data
    file_path = os.path.join(DERIV_DIR, 'behavioral', 'summary_stats', f'{sub}_summary.csv')

    # load encoding data
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dfs.append(df)

df_behav = pd.concat(dfs, axis = 0, join='outer')

print("Encoding Performance", df_behav["enc_performance"].agg(['mean','std', 'sem']))
print("Non-responses Encoding", df_behav["enc_percent_non_responses"].agg(['mean','std', 'sem']))
print("Non-responses Retrieval", df_behav["ret_percent_non_responses"].agg(['mean','std', 'sem']))

print(
    df_behav.groupby("age")["RT_ret_mean"]
    .agg(['mean', 'std', 'sem'])
)


# t-test for response times during distraction load
# low = df_behav.loc[df_behav["age"] == "young", "RT_enc_mean"]
# high   = df_behav.loc[df_behav["age"] == "old", "RT_enc_mean"]



# print(df_behav[['enc_percent_non_responses', 'ret_percent_non_responses']].describe())

# # compute descriptive statistics - memory readouts
# print(df_behav[['hitrate_global', 'hitrate_targets', 'hitrate_lowdist', 'hitrate_highdist']].describe())

# # compute descriptive statistics for reaction time readouts 
# print(df_behav[['RT_enc_mean', 'RT_ret_mean']].describe())


# split data by group
# young = df_behav.loc[df_behav["age"] == "young", "RT_enc_mean"]
# old   = df_behav.loc[df_behav["age"] == "old", "RT_enc_mean"]

# print('ENCODING')
# print()
# print(young.mean())
# print(old.mean())


# # split also by distraction load of target stimuli!
# high = df_behav["RT_ret_mean_highdist"]
# low = df_behav["RT_ret_mean_lowdist"]

# print(f'Mean Highdist = {high.mean()}')
# print(f'Mean Lowdist = {low.mean()}')


# # split by age and distraction load:
# high_young = df_behav.loc[df_behav["age"] == "young","RT_ret_mean_highdist"]
# low_young = df_behav.loc[df_behav["age"] == "young","RT_ret_mean_lowdist"]
# high_old = df_behav.loc[df_behav["age"] == "old","RT_ret_mean_highdist"]
# low_old = df_behav.loc[df_behav["age"] == "old","RT_ret_mean_lowdist"]


# print('high young mean', high_young.mean())
# print('low young mean', low_young.mean())
# print('high old mean', high_old.mean())
# print('low old mean', low_old.mean())

# # independent t-test (two-sample)
# t_stat, p_val = ttest_ind(young, old, equal_var=True)  

# print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
# n1 = len(young)
# n2 = len(old)

# # compute degrees of freedom
# df = n1 + n2 - 2
# print(df)


# ## descriptive stats for hitrates 
# young_hits = df_behav.loc[df_behav["age"] == "young", "hitrate_targets"]
# old_hits   = df_behav.loc[df_behav["age"] == "old", "hitrate_targets"]

# print('Stats - Hitrates (Targets)')
# print(young_hits.mean())
# print(old_hits.mean())

# # independent t-test (two-sample)
# t_stat, p_val = ttest_ind(young_hits, old_hits, equal_var=True)  

# print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
# n1 = len(young)
# n2 = len(old)

# # compute degrees of freedom
# df = n1 + n2 - 2
# print(df)


# print(young_hits) 
# subject 104 has only 9 % hitrate!! old (so probably exclude this subject)

# # plot distributions of mean RTs
# fig, ax = plt.subplots()
# seaborn.kdeplot(df_behav[['RT_enc_mean', 'RT_ret_mean']], ax = ax)
# fig.suptitle('Distribution of Means (Enc & Ret)')
# fig.savefig(os.path.join(DERIV_DIR, 'behavioral', 'distribution_mean_plot.png'))

# also as barplots (split by age)

# # reshape
# df_long = df_behav.melt(
#     id_vars="age",
#     value_vars=["RT_enc_mean", "RT_ret_mean"],
#     var_name="RT_type",
#     value_name="RT_value"
# )

# bp = seaborn.catplot(
#     data=df_long, kind="bar",
#     x="age", y="RT_value",
#     hue="RT_type",
#     errorbar="sd", palette="dark",
#     alpha=.6, height=6
# )

# bp.fig.savefig(os.path.join(DERIV_DIR, 'behavioral', 'barplot_age_RT_reshaped.png'))


# barplot of age & target hitrates
fig, ax = plt.subplots()
x = "age"
y = "hitrate_targets"
order = ['young', 'old']
pairs = [('young', 'old')]

ax = sns.barplot(
    data=df_behav,
    x=x, y=y,
    order=order,
    errorbar="se",
    alpha=0.8,
    ax=ax,
    palette=["#1f78b4","#33a02c"], width=0.5,
    capsize=.1
)

annot = Annotator(ax, pairs, data=df_behav, x=x, y=y, order=order)
annot.configure(
    test='t-test_ind',
    text_format='star',
    loc='inside',
    line_width=0,     
    fontsize=18,      
    verbose=2
)
# annot.apply_test()
# ax, test_results = annot.annotate()
ax.text(
    0.5, 0.9, 'ns',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=18
)
ax.set_ylim(0, 0.7)
ax.set_xlabel('Age', fontsize=14)
ax.set_ylabel('Hit Rate (%)', fontsize=14, labelpad=15)
ax.set_xticklabels(['Young', 'Old'], fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style ticks (better approach)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center')

fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_target_hitrate_annotated_{date}.pdf'), bbox_inches="tight")


# barplot of age & distraction hitrates
# fig, ax = plt.subplots()
# x = "age"
# y = "hitrate_distractors"
# order = ['young', 'old']
# pairs = [('young', 'old')]

# ax = sns.barplot(
#     data=df_behav,
#     x=x, y=y,
#     order=order,
#     errorbar="se",
#     alpha=0.8,
#     ax=ax,
#     palette=["#1f78b4","#33a02c"], width=0.5,
#     capsize=.1
# )

# annot = Annotator(ax, pairs, data=df_behav, x=x, y=y, order=order)
# annot.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
# annot.apply_test()
# ax, test_results = annot.annotate()
# ax.set_xlabel('Age', fontsize=14)
# ax.set_ylabel('Hit Rate Distractors (%)', fontsize=14)
# ax.set_xticklabels(['Young', 'Old'], fontsize=12)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # Style ticks (better approach)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.text(-0.15, 1.05, 'b', transform=ax.transAxes,
#             fontsize=18, fontweight='bold', va='center')
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
# fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_hitrate_distractors_annotated_{date}.pdf'), bbox_inches="tight")







# # barplot of age & retrieval reaction times 
fig, ax = plt.subplots()
x = "age"
y = "RT_ret_mean"
order = ['young', 'old']
pairs = [('young', 'old')]

ax = sns.barplot(
    data=df_behav,
    x=x, y=y,
    order=order,
    errorbar="se",
    alpha=0.8,
    ax=ax,
    palette=["#1f78b4","#33a02c"], width=0.5,
    capsize=.1
)

annot = Annotator(ax, pairs, data=df_behav, x=x, y=y, order=order)
annot.configure(
    test='t-test_ind',
    text_format='star',
    loc='inside',
    line_width=0,     
    fontsize=18,      
    verbose=2
)
# annot.apply_test()
# ax, test_results = annot.annotate()
#     ax.text(
#     0.5, 1, 'ns.',
#     ha='center',
#     va='bottom',
#     fontsize=14
# )
ax.text(
    0.5, 0.9, '**',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=18
)
ax.set_ylim(0, 1.2)
ax.set_xlabel('Age', fontsize=14)
ax.set_ylabel(' Retrieval Response Time (s)', fontsize=14, labelpad=15)
ax.set_xticklabels(['Young', 'Old'], fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center')
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_rts_retrieval_annotated_{date}.pdf'), bbox_inches="tight")



# # plot encoding RT as function of Age
# fig, ax = plt.subplots()
# x = "age"
# y = "RT_enc_mean"
# order = ['young', 'old']
# pairs = [('young', 'old')]

# ax = sns.barplot(
#     data=df_behav,
#     x=x, y=y,
#     order=order,
#     errorbar="se",
#     alpha=0.7,
#     ax=ax,
#     palette=["#1f78b4","#33a02c"], width=0.5,
#     capsize=.1
# )

# annot = Annotator(ax, pairs, data=df_behav, x=x, y=y, order=order)
# annot.configure(test='t-test_ind', text_format='star', loc='inside', verbose=2)
# annot.apply_test()
# ax, test_results = annot.annotate()
# ax.set_xlabel('Age', fontsize=14)
# ax.set_ylabel('Encoding Reaction Times (s)', fontsize=14, labelpad=15)
# ax.set_xticklabels(['Young', 'Old'], fontsize=12)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # Style ticks (better approach)
# ax.tick_params(axis='x', labelsize=12)
# ax.tick_params(axis='y', labelsize=12)
# ax.text(-0.075, 1.05, 'a',
#         transform=ax.transAxes,
#         fontsize=18, fontweight='bold',
#         ha='right', va='bottom')
# # fig.tight_layout()
# # fig.subplots_adjust(top=0.85)
# fig.subplots_adjust(hspace=0.2, wspace=0.2)
# fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_age_rts_encoding_annotated_{date}.pdf'), bbox_inches="tight")







# ## barplot of hitrates split as function of distraction (during encoding phase)
df_long = df_behav.melt(
    value_vars=['hitrate_lowdist', 'hitrate_highdist'],
    var_name='distraction',
    value_name='hitrate'
)

df_long['distraction'] = df_long['distraction'].replace({
    'hitrate_lowdist': 'low',
    'hitrate_highdist': 'high'
})

fig, ax = plt.subplots()
x = "distraction"
y = "hitrate"
order = ['low', 'high']
pairs = [('low', 'high')]

sns.barplot(
    data=df_long,
    x="distraction",
    y="hitrate",
    order=['low', 'high'],
    errorbar="se",
    alpha=0.8,
    ax=ax, 
    palette=['darkmagenta', 'orange'],
    width=0.5,
    capsize=.1

)

annot = Annotator(ax, pairs, data=df_long, x=x, y=y, order=order)
annot.configure(
    test='t-test_paired',
    text_format='star',
    loc='inside',
    line_width=0,     
    fontsize=18,      
    verbose=2
)
# annot.apply_test()
# ax, test_results = annot.annotate()
ax.text(
    0.5, 0.9, 'ns',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=18
)
ax.set_xlabel('Distraction Load', fontsize=14)
ax.set_ylabel('Target Hit Rate (%)', fontsize=14, labelpad=15)
ax.set_ylim(0, 0.7)
ax.set_xticklabels(['Low', 'High'], fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style ticks (better approach)
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.text(-0.15, 1.05, 'b', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center')
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_distraction_hits_annotated_{date}.pdf'), bbox_inches="tight")

# result_dist_hits = scipy.stats.ttest_rel(df_behav['hitrate_lowdist'], df_behav[ 'hitrate_highdist'])
# print('paired ttest: distraction hits', result_dist_hits)



# ## barplot of reaction times split as function of distraction load 
df_long = df_behav.melt(
    value_vars=['RT_ret_mean_lowdist', 'RT_ret_mean_highdist'],
    var_name='distraction',
    value_name='RT'
)

df_long['distraction'] = df_long['distraction'].replace({
    'RT_ret_mean_lowdist': 'low',
    'RT_ret_mean_highdist': 'high'
})

fig, ax = plt.subplots()
x = "distraction"
y = "RT"
order = ['low', 'high']
pairs = [('low', 'high')]

sns.barplot(
    data=df_long,
    x="distraction",
    y="RT",
    order=['low', 'high'],
    errorbar="se",
    ax=ax, 
    alpha=0.8,
    width=0.5,
    capsize=.1,
    palette=['darkmagenta', 'orange']
)

annot = Annotator(ax, pairs, data=df_long, x=x, y=y, order=order)
annot.configure(
    test='t-test_paired',
    text_format='star',
    loc='inside',
    line_width=0,     
    fontsize=18,      
    verbose=2
)
# annot.apply_test()
# ax, test_results = annot.annotate()
ax.text(
    0.5, 0.9, 'ns',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=18
)
ax.set_xlabel('Distraction Load', fontsize=14)
ax.set_ylabel('Retrieval Response Time (s)', fontsize=14, labelpad=15)
ax.set_ylim(0, 1.2)
ax.set_xticklabels(['Low', 'High'], fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style ticks 
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.text(-0.15, 1.05, 'a', transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='center')
fig.subplots_adjust(hspace=0.2, wspace=0.2)
fig.savefig(os.path.join(DERIV_DIR, 'figures', f'barplot_distraction_rts_annotated_{date}.pdf'), bbox_inches="tight")


# # compute paired t-test
# result = scipy.stats.ttest_rel(df_behav['RT_ret_mean_lowdist'], df_behav[ 'RT_ret_mean_highdist'])
# print('paired ttest: distraction reaction time', result)




## plot distraction load as a function of age
df_long = df_behav.melt(
    id_vars='age',
    value_vars=['hitrate_lowdist', 'hitrate_highdist'],
    var_name='distraction',
    value_name='hitrate'
)

df_long['distraction'] = df_long['distraction'].replace({
    'hitrate_lowdist': 'Low',
    'hitrate_highdist': 'High'
})


fig, axes = plt.subplots(1, 2, figsize=(10, 4))

groups = ['young', 'old']
titles = ['Younger Adults', 'Older Adults']

# create once
df_long['age_distraction'] = df_long['age'] + '_' + df_long['distraction']

palette = {
    'young_Low': "#a6cee3",   
    'young_High': "#1f78b4",  
    'old_Low': "#b2df8a",     
    'old_High': "#33a02c"     
}

for ax, group, title in zip(axes, groups, titles):

    data_sub = df_long[df_long['age'] == group]

    sns.barplot(
        data=data_sub,
        x='distraction',
        y='hitrate',
        hue='age_distraction',          
        order=['Low', 'High'],
        palette=palette,
        errorbar='se',
        capsize=.1,
        ax=ax, 
        width=0.5
    )

    # significance (within each age group)
    pairs = [('Low', 'High')]

    annot = Annotator(
        ax, pairs,
        data=data_sub,
        x='distraction',
        y='hitrate',
        order=['Low', 'High']
    )
    annot.configure(
        test='t-test_ind',
        text_format='star',
        loc='inside',
        line_width=0,     
        fontsize=18,      
        verbose=2
    )
    # annot.apply_test()
    # ax, test_results = annot.annotate()

    ax.text(
    0.5, 0.9, 'ns',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=14
)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Distraction Load', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

# remove legend
for ax in axes:
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# shared y-label
axes[0].set_ylabel('Target Hit Rate (%)', fontsize=12, labelpad=15)
axes[1].set_ylabel('Target Hit Rate (%)', fontsize=12,  labelpad=15)


# set ylim 
axes[0].set_ylim(0, 0.7)
axes[1].set_ylim(0, 0.7)

# panel labels
axes[0].text(-0.18, 1.05, 'a', transform=axes[0].transAxes,
             fontsize=16, fontweight='bold', va='center')

axes[1].text(-0.18, 1.05, 'b', transform=axes[1].transAxes,
             fontsize=16, fontweight='bold', va='center')

fig.tight_layout()

fig.savefig(
    os.path.join(DERIV_DIR, 'figures',
                 f'panel_age_distraction_with_stats_{date}.pdf'),
    bbox_inches="tight")


# ###################################################################################
## same for reaction times
df_long_rt = df_behav.melt(
    id_vars='age',
    value_vars=['RT_ret_mean_lowdist', 'RT_ret_mean_highdist'],
    var_name='distraction',
    value_name='RT'
)

df_long_rt['distraction'] = df_long_rt['distraction'].replace({
    'RT_ret_mean_lowdist': 'Low',
    'RT_ret_mean_highdist': 'High'
})

df_long_rt['age_distraction'] = (
    df_long_rt['age'] + '_' + df_long_rt['distraction']
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

groups = ['young', 'old']
titles = ['Younger Adults', 'Older Adults']

palette = {
    'young_Low': "#a6cee3",
    'young_High': "#1f78b4",
    'old_Low': "#b2df8a",
    'old_High': "#33a02c"
}

for ax, group, title in zip(axes, groups, titles):

    data_sub = df_long_rt[df_long_rt['age'] == group]

    sns.barplot(
        data=data_sub,
        x='distraction',
        y='RT',
        hue='age_distraction',
        order=['Low', 'High'],
        palette=palette,
        errorbar='se',
        capsize=.1,
        ax=ax,
        width=0.5
    )

    # remove legend
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    pairs = [('Low', 'High')]

    annot = Annotator(
        ax, pairs,
        data=data_sub,
        x='distraction',
        y='RT',
        order=['Low', 'High']
    )

    annot.configure(
        test='t-test_paired',
        text_format='star',
        loc='inside',
        verbose=0
    )

    # annot.apply_test()
    # annot.annotate()

    ax.text(
    0.5, 0.9, 'ns',
    transform=ax.transAxes,
    ha='center',
    va='bottom',
    fontsize=14
)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Distraction Load', fontsize=12)
    ax.set_ylabel('Retrieval Response Time (s)', fontsize=12, labelpad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

# set ylim 
axes[0].set_ylim(0, 1.2)
axes[1].set_ylim(0, 1.2)


# panel labels
axes[0].text(-0.18, 1.05, 'a', transform=axes[0].transAxes,
             fontsize=16, fontweight='bold', va='center')

axes[1].text(-0.18, 1.05, 'b', transform=axes[1].transAxes,
             fontsize=16, fontweight='bold', va='center')

fig.tight_layout()

fig.savefig(
    os.path.join(DERIV_DIR, 'figures',
                 f'panel_age_distraction_reactiontimes_with_stats_{date}.pdf'),
    bbox_inches="tight"
)




# compute paired t-test for response rates
result = scipy.stats.ttest_rel(df_behav['RT_ret_mean_lowdist'], df_behav[ 'RT_ret_mean_highdist'])
print('paired ttest: distraction reaction time', result)

# compute paired t-test for hit rates
result = scipy.stats.ttest_rel(df_behav['hitrate_lowdist'], df_behav[ 'hitrate_highdist'])
print('paired ttest: distraction hitrate', result)

# independent t-test for age & hits
young_hits = df_behav.loc[df_behav["age"] == "young", "hitrate_targets"]
old_hits   = df_behav.loc[df_behav["age"] == "old", "hitrate_targets"]

print('Stats - Hitrates (Targets)')
print(young_hits.mean())
print(old_hits.mean())
result = ttest_ind(young_hits, old_hits, equal_var=True) 
print('indp ttest: age hitrate', result)


# independent t-test for age & retrieval rts (still need to average!!)
young_rts = df_behav.loc[df_behav["age"] == "young", "RT_ret_mean"]
old_rts   = df_behav.loc[df_behav["age"] == "old", "RT_ret_mean"]

print('Stats - RTs')
print(young_rts.mean())
print(old_rts.mean())
result = ttest_ind(young_rts, old_rts, equal_var=True) 
print('indp ttest: age rts', result)
