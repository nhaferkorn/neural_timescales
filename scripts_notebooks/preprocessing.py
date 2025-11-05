"""This script pre-processes the raw EEG data."""

import os
import numpy as np
import mne
import glob
import re

from src.settings import DATA_DIR, BIDS_ROOT, EVENT_DICT, EVENT_DICT_CLEAN, REPORT_DIR

###################################################################################################
###################################################################################################

# def main():

########################################################
## SETUP

subjs_list = list()
subjs_dirs = list()
raw_files = list()

# and then implement later 
RUN_ICA = True

# list subjects directory - this part of the code is way to complicated and could be more straightforward
sub_ids = [item for item in os.listdir(BIDS_ROOT)
           if os.path.isdir(os.path.join(BIDS_ROOT, item)) and item.startswith('sub-')]

# this just yields a list with: ['sub-01', 'sub-02', 'sub-03']

for subject in sub_ids:
    subject_dir = os.path.join(BIDS_ROOT, subject, "eeg")
    subjs_dirs.append(subject_dir)
    raw_file = glob.glob(os.path.join(subject_dir, '*.bdf')) # this automatically returns list object
    print(raw_file)
    raw_files.extend(raw_file) # extend flattens the list
    
print(raw_files) 
# print("This is the subjects directory", "\n", subjs_dirs)

#########################################################
## RUN ACROSS SUBJECTS

# fix subj_label extraction & make code flexible: such that you can decide which subjects to run
# load raw files for each subject

for raw_file in raw_files:

# Get subject label and print status
    print(raw_file)
    match = re.search(r'sub-(\d+)', raw_file)
    subj_label = match.group(1)
    subjs_list.append(subj_label)
    print('\nCURRENTLY RUNNING SUBJECT: ', subj_label, '\n')

    raw = mne.io.read_raw_bdf(raw_file, preload=True, eog = ["EXG1", "EXG2", "EXG3", "EXG4"], misc = ["EXG5", "EXG6", "EXG7", "EXG8"], stim_channel="STATUS")
    raw = raw.drop_channels(ch_names = ["EXG7", "EXG8"])

    # okay, so  I have to set montage directly on raw object (and not a copy of it!) > this creates a new entry in the info object
    raw.set_montage("biosemi32", on_missing = "ignore")

    # set average reference (but only set after removing bad periods!!! because otherwise it will spread)
    # raw_avg_ref = raw.copy().set_eeg_reference(ref_channels = "average")

    # filter the data
    raw_filtered = raw.copy().filter(l_freq = 0.1, h_freq=45)

    ## downsample data 
    # raw.resample(500)

    ## check info structure
    print(raw.info)
    print(raw.info['ch_names'])
    print(raw.info['dig'])
    sfreq = raw.info["sfreq"]

    # instantiate report object; make sure the report for each subject gets saved
    report = mne.Report(title= f"Sub-{subj_label} Summary Report")

    # specify output path
    output_path = os.path.join(REPORT_DIR, f"subject{subj_label}_report.html")
    
    # This method also accepts a path, e.g., raw=raw_path
    report.add_raw(raw=raw_filtered, title="Raw Bandpass Filtered", psd=False)  # omit PSD plot
    
    
    # find subject-specific events
    events = mne.find_events(raw, stim_channel = "Status", initial_event=False)
    events[:,2] = events[:, 2] - 64512

    report.add_events(events=events, title= 'Events Plot', sfreq = sfreq)
    # report.save(output_path, overwrite=True)

    # interesting way to implement toggle switches in your analysis
    if RUN_ICA:
            raw_ica_filtered = raw.copy().filter(l_freq = 1.5, h_freq=35)
            ica = mne.preprocessing.ICA(n_components=15, max_iter="auto")
            ica.fit(raw_ica_filtered)
            ica

            # plot components
            ica.plot_components()

            # examine the ICs to see what they captured
            explained_var_ratio = ica.get_explained_variance_ratio(raw_filtered)

            for channel_type, ratio in explained_var_ratio.items():
                print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

            # use dedicated EOG channels as a "pattern" to check the ICs against, and automatically mark for exclusion any ICs that match EOG/ECG pattern
            eog_evoked = mne.preprocessing.create_eog_epochs(raw_filtered).average()
            ica.exclude = []

            # find which ICs match the EOG pattern
            eog_indices, eog_scores = ica.find_bads_eog(raw_filtered, threshold = 2.5)
            ica.exclude = eog_indices

            # barplot of ICA component "EOG match" scores
            ica.plot_scores(eog_scores)

            # plot diagnostics
            ica.plot_properties(raw_filtered, picks = eog_indices)

            # plot ICs applied to raw data, with EOG matches highlighted
            ica.plot_sources(raw_filtered, show_scrollbars=False)

            # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
            ica.plot_sources(eog_evoked)

    else:
        continue

    #  Try to adjust the threshold
    eog_events = mne.preprocessing.find_eog_events(raw_filtered, event_id=998, thresh = 200e-6)

 
    raw.copy().pick_types(eog=True).plot(events=eog_events, event_id={'EOG': 998})
    

    # Rejecting Epochs based on peak-to-peak channel amplitude 
    reject_criteria = dict(
        # eeg=100e-6,  # 100 µV
        eog=200e-6,    # 200 µV
    ) 

    flat_criteria = dict(eeg=1e-6)  # 1 µV

    epochs = mne.Epochs(
        raw,
        events, 
        tmin=-0.2,
        tmax=0.5,
        reject_tmax=0,
        reject=reject_criteria,
        flat=flat_criteria,
        reject_by_annotation=False,
        preload=True,
    )
    epochs.plot_drop_log()

    # save this figure of how many epochs were dropped to subject report
    report.add_epochs(epochs=epochs, title='Epochs Object')
    report.save(output_path, overwrite=True)


    # Look at Evoked Data - Check for Visual Components
    raw_lh_filtered = raw_avg_ref.filter(l_freq=0.1, h_freq = 45)
    epochs = mne.Epochs(raw_lh_filtered, events = events, event_id = EVENT_DICT_CLEAN)


    # visualizing epoched data - plot power spectrum of encoding stimulus onset baseline left
    epochs["Encoding Stimulus Onset Baseline Left"].compute_psd().plot(picks="eeg")


    #########################################################################
    #########################################################################
    ## EVOKED RESPONSES

    # compute evoked response; epochs are averaged in a channel-wise fashion
    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average()

    evoked_enc_base_left.plot(titles="left encoding")


    ## plot evoked response relative to Fixation Onset Encoding - we see a decay in power over time (which makes sense, but is also interesting)
    evoked_enc_fix = epochs["Fixation Onset Enc"].average()
    evoked_enc_fix.plot()

    # Compare left vs. right presented stimuli in encoding phase 
    # pick occipital electrodes
    picks = mne.pick_channels(raw.info["ch_names"], ['O1', 'Oz' ,'O2'])
    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average(picks)
    evoked_enc_base_left.plot(titles="left")

    ## Interesting: so, we definitely see a positive deflection centered at around 0.1 Hz, and then the signal plateus afterwards
    evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average(picks)
    evoked_enc_base_right.plot();

    # I guess it make sense to pool over targets and baseline stimuli and then check for left and right side separately 

    evoked_enc_base_left = epochs["Encoding Stimulus Onset Baseline Left"].average().plot()
    evoked_enc_base_right = epochs["Encoding Stimulus Onset Baseline Right"].average().plot()

    # I don't understand why 
    evoked_enc_dist_left = epochs['Encoding Stimulus Onset Distraction Left Target'].average().plot()

    # now, we get a much flatter power spectrum
    enc_pooled_left = epochs[['Encoding Stimulus Onset Baseline Left', 'Encoding Stimulus Onset Distraction Left Target']].average().plot()

    # Evoked estimates for retieval phase
    evoked_ret_dist_left = epochs['Retrieval Stimulus Onset Distraction Left Target'].average().plot()
    evoked_ret_base_left = epochs['Retrieval Stimulus Onset Baseline Left'].average().plot()