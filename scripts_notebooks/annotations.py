"""This script annotates raw EEG data.
"""
# make imports



# mark end of encoding block and start of retrieval block 
mapping_blocks = {13:"End Encoding" , 50:"Start Retrieval", 93:"End Retrieval"}
annot_from_events = mne.annotations_from_events(events, event_desc = mapping_blocks, sfreq = raw_filtered.info["sfreq"], orig_time=raw.info["meas_date"],)
raw_filtered.set_annotations(annot_from_events)

print(raw_filtered.annotations.onset)

#  Annotate blocks
onsets_blocks = [raw_filtered.first_time, raw_filtered.annotations.onset[1]]
durations_blocks = [raw_filtered.annotations.onset[0], raw_filtered.annotations.onset[2]]
descriptions_blocks = ["encoding block", "retrieval block"]

# okay so this works now!
block_annots = mne.Annotations(
    onset=onsets_blocks,
    duration=durations_blocks,
    description=descriptions_blocks,
    orig_time=raw_filtered.info["meas_date"],
)

# Annotate break between encoding & retrieval phase
onset_break = raw_filtered.annotations.onset[0]
duration_break = raw_filtered.annotations.onset[1] - raw_filtered.annotations.onset[0]
description_break = "BAD Break"

break_annots = mne.Annotations(
    onset=onset_break,
    duration=duration_break,
    description=description_break,
    orig_time=raw_filtered.info["meas_date"],
)


raw_filtered.set_annotations(annot_from_events + break_annots)



# Annotate rest periods 
mapping_rests = {90:"Rest onset", 91:"Rest offset"}

annot_from_events_rests = mne.annotations_from_events(events, event_desc = mapping_rests, sfreq = raw_filtered.info["sfreq"], orig_time=raw.info["meas_date"],)

# this should reset the annotations - yesss, it does!
raw_filtered.set_annotations(annot_from_events_rests)

print(raw_filtered.annotations.onset)

# okay - basically I have to loop through the list and always select the pairs: I guess I need to implement a counter of some sorts: but this doesnt really work
onsets_rests = []
durations_rests = []

# okay, this works; but might not be the most elegant solution
for i in range(0, len(raw_filtered.annotations.onset)-1, 2):
    # set onset of rest
    onsets_rest = raw_filtered.annotations.onset[i]
    onsets_rests.append(onsets_rest)

    # set duration of rest
    durations_rest = raw_filtered.annotations.onset[i+1] - raw_filtered.annotations.onset[i]
    durations_rests.append(durations_rest)


print(onsets_rests)
print(durations_rests)

descriptions_rest = ["BAD_Rest Period"] * len(onsets_rests)
orig_times = raw_filtered.info["meas_date"]


rest_annots = mne.Annotations(
    onset=onsets_rests, 
    duration=durations_rests,
    description=descriptions_rest,
    orig_time=orig_times)


raw_filtered.set_annotations(annot_from_events_rests + rest_annots + break_annots)