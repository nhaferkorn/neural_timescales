##### this script checks existing directory structure and content ############

import os
import mne

from pathlib import Path

    
## Starting off: we want data directory that includes two subdirectories: EEG & Behavioral
data_dir = Path(r"C:/Users/neleh/OneDrive - Radboud Universiteit/Documents/CNS_Master/Master_Thesis/neural_timescales/data")
eeg_dir = data_dir / "eeg"


# check if eeg and behavioral folders exist - listing subdirectories; but this is not case sensitive
def main():
    for item in data_dir.iterdir():
        print(f"{item} - {'dir' if item.is_dir() else 'file'}")

    if eeg_dir.exists():
        print("EEG directory exists")
    
    else:
        print("Creating EEG directory")
        eeg_dir.mkdir()

if __name__ == "__main__":
    main()