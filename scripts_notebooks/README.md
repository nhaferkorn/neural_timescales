# Scripts

**Preprocessing scripts:**  
`run_preprocessing.py`: runs all of the preprocessing steps (filtering, annotations, ica etc.)
`run_qc`: run quality control after preprocessing & plots the cleaned signal

**Timescale Analysis scripts:**  
On single-trial data:  
`run_timescales_time.py`: perform single-trial timescale estimation in time-domain (on autocorrelation function, using acf fit)  

On trial-averaged data:  
`timescales_evoked.py`: perform timescale estimation on trial-averaged data

**Statistical analysis scripts:**  
`run_stats_age.py`: computes independent samples cluster permutation test to compare timescales between Old (> 40) and Young Adults.  
`run_stats_group.py`: computes 1samp cluster permutation test to compare high vs. low distraction timescales during the encoding phase.  

**Other analysis scripts**:
`run_behavioral.py`: 
