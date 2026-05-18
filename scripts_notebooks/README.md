# Scripts

**Preprocessing scripts:**  
i. `preprocessing.py`:  
ii. `run_ica.py`:   
iii. `run_qc`: run quality control after preprocessing & plots the cleaned signal

**Timescale Analysis scripts:**  
On Single-trial level:  
`run_timescales_time.py`: perform single-trial timescale estimation in time-domain (on autocorrelation function, using acf fit)  
`run_timscales_freq.py`: perform single-trial timescale estimation in frequency domain (on psd, using fooof)  
`run_timescales_bayesian.py`: fit generative models to estimate timescales (using aABC tau)

On Evokeds:  
`run_timescales_evoked.py`: perform timescale estimation on evoked objects.

**Statistical Analysis scripts:**  
`run_stats_age.py`: computes independent samples cluster permutation test to compare timescales between Old (> 40) and Young Adults. Also produces topoplots of age effect.  
`run_stats_group.py`: computes 1samp cluster permutation test to compare high vs. low distraction timescales during the encoding phase.  
`run_stats_lateral.py`: 


**Other Analysis Scripts**:
`run_timefreq.py`: replicate time-frequency decomposition from Wynn et al.
`run_behavioral.py`: 

**Jupyter Notebooks (for exploration & plotting)**:  
`nb_behavioral_exploration.ipynb`: compute summary stats for behavioral data. 
`nb_timescales_exploration.ipynb`: check
`nb_simulations.ipybn`: simulate periodic & aperiodic signal, compute autocorr & estimate timescales using neurodsp (visualization purposes!)

