"""This script runs simulations and produces plots to explore the effects of oscillatory activity on timescale estimation."""


# make imports 
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import mne
import neurodsp
import fooof 

# import functions from neurodsp package
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.spectral import compute_spectrum
from neurodsp.spectral import compute_spectrum, trim_spectrum

# for animating it: https://matplotlib.org/stable/gallery/widgets/slider_demo.html

# specify signal parameters
n_seconds = 1
fs = 300
freq = 10

# simulate perfect oscillatory signal
sig_osc = neurodsp.sim.sim_oscillation(n_seconds, fs, freq)

print(sig_osc)

# plot the simulated signal
times_vector = np.arange(0,1,1/fs) # TODO: check how it works exactly
times = neurodsp.utils.create_times(n_seconds, fs)
print(times)

