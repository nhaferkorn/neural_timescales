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
import scipy as sp
import fooof 
from matplotlib.widgets import Button, Slider

# import functions from neurodsp package & statsmodels
from statsmodels.tsa.stattools import acf as compute_acf
from neurodsp.utils import create_times
from neurodsp.spectral import compute_spectrum, trim_spectrum
from neurodsp.aperiodic.autocorr import compute_autocorr
from neurodsp.plts.time_series import plot_time_series
from neurodsp.plts import plot_autocorr
from neurodsp.sim.update import SigIter
from neurodsp.sim.aperiodic import sim_powerlaw
from neurodsp.sim.multi import (sim_multiple, sim_from_sampler,
                                sim_across_values, sim_multi_across_values)
from neurodsp.sim.update import create_updater, create_sampler, ParamSampler
from neurodsp.plts.time_series import plot_time_series, plot_multi_time_series
from neurodsp.utils.data import create_times
# import paths & events
from timescales_memory.settings import PROJECT_DIR, EEG_DIR, EVENT_DICT, DERIV_DIR, RAW_CLEANED, events_of_interest, keys


# for animating it: https://matplotlib.org/stable/gallery/widgets/slider_demo.html

# specify signal parameters
n_seconds = 1
fs = 1000
freq = 10

# simulate perfect oscillatory signal
# sig_osc = neurodsp.sim.sim_oscillation(n_seconds, fs, freq)
# sig_powerlaw = neurodsp.sim.sim_powerlaw(n_seconds, fs, exponent=-2.0, f_range=None)

# # plot the simulated signal
# # times_vector = np.arange(0,1,1/fs) # TODO: check how it works exactly
# times = neurodsp.utils.create_times(n_seconds, fs)

# # plot signal
# plt.plot(times, sig_osc)
# plt.xlabel("Signal")
# plt.ylabel("Time")
# plt.title("Oscillation Plot")
# plt.savefig("oscillation_plot.png")  # Save figure to file


# # compute autocorrelation using neurodsp
# timepoints, autocorrs = compute_autocorr(sig_osc)

# # but it doesn't really make sense to plot them in same figure, cause they have a different x-axis!!!
# # plot autocorrelation of signal
# fig, (ax1) = plt.subplots(1)
# ax1 = plt.plot(times, autocorrs, 'b')
# ax1 = plt.plot(times, sig_osc, 'r')
# plt.xlabel("Signal")
# plt.ylabel("Time")
# plt.title("Oscillation Plot")
# plt.savefig("oscillation_plot_1.png")  # Save figure to file

# # plot osc signal & autocorr
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(times, sig_osc, 'r')
# ax2.plot(times, autocorrs, 'b')
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Signal")
# ax1.set_title("Sinusoidal Time Series")
# ax2.set_xlabel("Lags")
# ax2.set_ylabel("Autocorr")
# plt.tight_layout()
# plt.savefig("oscillation_plot_2.png")  


# # plot power law signal and autocorr
# timepoints, autocorrs2 = compute_autocorr(sig_powerlaw)
# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(times, sig_powerlaw, 'r')
# ax2.plot(times, autocorrs2, 'b')
# ax1.set_xlabel("Time")
# ax1.set_ylabel("Signal")
# ax1.set_title("Powerlaw Signal")
# ax2.set_xlabel("Lags")
# ax2.set_ylabel("Autocorr")
# ax2.set_title('Autocorrelation of Power Law')
# plt.tight_layout()
# plt.savefig("oscillation_plot_3.png")  


# fig = plt.figure()
# fig.suptitle("Gridspec Figure")

# gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[4, 1])
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# ax3 = fig.add_subplot(gs[2])
# ax4 = fig.add_subplot(gs[3])
# ax1.plot(times, sig_osc, 'r')
# ax2.plot(times, autocorrs, 'b')
# ax3.plot(times, sig_powerlaw, 'r')
# ax4.plot(times, autocorrs2, 'b')
# plt.savefig("oscillation_plot_4.png")  



# # compute autocorrelation using scipy.signal.correlate
# # FIXME: this doesn't make sense right now
# autocorr = sp.signal.correlate(sig_osc, sig_osc + 1/fs)


# # TODO implement autocorrelation from scratch
# def compute_autocorr(signal, fs):
#     # not sure if this works
#     mean_signal = np.mean(signal)

#     # I guess the lags should be somehow dependent on srate of the signal
#     # lags = 

#     # for lag in lags:
#     #     signal * signal + lag



##################################################################################################################################

# simulate different kinds of signals 


# normal oscillations at different fundamental frequencies
freqs = [10, 12, 15, 20, 25, 30]
signals_osc = []
signals_burst = []

for f in freqs:

    # simulate normal oscillatory signals
    sig_osc = neurodsp.sim.sim_oscillation(n_seconds, fs, f)
    signals_osc.append(sig_osc)

    # simulate bursty oscillatory signals
    # I still need to specify the burst-definition!!
    sig_burst = neurodsp.sim.sim_bursty_oscillation(n_seconds, fs, f)
    signals_burst.append(sig_burst)

    # simulate variable oscillation (that varies in frequency and cycle parameters)


print(len(signals_osc)) 

# plot the simulated signals
times = create_times(n_seconds, fs)

fig, axes = plt.subplots(2, 3)

for ax, sig in zip(axes.ravel(), signals_osc):
    plot_time_series(times, sig, ax = ax)

fig.suptitle('Simulated Osc Signals')

plt.tight_layout()

plt.savefig(os.path.join(DERIV_DIR, 'timescales', 'plots', 'simulations', 'sim_osc.png'))



# plot slider of simulated oscillatory signal
# here freq is my free parameter
# define simulation function to be plotted
## FIXME: cause plot_timeseries doesn't return anything!!
# def sim_f(n_seconds, fs, freq):
#     return neurodsp.sim.sim_oscillation(n_seconds, fs, freq)

# times = create_times(n_seconds, fs)

# # define initial freq
# init_freq = 2.0

# # create the fig
# fig, ax = plt.subplots()
# line, = plot_time_series(times, sim_f(n_seconds, fs, init_freq), ax = ax)

# # adjust the main plot to make room for sliders
# fig.subplots_adjust(left=0.25, bottom=0.25)

# # make a horizontal slider to control the frequency
# axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
# freq_slider = Slider(
#     ax=axfreq,
#     label='Frequency [Hz]',
#     valmin=0.1,
#     valmax=30,
#     valinit=init_freq,
# )

# # the function to be called anytime a slider's value changes
# def update(val):
#     line.set_ydata(sim_f(n_seconds, fs, freq_slider.val))
#     fig.canvas.draw_idle()

# # register the update function with each slider
# freq_slider.on_changed(update)

# plt.show()



## Simulating Multiple Signals

# the sim_multiple() function can be used to simulate multiple signals from the same set of parameters
params = {'n_seconds': 5, 'fs':300, 'exponent': -1, 'f_range': [0.5, None]}

# simulate multiple iterations from the same parameter definition
sigs = sim_multiple(sim_powerlaw, params, 3)
# outputs a Simulations object (class) that stores multiple simulated signals along with relevant metadata

# create a times definition corresponding to the simulations
times = create_times(params['n_seconds'], params['fs'])

# plot the simulated signals (but what kind of python object is this?)
# fig, ax = plt.subplots()

# # I don't like the ylabel and the white space in the figure before signal begins
# fig = plot_multi_time_series(times, sigs, ax=ax)
# for x in [1, 2, 3, 4]:
#     ax.axvline(x=x, ymin=0, ymax=1, ls = '--', c = 'b', lw = 2)
# plt.show()

# None Type (cause it doesn't return anything) - maybe if I save it?
# print(type(fig))

# compute autocorrelation for the signals
for sig in sigs:
    # compute timepoints & autocorr coeff
    timepoints, autocorr = compute_autocorr(sig)

colors = ['darkgreen', 'darkblue', 'darkred']

fig, axes = plt.subplots(3, 1)

for ax, sig, col in zip(axes.ravel(), sigs, colors):
    timepoints, autocorr = compute_autocorr(sig)
    plot_autocorr(timepoints, autocorr, ax=ax, colors = col)
    
    ax.set(xlabel='Lag (Samples)', ylabel='Autocorr')

fig.suptitle('Autocorrelation for Simulated Signals')
plt.tight_layout()
plt.show()


# simulate different osc and aperiodic signals

# aperiodic activity is simulated as a power law time series, with a specified exponent
# brown noise - exponent = -2 (does the sign matter??)
sig_noise_brown = sim_powerlaw(n_seconds, fs, exponent = -2)
sig_noise_white = sim_powerlaw(n_seconds, fs, exponent = 0)
sig_noise_pink = sim_powerlaw(n_seconds, fs, exponent = -1)



