#!/bin/bash

# middleman script for submitting analysis to HPC cluster
# activates right conda environment
# runs script: input 1 is the script, input 2 is the subject, as inherited from
# the runner file.

source activate neural_timescales
eval python $1 $2