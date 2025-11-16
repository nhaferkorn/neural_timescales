#!/bin/bash
# script to run:
my_script=$PWD/preprocessing_cluster_function.py


# submit for all subjects:
for subj in $(<subjectlist.txt); do # in Verzeichnis setzen lassen, und parameter: all oder einzelnd/liste
  echo "RUN sbatch $PWD/slurm_preprocessing.sh $my_script $subj" 
  sbatch $PWD/slurm_preprocessing.sh $my_script $subj 
done
