#!/bin/bash/sh
# script to run:
my_script=$PWD/preprocessing_cluster_function.py


# submit for all subjects:
for subj in $(<subjectlist.txt); do # in Verzeichnis setzen lassen, und parameter: all oder einzelnd/liste
    echo "$PWD/qsub_preprocessing.sh $my_script $subj" | sbatch --job-name=preprocessing_test_$subj --time=18:00:00 --mem=15gb
done
