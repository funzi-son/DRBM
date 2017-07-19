#!/bin/bash

## A list of datasets
#declare -a DATASETS=('nova-scotia' 'bach' 'elsass' 'jugoslav' 'schweiz' \
#    'oesterrh' 'kinder' 'shanxi')

# Root directory of the experiment and subfolders
ROOT_DIR=$1
echo "Saving results for experiment in the folder $1"

# Check if datasets folder exists
DATASETS_DIR="$ROOT_DIR/datasets"
if [ ! -d $DATASETS_DIR ]
then
    exit $?
fi
echo "Folder with list of datasets - $DATASETS_DIR, exists."

# Check if results folder exists, if not, create it
RESULTS_DIR="$ROOT_DIR/results"
if [ ! -d $RESULTS_DIR ]
then
    echo "Folder for results - $RESULTS_DIR, does not exist. Created it."
    mkdir $RESULTS_DIR
else
    echo "Folder for results - $RESULTS_DIR, exists."
fi

# Suffix for the file containing results
RESULTS_SUFFIX="-results.txt"

# for dtst in ${DATASETS[@]} # Loop condition when manually listing datasets

# Iterate over the list of datasets
for dfile in `ls -L $DATASETS_DIR | egrep '*-dataset.txt'`
do
    dname=`sed -s s/-dataset.txt// <<< $dfile`
    echo "Dataset: $dname"
    dataset_file="$DATASETS_DIR/$dfile" 
    output_file="$RESULTS_DIR/$dname$RESULTS_SUFFIX"
    echo "python save_results.py -d $dataset_file -o $output_file" # Check...
    python save_results.py -d $dataset_file -o $output_file
done
