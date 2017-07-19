#!/bin/bash

# Root directory of the experiment and subfolders
ROOT_DIR=$1
echo "Running experiment set up in the folder $1"

# Suffixes for various types of files
MOD_SUFFIX="-mod.cfg"
OPT_SUFFIX="-opt.cfg"
EVA_SUFFIX="-eva.cfg"

# Check if datasets folder exists
DATASETS_DIR="$ROOT_DIR/datasets"
if [ ! -d $DATASETS_DIR ]
then
    echo "Folder with list of datasets - $DATASETS_DIR, doesn't exist."
    exit $?
fi
echo "Folder with list of datasets - $DATASETS_DIR, exists."

# Check if results folder exists, if not, create it
RESULTS_DIR="$ROOT_DIR/results"
if [ ! -d $RESULTS_DIR ]
then
    echo "Folder for results - $RESULTS_DIR, doesn't exist. Created it."
    mkdir $RESULTS_DIR
else
    echo "Folder for results - $RESULTS_DIR, exists."
fi

# Check if model configuration folder exists
MOD_CFG_DIR="$ROOT_DIR/models"
if [ ! -d $MOD_CFG_DIR ]
then
    echo "Folder with model configurations - $MOD_CFG_DIR, doesn't exist."
    exit $?
fi

# Check if optimizer configuration folder exists
OPT_CFG_DIR="$ROOT_DIR/optimizers"
if [ ! -d $OPT_CFG_DIR ]
then
    echo "Folder with optimizer configurations - $OPT_CFG_DIR, doesn't exist."
    exit $?
fi

# Check if evaluation configuration folder exists
EVA_CFG_DIR="$ROOT_DIR/evaluators"
if [ ! -d $EVA_CFG_DIR ]
then
    echo "Folder with evaluator configurations - $EVA_CFG_DIR, doesn't exist."
    exit $?
fi

# Check if there exist model configuration files for each dataset or
# alternatively one file for all datasets.
SAME_MOD_CFG=0
if [ -f "$MOD_CFG_DIR/all$MOD_SUFFIX" ]
then
    echo "Using the same models specified in $MOD_CFG_DIR/all$MOD_SUFFIX with 
          all datasets."
    SAME_MOD_CFG=1
else
    for dfile in `ls -L $DATASET_DIR | egrep '*-dataset.txt'`
    do
        dname=`sed -s s/-dataset.txt// <<< $dfile`
        mfile="$MOD_CFG_DIR/$dname$MOD_SUFFIX"

        if [ ! -f $mfile ]
        then
            echo "Model configuration file for $dname dataset doesn't exist. 
                  Exiting..." 
            exit $?
        fi
    done
fi

# Check if there exist optimization configuration files for each dataset or
# alternatively one file for all datasets.
SAME_OPT_CFG=0
if [ -f "$OPT_CFG_DIR/all$OPT_SUFFIX" ]
then
    echo "Using the same optimizers specified in $OPT_CFG_DIR/all$OPT_SUFFIX 
          with models for all datasets."
    SAME_OPT_CFG=1
else
    for dfile in `ls -L $DATASET_DIR | egrep '*-dataset.txt'`
    do
        dname=`sed -s s/-dataset.txt// <<< $dfile`
        ofile="$OPT_CFG_DIR/$dname$OPT_SUFFIX"

        if [ ! -f $mfile ]
        then
            echo "Optimizer configuration file for $dname dataset doesn't exist 
                  Exiting..."
            exit $?
        fi
    done
fi

# Check if there exist evaluation configuration files for each dataset or
# alternatively one file for all datasets.
SAME_EVA_CFG=0
if [ -f "$EVA_CFG_DIR/all$EVA_SUFFIX" ]
then
    echo "Using the same evaluators specified in $EVA_CFG_DIR/all$EVA_SUFFIX 
          with models for all datasets."
    SAME_EVA_CFG=1
else
    for dfile in `ls -L $DATASET_DIR | egrep '*-dataset.txt'`
    do
        dname=`sed -s s/-dataset.txt// <<< $dfile`
        efile="$EVA_CFG_DIR/$dname$EVA_SUFFIX"

        if [ ! -f $efile ]
        then
            echo "Evaluation configuration file for $dname dataset doesn't
                  exist. Exiting..."
            exit $?
        fi
    done
fi

# for dtst in ${DATASETS[@]} # Loop condition when manually listing datasets

# AND FINALLY train and test models over all the datasets
for dfile in `ls -L $DATASETS_DIR | egrep '*-dataset.txt'`
do
    dname=`sed -s s/-dataset.txt// <<< $dfile`
    echo "Dataset: $dname"
    dat_file="$DATASETS_DIR/$dfile" 
    
    if [ $SAME_MOD_CFG -eq 1 ]
    then
        mod_file="$MOD_CFG_DIR/all$MOD_SUFFIX"
    else
        mod_file="$MOD_CFG_DIR/$dname$MOD_SUFFIX"
    fi

    if [ $SAME_OPT_CFG -eq 1 ]
    then
        opt_file="$OPT_CFG_DIR/all$OPT_SUFFIX"
    else
        opt_file="$OPT_CFG_DIR/$dname$OPT_SUFFIX"
    fi

    if [ $SAME_EVA_CFG -eq 1 ]
    then
        eva_file="$EVA_CFG_DIR/all$EVA_SUFFIX"
    else
        eva_file="$EVA_CFG_DIR/$dname$EVA_SUFFIX"
    fi

    echo "./train_models.py -d $dat_file -m $mod_file -o $opt_file"
    python2 train_models.py -d $dat_file -m $mod_file -o $opt_file
    echo "./test_models.py -d $dat_file -e $eva_file"
    python2 test_models.py -d $dat_file -e $eva_file
done

# TODO 
# ----
# * Output a summary of various experiment variables, parameters and 
#   hyperparameters. Ask for user approval before proceeding.
# * Write functions for those operations which are repeated for each of the
#   model, optimizer and evaluator.

# XXX: I'm keeping this just as a reference for the declaration of and
# iteration through BASH arrays.
#
# A list of datasets
#declare -a DATASETS=('nova-scotia' 'bach' 'elsass' 'jugoslav' 'schweiz' \
#    'oesterrh' 'kinder' 'shanxi')
#
#for dtst in ${DATASETS[@]} # WTF syntax?
#do
#    dataset_file="$ROOT/datasets/$dtst$DATASET_FILE_SUFFIX"
#    config_file="$ROOT/models/$CONFIG_FILE_PREFIX$dtst$CONFIG_FILE_SUFFIX"
#    python train_models.py -d $dataset_file -o $ROOT/pretrain.cfg -m $config_file
#    python test_models.py -d $dataset_file -e $ROOT/evaluate-semi-online.cfg
#done

