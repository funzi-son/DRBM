#!/usr/bin/python

"""Code for evaluating previously learned connectionist models for melody.

This script is to be run with the following command-line options with their
respective arguments:
    -e  Evaluation configuration file name.
    -d  Dataset list file name.

The EVALUATION CONFIGURATION FILE is parsed using the config Python module, and
must contain the following hyperparameter:
    method       : Whether the evaluation is to be 'online' or 'offline'.

... and if the evaluation method is 'online', the following hyperparameters
will also have to be specified:
    opt_type             : Optimizers type - stoch-gd or batch-gd
    learning_rate        : Learning rate.
    schedule             : The type of learning 
    rate_param           : The decay parameter to use in a learning rate
                           schedule.
    max_epoch            : Maximum number of training epochs.

and optionally, the following hyperparameters (not implemented yet):
    initial_momentum     : Momentum at the start of training.
    final_momentum       : Momentum to switch to during training.
    momentum_switchover  : Number of epochs after which to switch from initial
                           to final momentum.

The DATASET LIST FILE points to a set of files, each of which contains some
data to train the models. The data is contained in a dictionary saved in a
cPickle file, which has the following fields:
    X              : Input sequences (either a list or a Numpy ndarray)
    y              : Target sequences (either a list or a Numpy ndarray)
    fold_labels    : Fold labels for all the data-points (either a list or a
                     Numpy ndarray)
    n_classes      : Number of prediction classes
    softmax_config : Pattern of softmax groups (only applicable to the RBM and
                     the RTRBM)
    n_maps         : Essentially the context length (only applicable to the
                     NPMM, which hasn't yet been implemented)
"""


from collections import OrderedDict
from config import Config
import getopt
import gzip
import os
import cPickle
import sys
from test import evaluate_models


def usage():
    """
    Print usage of script.
    """
    print ("Usage: \n\t$ test_models.py -d dataset-list.txt "
           "-e evaluation-config.cfg\n") 


if __name__ == "__main__":
    try:  # Parse command-line arguments using getopt
        cl_optargs, _ = getopt.getopt(sys.argv[1:], 'd:e:h', ['help'])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    for o, a in cl_optargs:
        if o == '-d':  # Dataset list file
            assert os.path.isfile(a), "specified dataset file doesn't exist"
            dataset_file = a
            f_dataset = open(dataset_file, 'r')
        elif o == '-e':  # Evaluation configuration file
            assert os.path.isfile(a), "specified config file doesn't exist"
            config_file = a
            f_config = file(config_file)
            eva_cfg = OrderedDict(Config(f_config))  # Duck-typing
        elif o in ('-h', '--help'):  # Help!
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    # Iterate over the various datasets
    for dataset in f_dataset.readlines():
        dataset = dataset.strip()
        out_path = os.path.dirname(dataset)

        # Load data
        data = cPickle.load(gzip.open(dataset, 'rb'))
        data = dict((key, data[key]) for key in ('X', 'y', 'n_classes'))
        print "Dataset: %s\n" % (out_path)

        # Evaluate model(s) learned and saved on this dataset
        root_dir = os.path.join(out_path, 'models')
        if os.path.isdir(root_dir):
            evaluate_models(eva_cfg, data, root_dir)
        else:
            print "The directory %s doesn't exist. Skipping..." % (root_dir)

    f_dataset.close()
    print "End of program."
