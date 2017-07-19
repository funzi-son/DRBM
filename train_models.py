#!/usr/bin/python

"""Code for training connectionist melody models.

This script is to be run with the following command-line options with their
respective arguments:
    -d  Dataset list file (See config-examples/datasets).
    -m  Model configuration file (See config-examples/models).
    -o  Optimizer configuration file (See config-examples/optimizers).
"""

import cPickle
import getopt
import gzip
import os
import sys

from collections import OrderedDict
from config import Config
from IO import model_exists
from IO import save_model
from train import train_one_model
from utils import dict_product
from utils import dict_prettyprint


def usage():
    """
    Print usage of script.
    """
    print ("Usage: \n\t$ train_models.py -m model.cfg -o optimizer.cfg "
           "-d dataset-list.txt [-O]\n")


def grid_search(mod_hypers, opt_hypers, data, root, overwrite_model=False):
    """Search a grid of hyperparameters for the best model on given data. This
    can also be used to train a single model.

    Input
    -----
    mod_hypers : dict
      Model hyperparameters (as keys) and their values.
    opt_hypers : dict
      Optimization hyperparameters (as keys ) and their values.
    data : dict
      Contains inputs, labels and training/test fold information.
    root : string
      Path to folder in which to save the models.
    overwrite_model : bool
      Whether or not to overwrite a model if there already exists a file
      containing saved parameters corresponding to it.

    Output
    ------
    Each learned model, along with its hyperparameters and those of the
    optimizer and best validation scores, is saved to its own file in a folder
    named 'models' inside the folder specified by root.
    """
    # Carry out grid search
    mod_hyper_grid = dict_product(mod_hypers)
    for mod_hyper in mod_hyper_grid:
        opt_hyper_grid = dict_product(opt_hypers)
        for opt_hyper in opt_hyper_grid:
            if model_exists(mod_hyper, opt_hyper, root) \
                    and overwrite_model == False:
                # Don't re-learn if an identical model if it already exists
                print ("A saved file corresponding to the following (model and"
                       " optimization) hyperparameters exists. Skipping...")
                dict_prettyprint(mod_hyper, 'Model Hyperparameters')
                dict_prettyprint(opt_hyper, 'Optimiser Hyperparameters')
            else:
                print ("Learning a model with the following (model and "
                       "optimization) hyperparameters")
                dict_prettyprint(mod_hyper, 'Model Hyperparameters')
                dict_prettyprint(opt_hyper, 'Optimiser Hyperparameters')

                # Learn a model with one set of hyperparameters.
                grid_point = train_one_model(mod_hyper, opt_hyper, data)
                save_model(grid_point, root)

    print "Done with grid search.\n"


if __name__ == "__main__":
    try:
        CL_OPTARGS, _ = getopt.getopt(sys.argv[1:], 'd:hm:o:O', ['help'])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    # Parse command-line options and arguments
    overwrite_model = False
    for o, a in CL_OPTARGS:
        if o in ('-d', '--dataset'):  # Dataset list file
            assert os.path.isfile(a), ("specified dataset file %s doesn't "
                                       "exist" % (a))
            dataset_file = a
            f_dataset = open(dataset_file, 'r')
        elif o in ('-m', '--model'):  # Model configuration file
            assert os.path.isfile(a), ("specified config file %s doesn't "
                                       "exist" % (a))
            config_file = a
            f_config = file(config_file)
            mod_cfg = OrderedDict(Config(f_config))  # Duck-typing
        elif o in ('-o', '--optimiser'):  # Optimizer configuration file
            assert os.path.isfile(a), ("specified config file %s doesn't "
                                       "exist" % (a))
            config_file = a
            f_config = file(config_file)
            opt_cfg = OrderedDict(Config(f_config))  # Duck-typing
        elif o == '-O': # Overwrite saved model file
            overwrite_model = True
        elif o in ('-h', '--help'):  # Help!
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    # Iterate over the various datasets
    for dataset in f_dataset.readlines():
        dataset = dataset.strip()
        out_path = os.path.dirname(dataset)

        # Load data dictionary
        data_dict = cPickle.load(gzip.open(dataset, 'rb'))
        print "Dataset: %s\n" % (data_dict['name'])

        # Learn and save model(s), i.e, save the best model and optimization
        # hyperparameters, and their respective validation scores.
        grid_search(mod_cfg, opt_cfg, data_dict,
                    os.path.join(out_path, 'models'),
                    overwrite_model=overwrite_model)

    f_dataset.close()
    print "End of program."
