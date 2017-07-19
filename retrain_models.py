"""Code for re-training and evaluating the best of a set of previously trained
connectionist models for melody.

This script is to be run with two command-line options with their respective
arguments. These are:
    -n  Number of additional training iterations.
    -d  Dataset list file name.

Since we start here with an already trained model which we would like to
further train, the -n option takes the number of additional training iterations
(an integer) as argument. Only the dataset file has to be specified in order
for the script to know where to look. Given this, the script re-trains all the
models that remain in the 'models' folder in that dataset's folder. I intend to
manually remove the ones I don't wish to re-train.

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
                     NPMM)
"""


from collections import OrderedDict
from config import Config
import getopt
from IO import load_model
from IO import model_exists
from IO import save_model
import numpy as np
import os
import cPickle
import sys
from train import train_one_model
from utils import dict_product


def usage():
    """
    Print usage of script.
    """
    print ("Usage: \n\t$ retrain_models.py -d dataset-list.txt "
           "-o optimizer.cfg\n")


def retrain_models(data, root, opt_cfg):
    """Retrain all the models saved in a folder on given data.

    Input
    -----
    data : dict
      Contains inputs, labels and training/test fold information.
    root : string
      Path to folder which contains the models.
    opt_cfg: dict 
      Optimizer configuration for re-training.

    Output
    ------
    Each model in the folder root is re-trained for the specified number of
    epochs, and a new file is written file is updated with the results.
    """
    # Iterate over all models for the dataset intended to be retrained
    for fn in os.listdir(root):
        if fn.startswith('model') and fn.endswith('.pkl') and not 'eval' in fn:
            print "Re-training model saved in:\n\t%s\n" % (fn)
            
            # Load previously learned model
            model = load_model(os.path.join(root, fn))
            mod_hyper = model['mod_hypers']
            
            # Re-train with the given optimization hyperparameters
            opt_hyper_grid = dict_product(opt_cfg)
            for opt_hyper in opt_hyper_grid:
                if model_exists(mod_hyper, opt_hyper, root):
                    # Don't re-learn if an identical model already exists
                    # XXX: This might skip a model that's supposed to be
                    # re-trained if all the re-training hyperparameters are the
                    # same as the original ones. Should be handled manually for
                    # now.
                    print ("A saved file corresponding to the following (model"
                           " and optimization) hyperparameters exists. "
                           "Skipping...")
                    print mod_hyper
                    print opt_hyper
                else:
                    print ("Learning a model with the following (model and "
                           "optimization) hyperparameters")
                    print mod_hyper
                    print opt_hyper

                    # Learn a model with one set of hyperparameters.
                    grid_point = train_one_model(mod_hyper, opt_hyper, data,
                                                 params=model['models'])
                    save_model(grid_point, root)

    print "Done with re-training."


if __name__ == "__main__":
    try:
        cl_optargs, _ = getopt.getopt(sys.argv[1:], 'hd:o:', ['help'])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    n_epoch = 50
    for o, a in cl_optargs:
        if o == '-d':
            assert os.path.isfile(a), "specified dataset file doesn't exist"
            dataset_file = a
            f_dataset = open(dataset_file, 'r')
        elif o == '-o': #  Optimizer configuration file
            assert os.path.isfile(a), ("specified config file %s doesn't "
                                       "exist" % (a))
            config_file = a
            f_config = file(config_file)
            opt_cfg = OrderedDict(Config(f_config)) # Duck-typing
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    # Iterate over the various datasets
    for dataset in f_dataset.readlines():
        dataset = dataset.strip()
        out_path = os.path.dirname(dataset)

        # Load data
        data = cPickle.load(open(dataset, 'rb'))
        data = dict((key, data[key]) for key in ('X', 'y', 'fold_labels'))
        print "Dataset: %s" % (out_path)

        # Evaluate model(s) learned and saved on this dataset
        root_dir = os.path.join(out_path, 'models')
        if os.path.isdir(root_dir):
            retrain_models(data, root_dir, opt_cfg)
        else:
            print "The directory %s doesn't exist. Skipping..." % (root_dir)

    f_dataset.close()
    print "End of program"



