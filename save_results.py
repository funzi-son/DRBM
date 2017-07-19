#!/usr/bin/python

import getopt
import gzip
from IO import load_model
import numpy as np
import os
import cPickle
import sys


def usage():
    """Print usage of script.
    """
    print "Usage: \n\t$ save_results.py -d dataset-list.txt -o output.txt\n"


def save_results(dataset_file, output_file):
    """Compile results of grid search.
       
       Input
       -----
       dataset_file: Name of file containing a list of datasets.

       Output
       ------
       A file - compiled-all-results.txt which contains contain details of the
       models and results at the best and all points
         (respectively) of the hyperparameter grid.
    """

    # Open input and output files.
    f_dataset = open(dataset_file, 'r')
    f_all_res= open(output_file, 'w')

    # Loop over datasets in the list.
    for dataset in f_dataset.readlines():
        dataset = dataset.strip()
        root_dir = os.path.dirname(dataset)

        all_scores, all_hypers, all_file_names = compile_results(root_dir)

        sort_order = np.argsort(np.asarray([scores['mean_valid_crosent'] 
                                            for scores in all_scores], 
                                           dtype=np.float))
        all_sorted_scores = [all_scores[idx] for idx in sort_order]
        all_sorted_hypers = [all_hypers[idx] for idx in sort_order]
        all_sorted_file_names = [all_file_names[idx] for idx in sort_order]
        
        # Print dataset name.
        f_all_res.write('\nDataset: %s\n' % (root_dir))

        for (scores, hypers, file_name) \
            in zip(all_sorted_scores, all_sorted_hypers, all_sorted_file_names):
            # Create an entry for the current model in the file containing
            # all results.
            entry = create_entry(scores, hypers[0], hypers[1], hypers[2])
            f_all_res.write('Model file: %s\n' % (file_name))
            f_all_res.write(entry)

    f_dataset.close()
    f_all_res.close()


def compile_results(root_dir):
    """Compile all results in a given folder.

    Input
    -----
    root_dir : str
      Path to folder containing data and learned models.

    Output
    ------
    all_scores : list
      A list of scores corresponding to all evaluated models.
    all_hypers : list
      A list of hyperparameters corresponding to the returned scores.

    """
    # Since our folds typically have unequal numbers of data points, we
    # weight fold-wise scores by the sizes of their respective folds.
    var_dict = cPickle.load(gzip.open(root_dir+'/data.pkl.gz', 'rb'))
#    folds = var_dict['fold_labels']
#    n_folds = np.unique(folds).shape[0]
    n_folds = len(var_dict['X']['test'])

    try: # Sequence data
        fold_weights = [sum([np.shape(seq)[0] 
                             for seq in var_dict['X']['test'][fld]])
                        for fld in xrange(n_folds)]
    except IndexError: # Non-sequence data
        fold_weights = [var_dict['X']['test'][fld].shape[0]
                        for fld in xrange(n_folds)]

    n_pts = sum(fold_weights)
    fold_weights = [float(f_w)/float(n_pts) for f_w in fold_weights]

    # Loop over the evaluated model files in the dataset folder.
    model_dir = os.path.join(root_dir, 'models')
    all_scores = [] # Test scores
    all_hypers = [] # Hyperparameters
    all_files = [] # File names
    for fn in os.listdir(model_dir):
        if fn.startswith('model') and fn.endswith('.pkl.gz') and 'eval' in fn:
            model = load_model(os.path.join(model_dir, fn))

            # Get test and validation scores for the model
            valid_scores = np.asarray(model['validation'])
            test_scores, test_errors = get_test_scores(model)
           
            # Compute weighted validation negative log-likelihood mean and std.
            scores = {}
            scores['mean_valid_crosent'] = np.average(valid_scores,
                                                      weights=fold_weights)
            scores['std_valid_crosent'] = \
                np.sqrt(np.average((valid_scores-scores['mean_valid_crosent'])**2,
                                   weights=fold_weights))
            
            # Compute weighted test negative log-likelihood mean and std.
            scores['mean_test_crosent'] = np.average(test_scores,
                                                     weights=fold_weights) 
            scores['std_test_crosent'] = \
                np.sqrt(np.average((test_scores-scores['mean_test_crosent'])**2,
                                   weights=fold_weights))

            # Compute weighted test error mean and deviation
            scores['mean_test_error'] = np.average(test_errors,
                                                   weights=fold_weights)
            scores['std_test_error'] = \
                np.sqrt(np.average((test_errors-scores['mean_test_error'])**2,
                                   weights=fold_weights))
            
            # Append current model scores to a list
            all_scores.append(scores)
            all_hypers.append((model['mod_hypers'], model['opt_hypers'],
                               model['eva_hypers']))
            all_files.append(fn)
            
    return all_scores, all_hypers, all_files


def get_test_scores(model):
    """Given a model, return its test scores.

    Input
    -----
    model : dict
      A trained and evaluated model dictionary containing all the saved
      information.

    Output
    ------
    test_scores : np.ndarray
      Fold-wise test cross entropies.
    test_errors : np.ndarray
      Fold-wise test errors.

    """
    assert model['eva_hypers']['method'] in ('offline', 'online',
                                             'semi-online'), \
        "Unknown evaluation method."
    if model['eva_hypers']['method'] == 'offline':
        test_scores = np.asarray(
            [score[0] for score in model['test_offline']])
        test_errors = np.asarray(
            [score[1] for score in model['test_offline']])
    elif model['eva_hypers']['method'] == 'online':
        test_scores = np.asarray(
            [score[0] for score in model['test_online']])
        test_errors = np.asarray(
            [score[1] for score in model['test_online']])
    elif model['eva_hypers']['method'] == 'semi-online':
        test_scores = np.asarray(
            [score[0] for score in model['test_semionline']])
        test_errors = np.asarray(
            [score[1] for score in model['test_semionline']])

    return test_scores, test_errors
 

def create_entry(scores, mod_hypers, opt_hypers, eva_hypers):
    """Create an entry for an evaluated model in the results file.

    Input
    -----
    scores : dict
      Model scores
    mod_hypers : dict
      Model hyperparameters
    opt_hypers : dict
      Optimization hyperparameters
    eva_hypers : dict
      Evaluation hyperparameters

    Output
    ------
    entry : string
      A file entry that has been generated using the input scores and
      hyperparameters.
    """
    entry = ''
    for (key, val) in scores.items():
        entry += (key + ' : ' + str(val) + ' :: ')

    for (key, val) in mod_hypers.items():
        entry += (key + ' : ' + str(val) + ' :: ')

    for (key, val) in opt_hypers.items():
        entry += (key + ' : ' + str(val) + ' :: ')

    for (key, val) in eva_hypers.items():
        entry += (key + ' : ' + str(val) + ' :: ')

    return entry[:-4] + '\n\n'


if __name__ == "__main__":
    try:
        cl_optargs, _ = getopt.getopt(sys.argv[1:], 'd:ho:', ['help'])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(2)

    for o, a in cl_optargs:
        if o == '-d':
            assert os.path.isfile(a), "specified dataset file doesn't exist"
            dataset_file = a
        elif o == '-o':
#            assert os.path.isfile(a), "specified dataset file doesn't exist"
            output_file = a
        elif o in ('-h', '--help'):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"

    save_results(dataset_file, output_file)

