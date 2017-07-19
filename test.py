"""Functions to train and evaluate models.

This file contains the following functions:
    train_sequence_model
      Do a fold-wise training and evaluation of a sequence prediction model.

    evaluate_grid_point

    grid_search

    best_grid_point

"""


import numpy as np
import os

from evaluate import error
from evaluate import negative_log_likelihood
from IO import load_model
from IO import update_model
from models import PredictionModel 
from utils import dict_product

def evaluate_models(eva_hypers, data, root):
    """Evaluate all the models saved in a folder on given data.

    Input
    -----
    eval_hypers : dict
      Various hyperparameters pertaining to evaluation.
    data : dict
      Inputs, labels and training/test fold information.
    root : string
      Path to folder which contains the models.

    Output
    ------
    Each model in the folder root is evaluated, and the previously saved model
    file is updated with the results.
    """
    # Iterate over all models learned for the dataset specified by root
    for fn in os.listdir(root):
        if fn.startswith('model') and fn.endswith('.pkl.gz') and not 'eval' in fn:
            print "Evaluating model saved in:\n\t%s\n" % (fn)

            # Generate and iterate over evaluation hyperparameter grid
            eva_hyper_grid = dict_product(eva_hypers)
            for eva_hyper in eva_hyper_grid:
                model = load_model(os.path.join(root, fn))
                # XXX: Each of the below returned variables will be a 2-tuple
                # in case of online and semi-online evaluation as they'll also 
                # include validation scores, predictions and probabilities.
                predictions, probabilities, scores = \
                    evaluate_one_model(model, data, eva_hyper) 
                model['eva_hypers'] = eva_hyper

                if eva_hyper['method'] == 'offline':
                    model['probabilities'] = [probability for probability in
                                              probabilities]
                    model['predictions'] = [prediction for prediction in
                                            predictions]
                    model['test_offline'] = scores
                elif eva_hyper['method'] == 'online':
                    # Replace offline validation scores by the online ones
                    model['probabilities'] = [probability[1] for probability in
                                              probabilities]
                    model['predictions'] = [prediction[1] for prediction in
                                            predictions]
                    model['validation_offline'] = model['validation']
                    model['validation'] = [score[0][0] for score in scores]
                    model['test_online'] = [(score[0][1], score[1][1]) 
                                            for score in scores]
                elif eva_hyper['method'] == 'semi-online':
                    # Replace offline validation scores by the semi-online ones
                    model['probabilities'] = [probability[1] for probability in
                                              probabilities]
                    model['predictions'] = [prediction[1] for prediction in
                                            predictions]
                    model['validation_offline'] = model['validation']
                    model['validation'] = [score[0][0] for score in scores]
                    model['test_semionline'] = [(score[0][1], score[1][1]) 
                                                for score in scores]
                else:
                    assert False, 'unknown evaluation type'

                model = update_model(model, os.path.join(root, fn))

    print "Done with evaluation."


def evaluate_one_model(model, data, eva_hypers): 
    """Evaluate one point in the grid search.

    Input
    -----
    model : dict
      A dictionary containing model (and its optimization) hyperparameters,
      parameters, validation scores.
    data : dict
      Contains inputs, labels and training/test fold information.
    eva_hypers : dict
      Dictionary containing evaluation hyperparameters.

    Output
    -----
    scores : tuple(float)
      Test scores.
    """
    # NOTE: It might not make sense why this function exists, but it is 
    # required when incorporating both sequence and non-sequence models 
    # within this framework.
    if model['mod_hypers']['model_type'] in ('rnn', 'rnndrbm', 'rnnnade',
                                             'rtdrbm'): 
        return evaluate_sequence_model(model, data, eva_hypers) 
    elif model['mod_hypers']['model_type'] in ('hdrbm', 'rbm', 'drbm'):
        return evaluate_windowed_model(model, data, eva_hypers) 
    else:
        assert False, 'unknown model type'


def evaluate_sequence_model(model, data, eva_hypers):
    """Do a fold-wise training and evaluation of a sequence model.

    Input
    -----
    model : dict
      A dictionary containing model (and its optimization) hyperparameters,
      parameters, validation scores.
    data : dict
      Contains inputs, labels and training/test fold information.
    eva_hypers : dict
      Dictionary containing evaluation hyperparameters.

    Output
    ------
    scores: list(tuple)
      A list of test scores, one tuple per fold.
    """
    # Data check
    n_folds = len(data['X']['train'])

    # Data hyperparameters
    dat_hypers = {}
    dat_hypers['n_input'] = data['X']['train'][0][0].shape[1]
    dat_hypers['n_class'] = data['n_classes']
   
    # Model hyperparameters
    mod_hypers = model['mod_hypers']

    scores = []
    probabilities = []
    predictions = []
    for fld in xrange(n_folds):
        print "\nFold %d" % (fld+1)
        
        # Prepare test data for this fold
        x_test = data['X']['test'][fld]
        try:  # Supervised learning
            y_test = data['y']['test'][fld]
            y_test = [y_te[np.newaxis] if len(y_te.shape) == 0 else y_te 
                      for y_te in y_test]
            test_data = (x_test, y_test)
        except KeyError:  # Unsupervised learning
            test_data = (x_test,)

        dat_hypers['n_test'] = len(x_test)

        # Load the appropriate model
        model_params = model['models'][fld]
        mod = PredictionModel(mod_hypers, dat_hypers['n_input'],
                              dat_hypers['n_class'], init_params=model_params)

        # Evaluate cross-entropy and error scores
        # TODO: Read this from the evaluation hyperparameters. It would involve
        # abstracting the score lists as well - they will no longer be
        # "probabilities" and "predictions".
        y_pred = []
        for i in xrange(dat_hypers['n_test']):
            y_pred.append(mod.predict_function(x_test[i]))
        test_nll = negative_log_likelihood(
            np.concatenate(tuple(y_pred), axis=0),
            np.concatenate(tuple(y_test), axis=0))
        test_acc = error(np.concatenate(tuple(y_pred), axis=0),
                         np.concatenate(tuple(y_test), axis=0))

        print "Test negative log-likelihood (offline): %.3f\n" % (test_nll)
        print "Test error (offline): %.3f\n" % (test_acc)
        
        probabilities.append(np.concatenate(tuple(y_pred), axis=0))
        predictions.append(np.argmax(np.concatenate(tuple(y_pred), axis=0), 
                                     axis=-1))
        scores.append((test_nll, test_acc))

    return predictions, probabilities, scores


def evaluate_windowed_model(model, data, eva_hypers):
    """Do a fold-wise training and evaluation of a sequence prediction model.

    Input
    -----
    model : dict
      A dictionary containing model (and its optimization) hyperparameters,
      parameters, validation scores.
    data : dict
      Contains inputs, labels and training/test fold information.
    eva_hypers : dict
      Dictionary containing evaluation hyperparameters.

    Output
    ------
    scores: list(tuple)
      A list of test scores, one tuple per fold.
    """
    # Data check
    n_folds = len(data['X']['train'])
    
    # Data hyperparameters
    dat_hypers = {} # Note that I don't need the variable "data" anymore
    dat_hypers['n_input'] = data['X']['train'][0].shape[1]
    dat_hypers['n_class'] = data['n_classes']

    # Model hyperparameters
    mod_hypers = model['mod_hypers']

    scores = []
    probabilities = []
    predictions = []
    for fld in xrange(n_folds):
        print "\nFold %d" % (fld+1)
        
        # Prepare test data for this fold
        x_test = data['X']['test'][fld]
        try: # Supervised learning
            y_test = data['y']['test'][fld]
            test_data = (x_test, y_test)
        except KeyError: # Unsupervised learning
            test_data = (x_test,)

        dat_hypers['n_test'] = x_test.shape[0]

        # Load the appropriate model
        model_params = model['models'][fld]
        mod = PredictionModel(mod_hypers, dat_hypers['n_input'],
                              dat_hypers['n_class'], init_params=model_params)

        # Cross entropy and error
        # TODO: Read this from the evaluation hyperparameters. It would involve
        # abstracting the score lists as well - they will no longer be
        # "probabilities" and "predictions".
        y_pred = mod.predict_function(x_test)
        test_nll = negative_log_likelihood(y_pred, y_test)
        test_acc = error(y_pred, y_test)

        print "Test negative log-likelihood (offline): %.3f\n" % (test_nll)
        print "Test error (offline): %.3f\n" % (test_acc)

        probabilities.append(y_pred)
        predictions.append(np.argmax(y_pred, axis=-1))

        scores.append((test_nll, test_acc))

    return predictions, probabilities, scores
