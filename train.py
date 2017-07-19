"""Functions to train and evaluate models.

This file contains the following functions:
    train_one_model : Evaluate one point in the grid search.

    train_sequence_model : Do a fold-wise training and evaluation of a sequence
        prediction model.
"""


import numpy as np
from models import PredictionModel 
from utils import make_batches
from utils import truncate_sequences


def initialise_optimiser(opt_hypers, model):
    """Load optimiser class given the name.

    Input
    -----
    opt_type : dict
      Optimiser configuration dictionary.

    Output
    ------
     : class instance
      Instance of the optimiser class.

    """
    if opt_hypers['opt_type'] == 'gd':
        from optimize import gradient_descent
        return gradient_descent(model, opt_hypers)
    elif opt_hypers['opt_type'] == 'adadelta':
        from optimize import adadelta
        return adadelta(model, opt_hypers)
    elif opt_hypers['opt_type'] == 'rmsprop':
        from optimize import rmsprop
        return rmsprop(model, opt_hypers)
    else:
        raise NotImplementedError('unknown optimiser type')


def train_one_model(mod_hyper, opt_hyper, data, params=None):
    """
    Evaluate one point in the grid.

    Input
    -----
    mod_hyper : dict
      Model hyperparameters (as keys) and their values.
    opt_hyper : dict
      Optimizer hyperparameters (as keys) and their values.
    data : dict
      A dictionary containing, inputs, labels and fold information.
    params : tuple(np.ndarray) [optional]
      Initial values of model parameters.

    Output
    -----
    grid_point : dict
      A dictionary containing learning hyperparameters, scores and model
      parameters.
    """
    if mod_hyper['model_type'] in ('rnn', 'rnndrbm', 'rnnnade', 'rtdrbm'):
        params, scores = train_sequence_model(mod_hyper, opt_hyper, data,
                                              params)
    elif mod_hyper['model_type'] in ('drbm', 'rbm', 'hdrbm'):
        params, scores = train_nonsequence_model(mod_hyper, opt_hyper, data,
                                                 params)
    else:
        assert False, 'unknown model type'

    # Populate learning results dictionary
    grid_point = {}
    grid_point['mod_hypers'] = mod_hyper
    grid_point['opt_hypers'] = opt_hyper
    grid_point['models'] = params
    grid_point['validation'] = scores

    return grid_point


def train_sequence_model(mod_hypers, opt_hypers, data, init_params=None):
    """Do a fold-wise training and evaluation of a sequence model.

    Input
    -----
    mod_hyper : dict
      Model hyperparameters (as keys) and their values.
    opt_hyper : dict
      Optimizer hyperparameters (as keys) and their values.
    data : dict
      A dictionary containing, inputs, labels and fold information.
    init_params : tuple(np.ndarray) [optional]
      Initial values of model parameters.

    Output
    ------
    models: list(list)
      A list of lists of model parameters, one list per fold.
    scores: list(tuple)
      A list of validation score tuples, one tuple per fold.
    """
    # Data check
    n_folds = len(data['X']['train'])

    # Data hyperparameters
    dat_hypers = {}
    try: # Supervised learning
        dat_hypers['n_input'] = data['X']['train'][0][0].shape[1]
        dat_hypers['n_class'] = data['n_classes']
    except KeyError: # Unsupervised learning
        dat_hypers['n_input'] = data['n_dims']
        dat_hypers['n_class'] = None

    params = []
    scores = []
    for fld in xrange(n_folds):
        print "\nFold %d" % (fld+1)

        # Prepare training and validation data for this fold
        x_train = data['X']['train'][fld]
        x_valid = data['X']['verify'][fld]
        try:  # Supervised learning
            y_train = data['y']['train'][fld]
            y_valid = data['y']['verify'][fld]

            # NOTE: This step is to augment a single-element 1d-array into a
            # 2d-array. It tends to cause problems later on in the evaluation
            # stage due to the np.squeeze() function which squeezes 1d-arrays
            # with a single element into a scalar.
            # FIXME: Need to fix the squeezing of unit-length sequences
            y_train = [y_tr[np.newaxis] if len(y_tr.shape) == 0 else y_tr
                       for y_tr in y_train]
            y_valid = [y_va[np.newaxis] if len(y_va.shape) == 0 else y_va
                       for y_va in y_valid]

            dataset = ((x_train, y_train), (x_valid, y_valid))
        except KeyError:  # Unsupervised learning
            dataset = ((x_train,), (x_valid,))

        dat_hypers['n_train'] = len(data['X']['verify'][fld])
        dat_hypers['n_valid'] = len(data['X']['train'][fld])

        # Update training data by truncating sequences as required
        try:
            x_train, y_train, _ = truncate_sequences(
                x_train, y=y_train, seq_len=opt_hypers['batch_size'])
            dataset = ((x_train, y_train), (x_valid, y_valid))
        except UnboundLocalError:
            x_train, _ = truncate_sequences(
                x_train, y=None, seq_len=opt_hypers['batch_size'])
            # TODO: Consider unsupervised learning sequence truncation as well
            dataset = ((x_train,), (x_valid,))


        # Import model class as required
        # TODO: This does not take into consideration the possibility of
        # passing initial parameters, which is relevant particularly to
        # re-training and pre-training models.
#        model = initialise_model(mod_hypers, dat_hypers, init_params=None)
        model = PredictionModel(mod_hypers, dat_hypers['n_input'],
                                dat_hypers['n_class'], init_params=None)

        # Load optimiser
        optimser = initialise_optimiser(opt_hypers, model)

        # Load optimiser
        # Train model
        param, score = optimser.optimize(dataset)

        params.append(param)
        scores.append(score)

    return params, scores


def train_nonsequence_model(mod_hypers, opt_hypers, data, init_params=None):
    """Do a fold-wise training and evaluation of a nonsequence prediction
    model.

    Input
    -----
    mod_hyper : dict
      Model hyperparameters (as keys) and their values.
    opt_hyper : dict
      Optimizer hyperparameters (as keys) and their values.
    data : dict
      A dictionary containing, inputs, labels and fold information.
    init_params : tuple(np.ndarray) [optional]
      Initial values of model parameters.

    Output
    ------
    models: list(list)
      A list of lists of model parameters, one list per fold.
    scores: list(tuple)
      A list of validation score tuples, one tuple per fold.
    """
    # Data check
    n_folds = len(data['X']['train'])

    # Data hyperparameters
    dat_hypers = {}
    try: # Supervised learning
        dat_hypers['n_input'] = data['X']['train'][0].shape[1]
        dat_hypers['n_class'] = data['n_classes']
    except KeyError: # Unsupervised learning
        dat_hypers['n_input'] = data['n_dims']
        dat_hypers['n_class'] = None

    params = []
    scores = []
    for fld in xrange(n_folds):
        print "\nFold %d" % (fld+1)

        # Prepare training and validation data for this fold
        x_train = data['X']['train'][fld]
        x_valid = data['X']['verify'][fld]
        try:  # Supervised learning
            y_train = data['y']['train'][fld]
            y_valid = data['y']['verify'][fld]

            dataset = ((x_train, y_train), (x_valid, y_valid))
        except KeyError:  # Unsupervised learning
            dataset = ((x_train,), (x_valid,))

        dat_hypers['n_train'] = len(data['X']['train'][fld])
        dat_hypers['n_valid'] = len(data['X']['verify'][fld])

        # Split data into batches
        try:  # Supervised learning
            x_train, y_train = make_batches(x_train, y_train,
                                            opt_hypers['batch_size'])
            x_valid, y_valid = make_batches(x_valid, y_valid,
                                            x_valid.shape[0])
            dataset = ((x_train, y_train), (x_valid, y_valid))
        except UnboundLocalError:  # Unsupervised learning
            # TODO: Consider unsupervised learning batch-making as well
            dataset = ((x_train,), (x_valid,))

        # Import model class as required
        # TODO: This does not take into consideration the possibility of
        # passing initial parameters, which is relevant particularly to
        # re-training and pre-training models.
#        model = initialise_model(mod_hypers, dat_hypers)
        model = PredictionModel(mod_hypers, dat_hypers['n_input'],
                                dat_hypers['n_class'], init_params=None)

        # Load optimiser
        optimser = initialise_optimiser(opt_hypers, model)

        # Train model
        param, score = optimser.optimize(dataset)

        params.append(param)
        scores.append(score)

    return params, scores
