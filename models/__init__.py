"""Class definition for all the models defined in this folder."""


# Author: Srikanth Cherla
# City University London (2014)
# Contact: abfb145@city.ac.uk


from IO import generate_file_name
import numpy as np
import theano
import theano.tensor as T
import time

theano.config.exception_verbosity = 'high'


# A set of userful debug functions
def list_isnan(obj):
    """Determine whether there is a nan in any variable in a list of objrices.

    Input
    -----
    obj : list(np.array)
        List of numpy arrays to check for nan in. This list can also contain
        more lists inside it.

    Output
    ------
    True if any elements of any of the numpy arrays inside the list(s) contain
    a nan.
    """
    if type(obj) == list:
        return np.any(np.isnan(np.asarray([list_isnan(o) for o in obj])))
    else:
        return np.any(np.isnan(obj))


def list_isinf(obj):
    """Determine whether there is a inf in any variable in a list of objrices.

    Input
    -----
    obj : list(np.array)
        List of numpy arrays to check for nan in. This list can also contain
        more lists inside it.

    Output
    ------
    True if any elements of any of the numpy arrays inside the list(s) contain
    a nan.
    """
    if type(obj) == list:
        return np.any(np.isinf(np.asarray([list_isinf(o) for o in obj])))
    else:
        return np.any(np.isinf(obj))


def list_norm(obj, degree=1):
    """Computes the norms of a given degree of a list of objrices.

    Input
    -----
    obj : list(np.array)
        List of numpy arrays to check for nan in. This list can also contain
        more lists inside it.

    Output
    ------
    True if any elements of any of the numpy arrays inside the list(s) contain
    a nan.
    """
    if type(obj) == list:
        return [list_norm(o, degree) for o in obj]
    else:
        return np.sum(np.abs(obj)**degree)


class PredictionModel(object):
    """General wrapper class for various prediction models"""

    def __init__(self, hypers, n_input, n_class=None, init_params=None):
        """Constructs and compiles Theano functions for learning and
        prediction.

        Input
        -----
        hypers : dictionary
          Model hyperparameters
        n_input : integer
          Number of inputs to the model
        n_class : integer
          Number of outputs (None if unsupervised)
        init_params : list
          Model initial parameters

        Class attributes
        ----------------
        self.hypers : dict
          Model hyperparameters
        self.n_input : int
          Number of inputs to the model
        self.n_class : int
          Number of outputs (None if unsupervised)
        self.x : T.matrix
          Symbolic variable of the model's inputs
        self.y : T.matrix
          Symbolic variable of the model's outputs
        self.y_pred : T.???
          Symbolic variable of the model's predictions
        self.cost : T.???
          Symbolic variable of the model's cost
        self.params : list(T.shared)
          List of model parameters (each a symbolic variable)
        self.grads : list (T.grad)
          List of model parameter gradients (each a symbolic variable)
        """
        self.hypers = hypers
        self.n_input = int(n_input)
        try:
            self.n_class = int(n_class)
        except TypeError:
            self.n_class = None
        self.uid = time.strftime('%Y-%m-%d-%H-%M-%S') + \
            generate_file_name('', hypers, '', '')

        # Build the model graph.
        (self.x, self.y, self.y_pred, self.cost, self.params, 
         self.grads) = build_prediction_model(n_input, n_class, 
                                              hypers, init_params)

        # Functions for prediction, parameters and gradients
        self.predict_function = theano.function([self.x], self.y_pred,
                                                allow_input_downcast=True)
        self.get_model_parameters = theano.function([], self.params)
        if self.y is not None:
            self.get_gradients = theano.function([self.x, self.y], self.grads,
                                                 allow_input_downcast=True)
        else:
            self.get_gradients = theano.function([self.x], self.grads,
                                                 allow_input_downcast=True) 

        
def build_prediction_model(n_input, n_class, hypers, init_params):
    """Build the specific model graph.

    Input
    -----
    n_input : integer
      Number of inputs to the model
    n_class : integer
      Number of outputs (None if unsupervised)
    hypers : dictionary
      Model hyperparameters
    init_params : list
      Model initial parameters

    Output
    ------
    x : T.matrix
      Symbolic variable of the model's inputs
    y : T.matrix
      Symbolic variable of the model's outputs
    y_pred : T.nnet.softmax
      Symbolic variable of the model's predictions
    cost : T.???
      Symbolic variable of the model's cost
    params : list(T.shared)
      List of model parameters (each a symbolic variable)
    grads : list (T.grad)
      List of model parameter gradients (each a symbolic variable)
    """
    if hypers['model_type'] == 'drbm':
        from drbm import build_model
    elif hypers['model_type'] == 'rbm':
        from rbm import build_model
    elif hypers['model_type'] == 'hdrbm':
        from hdrbm import build_model 
    elif hypers['model_type'] == 'rnn':
        from rnn import build_model 
    elif hypers['model_type'] == 'rnndrbm':
        from rnndrbm import build_model 
    elif hypers['model_type'] == 'rnnnade':
        from rnnnade import build_model 
    elif hypers['model_type'] == 'rtdrbm':
        from rtdrbm import build_model 
    else:
        raise NotImplementedError('unknown model type')

    try:
        return build_model(n_input=n_input, n_class=n_class, hypers=hypers,
                           init_params=init_params)
    except TypeError:
        return build_model(n_input=n_input, hypers=hypers,
                           init_params=init_params) 
