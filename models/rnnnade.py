"""The recurrent neural network neural autoregressive distribution estimator"""


# Author: Srikanth Cherla
# City University London (2015)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T

theano.config.exception_verbosity = 'high'


def build_model(n_input, hypers, init_params):
    """Function to build the Theano graph for the DRTRBM.

    Input
    -----
    n_input : int
      Number of visible units.
    hypers : dict
      Dictionary of model hyperparameters.
    init_params : list(np.ndarray)
      Initial values of model parameters (if any).

    Output
    ------
    x : T.matrix
      Input matrix (with number of data points as first dimension).
    y : None
      No class label vector is returned due to unsupervised learning case.
    p_y_given_x : T.nnet.softmax
      Posterior probability of y given x.
    cost: ???
      Cost function of the DRBM which is to be optimized.
    params: list(T.shared)
      A list containing the parameters of the model.
    grads: list(T.grad)
      A list containing the gradients of the parameters of the model.
    """
    rng = np.random.RandomState(hypers['seed'])

    n_hidden = int(hypers['n_hidden'])
    n_rhidden = int(hypers['n_rhidden'])
    L1_decay = float(hypers['weight_decay'])
    L2_decay = float(hypers['weight_decay'])

    # Initialize inputs and initial hidden layer activations.
#    X = T.matrix(name='X', dtype=theano.config.floatX)
#    y = T.vector(name='y', dtype='int32')
#    Y = T.eye(n_class)[y]
#    v = T.concatenate((X, Y), axis=1)
    v = T.matrix(name='v', dtype=theano.config.floatX)

    # Initialize model parameters
    if init_params is None:
        W_init = np.asarray(
            rng.normal(size=(n_input, n_hidden), scale=0.01),
            dtype=theano.config.floatX)
        V_init = np.asarray(
            rng.normal(size=(n_hidden, n_input), scale=0.01),
            dtype=theano.config.floatX)
        Wrh_init = np.asarray(
            rng.normal(size=(n_rhidden, n_hidden), scale=0.0001),
            dtype=theano.config.floatX)
        Wrv_init = np.asarray(
            rng.normal(size=(n_rhidden, n_input), scale=0.0001),
            dtype=theano.config.floatX)
        Wrr_init = np.asarray(
            rng.normal(size=(n_rhidden, n_rhidden), scale=0.0001),
            dtype=theano.config.floatX)
        Wvr_init = np.asarray(
            rng.normal(size=(n_input, n_rhidden), scale=0.0001),
            dtype=theano.config.floatX)
        bv_init = np.zeros((n_input,), dtype=theano.config.floatX)
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        br_init = np.zeros((n_rhidden,), dtype=theano.config.floatX)
    else:
        W_init = init_params[0]
        V_init = init_params[1]
        Wrh_init = init_params[2]
        Wrv_init = init_params[3]
        Wvr_init = init_params[4]
        Wrr_init = init_params[5]
        bv_init = init_params[6]
        bh_init = init_params[7]
        br_init = init_params[8]
        
    W = theano.shared(W_init, name='W')        # NADE weight matrix 1
    V = theano.shared(V_init, name='V')        # NADE weight matrix 2
    Wrh = theano.shared(Wrh_init, name='Wrh')  # Weights between h(t-1) & v(t)
    Wrv = theano.shared(Wrv_init, name='Wrv')  # Weights between h(t-1) & v(t)
    Wrr = theano.shared(Wrr_init, name='Wrr')  # Weights between h(t-1) & h(t)
    Wvr = theano.shared(Wvr_init, name='Wvr')  # Weights between h(t-1) & v(t)
    bv = theano.shared(bv_init, name='bv')     # Visible biases
    bh = theano.shared(bh_init, name='bh')     # Hidden biases
    br = theano.shared(br_init, name='br')     # Hidden biases
    r0 = T.zeros((n_rhidden,))                 # Initial hidden states
    
    params = [W, V, Wrh, Wrv, Wrr, Wvr, bv, bh, br]

    # Model recurrence
    def recurrence(v_t, r_tm1):
        """One step of the model's recurrence."""
        bv_t = bv + T.dot(r_tm1, Wrv)
        bh_t = bh + T.dot(r_tm1, Wrh)
        r_t = T.nnet.sigmoid(br + T.dot(v_t, Wvr) + T.dot(r_tm1, Wrr))
        return [r_t, bv_t, bh_t]

    (_, bv_t, bh_t), _ = theano.scan(
        lambda v_t, r_tm1, *_: recurrence(v_t, r_tm1),
        sequences=v, outputs_info=[r0, None, None], non_sequences=params)
    
    y_pred, cost = build_nade(v, W, V, bv_t[:], bh_t[:])

    # Add weight decay (regularization) to cost.
    L1 = abs(W).sum()
    L2_sqr = (W**2).sum()
    cost += (L1_decay*L1 + L2_decay*L2_sqr)
    grads = T.grad(cost, params)

    return (v, None, y_pred, cost, params, grads) 


def build_nade(v, W, V, bv, bh):
    """Construct a NADE using the time-dependent parameters.

    Input
    -----
    v : Theano vector or matrix
      If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
      Weight (visible-to-hidden) matrix of the NADE.
    V : Theano matrix
      Weight (hidden-to-visible) matrix of the NADE.
    bv : Theano vector
      Visible bias vector of the RBM.
    bh : Theano vector
      Hidden bias vector of the RBM.
    n_class : Integer
      Number of output classes.

    Output
    ------
    cost : ???
      Expression whose gradient with respect to W, bv, bh is the
      negative log-likelihood of p(y|x) under the NADE. The cost is averaged in
      the batch case.
    """
    # Initialize inputs, outputs, NADE parameters and binary class-labels.
    a = T.shape_padright(v) * T.shape_padleft(W)
    a = a.dimshuffle(1, 0, 2)

    bh_init = bh
    if bh.ndim == 1:
        bh_init = T.dot(T.ones((v.shape[0], 1)), T.shape_padleft(bh))

    (activations, s), _ = theano.scan(
        lambda V_i, a_i, partial_im1: (a_i+partial_im1, 
                                       T.dot(V_i, 
                                             T.nnet.sigmoid(partial_im1.T))), 
        sequences=[V.T, a], outputs_info=[bh_init, None]) 
    s = s.T + bv
    y = T.nnet.sigmoid(s)

    cost = -v*T.log(y) - (1-v)*T.log(1-y)
    cost = cost.sum() / v.shape[0]

    return y, cost
