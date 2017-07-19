"""The recurrent temporal discriminative restricted Boltzmann machine"""


# Author: Srikanth Cherla
# City University London (2016)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T

theano.config.exception_verbosity = 'high'


def build_model(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the RTDRBM.

    Input
    -----
    n_input : int
      Number of visible units.
    n_class : int
      Number of classification categories.
    hypers : dict
      Dictionary of model hyperparameters
    init_params : list(np.ndarray)
      Initial values of model parameters (if any).

    Output
    ------
    x : T.matrix
      Input matrix (with number of data points as first dimension).
    y : T.ivector
      Class labels corresponding to x.
    p_y_given_x : T.nnet.softmax
      Posterior probability of y given x.
    cost: ???
      Cost function of the DRBM which is to be optimized.
    params: list(T.shared)
      A list containing the parameters of the model.
    grads: list(T.grad)
      A list containing the gradients of the parameters of the model.
    """
    # Initialise random number generators
    #rng = np.random.RandomState(hypers['seed'])
    rng = np.random.RandomState(None)
    t_rng = T.shared_randomstreams.RandomStreams(hypers['seed'])

    n_visible = n_input + n_class
    n_hidden = int(hypers['n_hidden'])  # Number of hidden units
    L1_decay = float(hypers['weight_decay'])  # L1 regularisation
    L2_decay = float(hypers['weight_decay'])  # L2 regularisation
    drop_prob = float(hypers['drop_prob'])  # Dropout probabilities

    # Initialise inputs and initial hidden layer activations.
    X = T.matrix(name='X', dtype=theano.config.floatX)
    y = T.vector(name='y', dtype='int32')
    Y = T.eye(n_class)[y]
    v = T.concatenate((X, Y), axis=1)

    # Initialise model parameters
    if init_params is None:
        W_init = np.asarray(
            rng.normal(size=(n_visible, n_hidden), scale=0.01),
            dtype=theano.config.floatX)
        Whv_init = np.asarray(
            rng.normal(size=(n_hidden, n_visible), scale=0.0001),
            dtype=theano.config.floatX)
        Whh_init = np.asarray(
            rng.normal(size=(n_hidden, n_hidden), scale=0.0001),
            dtype=theano.config.floatX)
        bv_init = np.zeros((n_visible,), dtype=theano.config.floatX)
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
    else:
        W_init = init_params[0]
        Whv_init = init_params[1]
        Whh_init = init_params[2]
        bv_init = init_params[3]
        bh_init = init_params[4]
        
    W = theano.shared(W_init, name='W')        # RBM weight matrix
    Whv = theano.shared(Whv_init, name='Whv')  # Weights between h(t-1) & v(t)
    Whh = theano.shared(Whh_init, name='Whh')  # Weights between h(t-1) & h(t)
    bv = theano.shared(bv_init, name='bv')     # Visible biases
    bh = theano.shared(bh_init, name='bh')     # Hidden biases
    h0 = T.zeros((n_hidden,))                  # Initial hidden states
    
    params = [W, Whv, Whh, bv, bh]

    # Model recurrence
    def recurrence(v_t, h_tm1):
        """One step of the model's recurrence."""
        bv_t = bv + T.dot(h_tm1, Whv)
        bh_t = bh + T.dot(h_tm1, Whh)
        h_t = T.nnet.sigmoid(bh + T.dot(v_t, W) + T.dot(h_tm1, Whh))
        return [h_t, bv_t, bh_t]

    (_, bv_t, bh_t), _ = theano.scan(
        lambda v_t, h_tm1, *_: recurrence(v_t, h_tm1),
        sequences=v, outputs_info=[h0, None, None], non_sequences=params)
    
    cost = build_drbm(v, W, bv_t[:], bh_t[:], n_class, drop_prob, t_rng)

    # Add weight decay (regularization) to cost.
    L1 = abs(W).sum()
    L2_sqr = (W**2).sum()

    cost += (L1_decay*L1 + L2_decay*L2_sqr)
    grads = T.grad(cost, params)

    # Prediction at each time-instant.
    Y_class = T.eye(n_class, dtype=theano.config.floatX)

    # Only relevant while using the model for predictin (and not training)
    def drbm_fprop(x_t, h_tm1):
        """Discriminative inference in the model."""
        U = W[-n_class:, :]  # or, U = W[n_input:, :]
        V = W[:-n_class, :]  # or, V = W[:n_input, :]
        c = bh
        d = bv[-n_class:]  # or, d = bv[:n_input]

        c_t = c + T.dot(h_tm1, Whh)
        d_t = d + T.dot(h_tm1, Whv[:, -n_class:])

        s_hid = theano.tensor.dot(x_t, V) + c_t

        # Re-weight hidden layer according to dropout
        s_hid = drop_prob * s_hid
        
        energies, _ = theano.scan(lambda y_class, U, s_hid:
                                  s_hid + theano.tensor.dot(y_class, U),
                                  sequences=[Y_class],
                                  non_sequences=[U, s_hid])
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i +
            theano.tensor.sum(theano.tensor.log(1+theano.tensor.exp(e_i))),
            sequences=[d_t, energies], non_sequences=[])

        p_t = T.nnet.softmax(log_p.T)
        v_t = T.concatenate((x_t, p_t.T[:, 0]), axis=0) # Only during prediction
        h_t = T.nnet.sigmoid(T.dot(v_t, W) + bh + T.dot(h_tm1, Whh))

        return [p_t, h_t]

    # Sequential prediction loop.
    (p_y_given_x, _), _ = theano.scan(
        lambda x_t, h_tm1, *_: drbm_fprop(x_t, h_tm1), sequences=X,
        outputs_info=[None, h0], non_sequences=params)
    p_y_given_x = p_y_given_x[:, 0, :]

    return (X, y, p_y_given_x, cost, params, grads)


def build_drbm(v, W, bv, bh, n_class, drop_prob, t_rng):
    """Construct a DRBM using the time-dependent parameters.

    Input
    -----
    v : Theano vector or matrix
      If a matrix, multiple chains will be run in parallel (batch).
    W : Theano matrix
      Weight matrix of the RBM.
    bv : Theano vector
      Visible bias vector of the RBM.
    bh : Theano vector
      Hidden bias vector of the RBM.
    n_class : int
      Number of output classes.
    t_rng : T.shared_randomstreams.RandomStreams
      Theano random number generator
    drop_prob : float
      Dropout probability

    Output
    ------
    cost : ???
      Expression whose gradient with respect to W, bv, bh is the
      negative log-likelihood of p(y|x) under the DRBM. The cost is averaged in
      the batch case.
    """

    # Initialise inputs, outputs, DRBM parameters and binary class-labels.
    x = v[:, :-n_class]
    y = T.cast(v[:, -n_class:], 'int32')

    U = W[-n_class:, :]
    V = W[:-n_class, :]
    c = bh
    d = bv[:, -n_class:]
    Y_class = T.eye(n_class, dtype=theano.config.floatX)

    # Compute hidden state activations and energies.
    s_hid = theano.tensor.dot(x, V) + c

    # Apply dropout --> son commented
    #s_hid = T.switch(t_rng.binomial(size=s_hid.shape, p=drop_prob), s_hid, 0)

    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + theano.tensor.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])
    log_p, _ = theano.scan(
        lambda d_i, e_i: d_i +
        theano.tensor.sum(theano.tensor.log(1+theano.tensor.exp(e_i)),
                          axis=1),
        sequences=[d.T, energies], non_sequences=[])
    p_y_given_x = T.nnet.softmax(log_p.T)

    # Cross entropy loss
    cost = -T.mean(T.sum(T.log(p_y_given_x) * y, axis=1))

    return cost
