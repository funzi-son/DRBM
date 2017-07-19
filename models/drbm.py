"""The discriminative restricted Boltzmann machine."""


# Author: Srikanth Cherla
# City University London (2014)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T

theano.config.exception_verbosity = 'high'


def build_model(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the DRBM.

    Input
    -----
    n_input : int
      Dimensionality of input features to the model.
    n_class : int
      Number of class-labels. 
    hypers : dict
      Model hyperparameters.
    init_params : list
      A list of initial values for the model parameters.

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
    n_hidden = int(hypers['n_hidden'])
    L1_reg = float(hypers['weight_decay'])
    L2_reg = float(hypers['weight_decay'])
    activation = str(hypers['activation'])
    bin_size = int(hypers['bin_size'])
    RNG = np.random.RandomState(hypers['seed'])

    # 1. Initialize inputs and targets
    x = T.matrix(name='x', dtype=theano.config.floatX)

    # 2. Initialize outputs
    y = T.ivector(name='y')

    # 3. Initialize model parameters
    if init_params is None:
        U_init = np.asarray((RNG.rand(n_class, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_class, n_hidden)),
                            dtype=theano.config.floatX)
        V_init = np.asarray((RNG.rand(n_input, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_input, n_hidden)),
                            dtype=theano.config.floatX)
        c_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        d_init = np.zeros((n_class,), dtype=theano.config.floatX)
    else:
        U_init = init_params[0]
        V_init = init_params[1]
        c_init = init_params[2]
        d_init = init_params[3]

    U = theano.shared(U_init, name='U')  # Class-hidden weights
    V = theano.shared(V_init, name='V')  # Input-hidden weights
    c = theano.shared(c_init, name='c')  # Hidden biases
    d = theano.shared(d_init, name='d')  # Class biases
    params = [U, V, c, d]

    # Predict posterior probabilities and class-labels
    p_y_given_x = drbm_fprop(x, params, n_class, activation, bin_size)

    # Loss functions
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')
    if hypers['loss'] == 'll': # Log-likelihood
        loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1))
    elif hypers['loss'] == 'ce': # Cross-entropy
        loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1) + \
                       T.sum(T.log(1-p_y_given_x) * (1-Y_class[y]), axis=1))
    elif hypers['loss'] == 'se': # Squared-error
        loss = T.mean(T.sum((Y_class[y] - p_y_given_x) ** 2, axis=1))

    # Regularization with L1 and L2 norms
    L1 = abs(V).sum() + abs(U).sum()
    L2 = (V**2).sum() + (U**2).sum()

    cost = loss + L1_reg*L1 + L2_reg*L2
    grads = T.grad(cost, params)

    return x, y, p_y_given_x, cost, params, grads


def drbm_fprop(x, params, n_class, activation, bin_size):
    """Posterior probability of classes given inputs and model parameters.

    Input
    -----
    x: T.matrix (of type theano.config.floatX)
      Input data matrix.
    params: list
      A list containing the four parameters of the DRBM (see class definition).
    n_class: integer
      Number of classes.

    Output
    ------
    p_y_given_x: T.nnet.softmax
      Posterior class probabilities of the targets given the inputs.
    """
    # Initialize DRBM parameters and binary class-labels.
    U = params[0]
    V = params[1]
    c = params[2]
    d = params[3]
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')

    # Compute hidden state activations and energies.
    s_hid = T.dot(x, V) + c # propagated signals from x
    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + T.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])

    # Compute log-posteriors and then posteriors.
    if activation == 'sigmoid':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(1+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
        p_y_given_x = T.nnet.softmax(log_p.T)
    elif activation == 'tanh':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(T.exp(-e_i)+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
        p_y_given_x = T.nnet.softmax(log_p.T)
    elif activation == 'binomial':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + \
            T.sum(T.log((1-T.exp(bin_size*e_i))/(1-T.exp(e_i))), axis=1),
            sequences=[d, energies], non_sequences=[])
        p_y_given_x = T.nnet.softmax(log_p.T)
    elif activation == 'relu':
        # XXX: The 0.01 here is a bit hacky - maybe make it a hyperparameter?
        p, _ = theano.scan(
                lambda d_i, e_i: T.exp(d_i) * T.prod(0.01/(1-T.exp(e_i)),
                                                     axis=1), 
                sequences=[d, energies], non_sequences=[])
        p = p.T
        p_y_given_x = p / T.sum(p, axis=0)
    else:
        raise NotImplementedError

    return p_y_given_x
