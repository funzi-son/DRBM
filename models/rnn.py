"""The recurrent neural network neural."""


# Author: Srikanth Cherla
# City University London (2016)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T

theano.config.exception_verbosity = 'high'


def build_model(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the standard RNN.

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
    output_type = str(hypers['output_type'])
    RNG = np.random.RandomState(hypers['seed'])

    # 1. Initialize inputs (where first dimension is time)
    x = T.matrix()

    # 2. Initialize hidden activations
    if activation == 'tanh':
        h_nonlin = T.tanh
    elif activation == 'sigmoid':
        h_nonlin = T.nnet.sigmoid
    elif activation == 'relu':
        h_nonlin = lambda x: x * (x > 0)
    elif activation == 'capped':
        h_nonlin = lambda x: T.minimum(x * (x > 0), 6)
    else:
        raise NotImplementedError

    # 3. Initialize output activations
    if output_type == 'real':
        y = T.matrix(name='y', dtype=theano.config.floatX)
    elif output_type == 'binary':
        y = T.matrix(name='y', dtype='int32')
    elif output_type == 'softmax':  # only vector labels supported
        y = T.vector(name='y', dtype='int32')
    elif output_type == 'ctc':
        raise NotImplementedError
    else:
        raise NotImplementedError


    # 4. Initialize model parameters
    if init_params is None:
        # XXX: Consider changing initialization to the same as MLP.
        W_init = np.asarray(RNG.uniform(size=(n_hidden, n_hidden),
                                        low=-.01, high=.01),
                            dtype=theano.config.floatX)
        W_in_init = np.asarray(RNG.uniform(size=(n_input, n_hidden),
                                           low=-.01, high=.01),
                               dtype=theano.config.floatX)
        W_out_init = np.asarray(RNG.uniform(size=(n_hidden, n_class),
                                            low=-.01, high=.01),
                                dtype=theano.config.floatX)
        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        by_init = np.zeros((n_class,), dtype=theano.config.floatX)
    else:
        W_init = init_params[0]
        W_in_init = init_params[1]
        W_out_init = init_params[2]
        h0_init = init_params[3]
        bh_init = init_params[4]
        by_init = init_params[5]

    W = theano.shared(value=W_init, name='W')  # Recurrent weights
    W_in = theano.shared(value=W_in_init, name='W_in')  # Input-to-hidden
    W_out = theano.shared(value=W_out_init, name='W_out')  # Hidden-to-output
    h0 = theano.shared(value=h0_init, name='h0')  # Initial hidden states
    bh = theano.shared(value=bh_init, name='bh')  # Hidden biases
    by = theano.shared(value=by_init, name='by')  # Output biases

    params = [W, W_in, W_out, h0, bh, by]  # list of all model parameters

    # Most recent updates to use with momentum
    last_updates = [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                  dtype=theano.config.floatX))
                    for param in params]

    # recurrent function (using tanh activation function) and linear output
    # activation function
    def step(x_t, h_tm1):
        """One step of the RNN's recurrence in time."""
        h_t = h_nonlin(T.dot(x_t, W_in) +
                       T.dot(h_tm1, W) + bh)
        y_t = T.dot(h_t, W_out) + by
        return h_t, y_t

    # the hidden state `h` for the entire sequence, and the output for the
    # entire sequence `y` (first dimension is always time)
    [_, y_out], _ = theano.scan(step, sequences=x,
                                 outputs_info=[h0, None])

    # 5. Initialize cost function
    if output_type == 'real':
        y_pred = y_out
        cost = T.mean((y_out - y) ** 2)
    elif output_type == 'binary':
        y_pred = T.nnet.sigmoid(y_out)
        cost = T.mean(T.nnet.binary_crossentropy(p_y_given_x, y))
    elif output_type == 'softmax':
        y_pred = T.nnet.softmax(y_out)
        cost = -T.mean(T.log(y_pred)[T.arange(y.shape[0]), y])
    else:
        raise NotImplementedError

    # L1-norm
    L1 = 0
    L1 += abs(W).sum()
    L1 += abs(W_in).sum()
    L1 += abs(W_out).sum()

    # Square of L2-norm
    L2_sqr = 0
    L2_sqr += (W ** 2).sum()
    L2_sqr += (W_in ** 2).sum()
    L2_sqr += (W_out ** 2).sum()

    cost += (L1_reg*L1 + L2_reg*L2_sqr)
    grads = T.grad(cost, params)

    return (x, y, y_pred, cost, params, grads)
