"""The restricted Boltzmann machine"""


# Author: Srikanth Cherla
# City University London (2014)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T

theano.config.exception_verbosity = 'high'


def build_model(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the RBM.

    Input
    -----
    n_input : integer
      Dimensionality of input features to the model.
    n_class : integer
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
    n_visible = n_input + n_class
    n_hidden = int(hypers['n_hidden'])
    L1_decay = float(hypers['weight_decay'])
    L2_decay = float(hypers['weight_decay'])
    n_gibbs = int(hypers['n_gibbs'])
    activation = str(hypers['activation'])

    # Random number generators
    T_RNG = T.shared_randomstreams.RandomStreams(hypers['seed'])
    N_RNG = np.random.RandomState(hypers['seed'])

    # 1. Initialize visible layer, inputs and targets
    x = T.matrix(name='x', dtype=theano.config.floatX)
    y = T.ivector(name='y') # XXX: What should be the type of this?
    Y = T.eye(n_class)[y]
    v = T.concatenate((x, Y), axis=1)

    # Initialize model parameters
    if init_params is None:
        W_init = np.asarray(
            N_RNG.normal(size=(n_visible, n_hidden), scale=0.01),
            dtype=theano.config.floatX)
        bv_init = np.zeros((n_visible,), dtype=theano.config.floatX)
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
    else:
        W_init = init_params[0]
        bv_init = init_params[1]
        bh_init = init_params[2]

    W = theano.shared(W_init, name='W')     # RBM weight matrix
    bv = theano.shared(bv_init, name='bv')  # Visible biases
    bh = theano.shared(bh_init, name='bh')  # Hidden biases

    params = [W, bv, bh]
    
    # Build Gibbs chain and graph to compute the cost function
    v_sample, cost, updates_train = build_chain(v, n_input, n_class, W,
                                                bv, bh, k=n_gibbs,
                                                activation=activation,
                                                T_RNG=T_RNG)
    
    # Add weight decay (regularization) to cost.
    L1 = abs(W).sum()
    L2_sqr = (W**2).sum()
    cost += (L1_decay*L1 + L2_decay*L2_sqr)
    grads = T.grad(cost, params, consider_constant=v_sample)

    # Expressions to compute conditional distribution.
    p_y_given_x = drbm_fprop(x, params, n_class, activation)

    return (x, y, p_y_given_x, cost, params, grads)


def build_chain(v, n_input, n_class, W, bv, bh, k=1, activation='sigmoid', 
                T_RNG=None):
    """Construct a k-step Gibbs chain starting at v for an RBM.

    Input
    -----
    v : T.matrix or T.vector
      If a matrix, multiple chains will be run in parallel (batch).
    n_input : int
      Dimensionality of input feature.
    n_class : int
      Number of output classes.
    W : T.matrix
      Weight matrix of the RBM.
    bv : T.vector
      Visible bias vector of the RBM.
    bh : T.vector
      Hidden bias vector of the RBM.
    k : int
      Length of the Gibbs chain (number of sampling steps).
    activation : str
      Type of activation function.
    T_RNG : T.streams.RandomStreams
      Theano random number generator.

    Output
    ------
    v_sample : Theano vector or matrix with the same shape as `v`
      Corresponds to the generated sample(s).
    cost : Theano scalar
      Expression whose gradient with respect to W, bv, bh is the CD-k
      approximation to the log-likelihood of `v` (training example) under the
      RBM. The cost is averaged in the batch case.
    updates: dictionary of Theano variable -> Theano variable
      The `updates` object returned by scan."""
    if T_RNG is None:
        T_RNG = T.shared_randomstreams.RandomStreams(860331)

    # One iteration of the Gibbs sampler.
    def gibbs_step(v):
        """One step of Gibbs sampling in the RBM."""
        # Compute hidden layer activations given visible layer
        if activation == 'sigmoid':
            mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
            h = T_RNG.binomial(size=mean_h.shape, n=1, p=mean_h,
                               dtype=theano.config.floatX)
        elif activation == 'tanh':
            raise NotImplementedError
        elif activation == 'relu': # XXX: Not working
            mean_h = T.maximum(0, T.dot(v, W) + bh)
            h = T.maximum(0, mean_h + T_RNG.normal(size=mean_h.shape, avg=0.0,
                                                   std=T.nnet.sigmoid(mean_h)))
        else:
            raise NotImplementedError
        
        # Compute visible layer activations given hidden layer
        acts_v = T.dot(h, W.T) + bv

#        # Multinomial visible units sampling (equally sized)
#        # TODO: Make this an if-else section based on an input hyperparameter
#        acts_in = acts_v[:, :n_input]
#        probs_in = T.nnet.softmax(acts_in)
#        v_in = T_RNG.multinomial(n=1, pvals=probs_in,
#                                 dtype=theano.config.floatX)
#        acts_out = acts_v[:, -n_class:]
#        probs_out = T.nnet.softmax(acts_out)
#        v_out = T_RNG.multinomial(n=1, pvals=probs_out,
#                                  dtype=theano.config.floatX)
#        mean_v = T.concatenate((probs_in, probs_out), axis=1)
#        v = T.concatenate((v_in, v_out), axis=1)

        # Binomial visible units sampling
        mean_v = T.nnet.sigmoid(acts_v)
        v = T_RNG.binomial(size=mean_v.shape, n=1, p=mean_v,
                           dtype=theano.config.floatX)

        return mean_v, v

    # k-step Gibbs sampling loop
    chain, updates = theano.scan(lambda v: gibbs_step(v)[1],
                                 outputs_info=[v], non_sequences=[],
                                 n_steps=k)
    v_sample = chain[-1]

    def free_energy(v):
        """Free energy of RBM visible layer."""
        return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
    
    cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]

    return v_sample, cost, updates


def drbm_fprop(x, params, n_class, activation):
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
    U = params[0][-n_class:, :]  # or, U = W[n_input:, :]
    W = params[0][:-n_class, :]  # or, V = W[:n_input, :]
    d = params[1][-n_class:]  # or, d = bv[:n_input]
    c = params[2]
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')

    # Compute hidden state activations and energies.
    s_hid = T.dot(x, W) + c
    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + T.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])

    # Compute log-posteriors and then posteriors.
    if activation == 'sigmoid':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(1+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
    elif activation == 'tanh':
        raise NotImplementedError
    elif activation == 'relu':
        raise NotImplementedError
    else:
        raise NotImplementedError

    p_y_given_x = T.nnet.softmax(log_p.T)  # XXX: Can the transpose be avoided?

    return p_y_given_x
