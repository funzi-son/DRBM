"""The hybrid discriminative restricted Boltzmann machine"""


# Author: Srikanth Cherla
# City University London (2014)
# Contact: abfb145@city.ac.uk


from models import np
from models import theano
from models import T
#from drbm import drbm_fprop

theano.config.exception_verbosity = 'high'


def build_model(n_input, n_class, hypers, init_params):
    """Function to build the Theano graph for the HDRBM.

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
    bin_size = int(hypers['bin_size'])
    activation = str(hypers['activation'])
    alpha = float(hypers['alpha'])

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
        U_init = np.asarray((N_RNG.rand(n_class, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_class, n_hidden)),
                            dtype=theano.config.floatX)
        V_init = np.asarray((N_RNG.rand(n_input, n_hidden) * 2 - 1) /
                            np.sqrt(max(n_input, n_hidden)),
                            dtype=theano.config.floatX)
        a_init = np.zeros((n_input,), dtype=theano.config.floatX)
        c_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        d_init = np.zeros((n_class,), dtype=theano.config.floatX)
    else:
        U_init = init_params[0]
        V_init = init_params[1]
        a_init = init_params[2]
        c_init = init_params[3]
        d_init = init_params[4]

    U = theano.shared(U_init, name='U')  # Class-hidden weights
    V = theano.shared(V_init, name='V')  # Input-hidden weights
    a = theano.shared(a_init, name='a')  # Input biases
    c = theano.shared(c_init, name='c')  # Hidden biases
    d = theano.shared(d_init, name='d')  # Class biases
    params = [U, V, a, c, d]

    W = T.concatenate((V, U), axis=0)
    b = T.concatenate((a, d), axis=0)
    
    # Build Gibbs chain and graph to compute the cost function
    v_sample, gen_loss, updates_train = build_chain(v, n_input, n_class, W,
                                                    b, c, bin_size, k=n_gibbs,
                                                    activation=activation,
                                                    T_RNG=T_RNG)
  
    # Predict posterior probabilities and class-labels
    p_y_given_x = drbm_fprop(x, params, n_class, activation, bin_size)

    # Loss functions
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')
    if hypers['loss'] == 'll': # Log-likelihood
        dis_loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1))
    elif hypers['loss'] == 'ce': # Cross-entropy
        dis_loss = -T.mean(T.sum(T.log(p_y_given_x) * Y_class[y], axis=1) + \
                           T.sum(T.log(1-p_y_given_x) * (1-Y_class[y]), axis=1))
    elif hypers['loss'] == 'se': # Squared-error
        dis_loss = T.mean(T.sum((Y_class[y] - p_y_given_x) ** 2, axis=1))

    # Add weight decay (regularization) to cost.
    cost = dis_loss + alpha * gen_loss
    
    # Regularization with L1 and L2 norms
    L1 = abs(V).sum() + abs(U).sum()
    L2_sqr = (V**2).sum() + (U**2).sum()

    cost += (L1_decay*L1 + L2_decay*L2_sqr)
    grads = T.grad(cost, params, consider_constant=[v_sample])

    return (x, y, p_y_given_x, cost, params, grads)


def build_chain(v, n_input, n_class, W, bv, bh, bin_size, k=1,
                activation='sigmoid', T_RNG=None):
    """Construct a k-step Gibbs chain starting at v for an RBM.

    Input
    -----
    v : Theano vector or matrix
      If a matrix, multiple chains will be run in parallel (batch).
    n_input : integer
      Dimensionality of input feature.
    n_class : integer
      Number of output classes.
    W : Theano matrix
      Weight matrix of the RBM.
    bv : Theano vector
      Visible bias vector of the RBM.
    bh : Theano vector
      Hidden bias vector of the RBM.
    k : scalar or Theano scalar
      Length of the Gibbs chain.

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

    def gibbs_step(v):
        """One step of Gibbs sampling in the RBM."""
        # Compute hidden layer activations given visible layer
        if activation == 'sigmoid':
            mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
            h = T_RNG.binomial(size=mean_h.shape, n=1, p=mean_h,
                               dtype=theano.config.floatX)
        elif activation == 'tanh':
            proj = T.dot(v, W) + bh
            mean_h = T.exp(proj) / (T.exp(proj) + T.exp(-proj))
            h = T_RNG.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=theano.config.floatX)
            h = h*2-1
            #h = T.set_subtensor(h[h==0], -1)
        elif activation == 'binomial':
            assert bin_size > 1
            mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
            h = T_RNG.binomial(size=mean_h.shape, n=bin_size, p=mean_h,
                               dtype=theano.config.floatX)
        elif activation == 'relu':
#            mean_h = T.maximum(0, T.dot(v, W) + bh)
            mean_h = T.dot(v, W) + bh
            h = T.maximum(0, mean_h + T_RNG.normal(size=mean_h.shape, avg=0.0,
                                                   std=T.nnet.sigmoid(mean_h)))
        else:
            raise NotImplementedError
        
        # Compute visible layer activations given hidden layer
        acts_v = T.dot(h, W.T) + bv

        # Bernoulli visible units sampling
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
    c = params[3]
    d = params[4]
    Y_class = theano.shared(np.eye(n_class, dtype=theano.config.floatX),
                            name='Y_class')

    # Compute hidden state activations and energies.
    s_hid = T.dot(x, V) + c
    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + T.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])

    max_energy = T.max(energies)

    # Compute log-posteriors and then posteriors.
    if activation == 'sigmoid':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(1+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
    elif activation == 'tanh':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(T.exp(-e_i)+T.exp(e_i)), axis=1),
            sequences=[d, energies], non_sequences=[])
    elif activation == 'binomial':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + \
            T.sum(T.log((1-T.exp(bin_size*e_i))/(1-T.exp(e_i))), axis=1),
            sequences=[d, energies], non_sequences=[])
#        log_p, _ = theano.scan(
#            lambda d_i, e_i: d_i + \
#            T.sum(T.log((1-T.exp(bin_size*(e_i)))/(1-T.exp(e_i-T.max(T.abs_(energies), axis=0)))), axis=1),
#            sequences=[d, energies], non_sequences=[])
    elif activation == 'relu':
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i + T.sum(T.log(1/(1-T.exp(e_i))), axis=1),
            sequences=[d, energies], non_sequences=[])
    else:
        raise NotImplementedError

    p_y_given_x = T.nnet.softmax(log_p.T)  # XXX: Can the transpose be avoided?

    return p_y_given_x


if __name__ == '__main__':
    print "Did not implement a main() function. Use with train_cv.py."
