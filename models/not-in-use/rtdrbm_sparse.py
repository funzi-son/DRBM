"""Implementation of an online version of the Recurrent Temporal Discriminative
Restricted Boltzmann Machine which is trained using backpropagation through
time and stochastic gradient descent."""


# Author: Srikanth Cherla
# City University London (2015)
# Contact: abfb145@city.ac.uk


from evaluate import accuracy
from evaluate import negative_log_likelihood
import numpy as np
from optimize import sgd
from optimize import stoch_gd
import scipy.sparse as scsp
import theano
import theano.sparse as S
import theano.tensor as T
import time
from utils import index_to_onehot
from utils import convolve_sequences
from utils import truncate_sequences

theano.config.exception_verbosity = 'high'


#############################
# Section: Model definition #
#############################
class RTDRBM(object):
    """Recurrent Temporal Discriminative RBM class"""

    def __init__(self, n_input, n_class, hypers, init_params=None):
        """Constructs and compiles Theano functions for training and
        prediction.

        Input
        -----
        n_input : integer
          Number of inputs which make up a part of the visible layer.
        n_class : integer
          Number of output classes which make up the rest of the visible layer.
        hypers : dictionary
          Model hyperparameters.
        init_params : list
          Model parameters.
        """
        self.model_type = str(hypers['model_type'])
        self.n_input = int(n_input)
        self.n_class = int(n_class)
        self.n_hidden = int(hypers['n_hidden'])
        self.tr_online = int(hypers['tr_online'])
        self.tr_offline = int(hypers['tr_offline'])
        self.L1 = float(hypers['weight_decay'])
        self.L2_sqr = float(hypers['weight_decay'])
        self.seed = float(hypers['seed'])

        # Build the model graph.
        (x, y, v, cost, params, p_y_given_x, y_out) = build_rtdrbm(
            n_input+n_class, n_class, hypers, init_params)

        # Compute parameter gradients w.r.t cost.
        lr = T.scalar('learning_rate', dtype=theano.config.floatX)
        mom = T.scalar('momentum', dtype=theano.config.floatX)
        gradients = T.grad(cost, params)
        updates_train = [(param, param - lr * gradient)
                         for param, gradient in zip(params, gradients)]

        # Functions for training, evaluating and saving the model.
        self.train_function = theano.function([x, y, lr], cost,
                                              updates=updates_train,
                                              allow_input_downcast=True)
        self.predict_proba = theano.function([x], p_y_given_x,
                                             allow_input_downcast=True)
        self.predict_function = theano.function([x], y_out,
                                                allow_input_downcast=True)
        self.get_model_parameters = theano.function([], params)


def build_drbm(v, W, bv, bh, n_class):
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
    n_class : Integer
      Number of output classes.

    Output
    ------
    cost : ???
      Expression whose gradient with respect to W, bv, bh is the
      negative log-likelihood of p(y|x) under the DRBM. The cost is averaged in
      the batch case.
    """
    # Initialize inputs, outputs, DRBM parameters and binary class-labels.
    x = v[:, :-n_class]
    y = T.cast(v[:, -n_class:], 'int32')

    U = W[-n_class:, :]
    V = W[:-n_class, :]
    c = bh
    d = bv[:, -n_class:]
    Y_class = T.eye(n_class, dtype=theano.config.floatX)

    # Compute hidden state activations and energies.
    s_hid = S.dot(x, V) + c
    # XXX: s_hid = T.dot(x, V) + c
    energies, _ = theano.scan(lambda y_class, U, s_hid:
                              s_hid + T.dot(y_class, U),
                              sequences=[Y_class],
                              non_sequences=[U, s_hid])
    log_p, _ = theano.scan(
        lambda d_i, e_i: d_i +
        T.sum(T.log(1+T.exp(e_i)),
                          axis=1),
        sequences=[d.T, energies], non_sequences=[])
    p_y_given_x = T.nnet.softmax(log_p.T)

    # Cross entropy loss
    cost = -T.mean(T.sum(T.log(p_y_given_x) * y, axis=1))

    return cost


def build_rtdrbm(n_visible, n_class, hypers, init_params=None):
    """Function to build the Theano graph for the DRTRBM.

    Input
    -----
    n_visible : integer
      Number of visible units.
    n_hidden : integer
      Number of hidden units of the conditional RBMs.
    n_class : integer
      Number of classification categories.
    L1_decay : float
      L1 weight decay.
    L2_decay : float
      L2 weight decay.
    init_params : list(np.ndarray)
      Initial values of model parameters (if any).

    Output
    ------
    v : Theano matrix
      Symbolic variable holding an input sequence (used during training)
    cost : Theano scalar
      Expression whose gradient (considering v_sample constant) corresponds to
      the LL gradient of the RTDRBM (used during training)
    params : tuple of Theano shared variables
      The parameters of the model to be optimized during training.
    p_y_given_x : T.softmax
      Symbolic variable holding a sequence of posterior probabilities of
      predictions made by the model.
    y_out: T.argmax
      Class-labels predicted by the model.
    """
    rng = np.random.RandomState(hypers['seed'])

    n_hidden = int(hypers['n_hidden'])
    L1_decay = float(hypers['weight_decay'])
    L2_decay = float(hypers['weight_decay'])

    # Initialize inputs and initial hidden layer activations.
    X = S.csr_matrix(name='X', dtype=theano.config.floatX)
    # XXX: X = T.matrix(name='X', dtype=theano.config.floatX)
    y = T.vector(name='y', dtype='int32')
    Y = S.csr_from_dense(T.eye(n_class)[y])
    # XXX: Y = T.eye(n_class)[y] 
    v = S.hstack([X, Y], format='csr', dtype=theano.config.floatX)
    # XXX: v = T.concatenate((X, Y), axis=1)

    # Initialize model parameters
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
    def recurrence(i_t, h_tm1):
        """One step of the model's recurrence."""
        bv_t = bv + T.dot(h_tm1, Whv)
        bh_t = bh + T.dot(h_tm1, Whh)
        h_t = T.nnet.sigmoid(bh + S.dot(v[i_t:i_t+1], W) + T.dot(h_tm1, Whh))
        # XXX: h_t = T.nnet.sigmoid(bh + T.dot(v_t, W) + T.dot(h_tm1, Whh))
        return [h_t, bv_t, bh_t]

    # NOTE: Conditional RBMs can be trained in batches using the below idea.
    # For training, the deterministic recurrence is used to compute all the
    # {bv_t, bh_t, 1 <= t <= T} given v. So here, the (time-dependent) dynamic
    # biases are pre-computed using scan, and are batch-processed in
    # build_drbm. This is something I may be able to exploit even during
    # prediction, since I'm forward-propagating the one-hot class vectors
    # anyway.
#    (_, bv_t, bh_t), _ = theano.scan(
#        lambda v_t, h_tm1, *_: recurrence(v_t, h_tm1),
#        sequences=v, outputs_info=[h0, None, None], non_sequences=params)
    (_, bv_t, bh_t), _ = theano.scan(
        lambda i_t, h_tm1, *_: recurrence(i_t, h_tm1),
        sequences=T.arange(v.shape[0], dtype='int64'), 
        outputs_info=[h0, None, None], 
        non_sequences=params+[v])
    cost = build_drbm(v, W, bv_t[:], bh_t[:], n_class)

    # Add weight decay (regularization) to cost.
    L1 = abs(W).sum()
    L2_sqr = (W**2).sum()
    cost += (L1_decay*L1 + L2_decay*L2_sqr)

    # Prediction at each time-instant.
    Y_class = T.eye(n_class, dtype=theano.config.floatX)

    def drbm_fprop(x_t, h_tm1):
        """Discriminative inference in the model."""
        U = W[-n_class:, :]  # or, U = W[n_input:, :]
        V = W[:-n_class, :]  # or, V = W[:n_input, :]
        c = bh
        d = bv[-n_class:]  # or, d = bv[:n_input]

        c_t = c + T.dot(h_tm1, Whh)
        d_t = d + T.dot(h_tm1, Whv[:, -n_class:])
        
        s_hid = S.dot(x_t, V) + c_t
        # XXX: s_hid = T.dot(x_t, V) + c_t
        energies, _ = theano.scan(lambda y_class, U, s_hid:
                                  s_hid + T.dot(y_class, U),
                                  sequences=[Y_class],
                                  non_sequences=[U, s_hid])
        log_p, _ = theano.scan(
            lambda d_i, e_i: d_i +
            T.sum(T.log(1+T.exp(e_i))),
            sequences=[d_t, energies], non_sequences=[])

        p_t = T.nnet.softmax(log_p.T)
        v_t = S.vstack((x_t, S.csr_from_dense(p_t.T[:, 0])), format='csr')
        # XXX: v_t = T.concatenate((x_t, p_t.T[:, 0]), axis=0) # Only during prediction
        h_t = T.nnet.sigmoid(S.dot(v_t, W) + bh + T.dot(h_tm1, Whh))
        # XXX: h_t = T.nnet.sigmoid(T.dot(v_t, W) + bh + T.dot(h_tm1, Whh))

        return [p_t, h_t]

    # Sequential prediction loop.
    (p_y_given_x, _), _ = theano.scan(
        lambda x_t, h_tm1, *_: drbm_fprop(x_t, h_tm1), sequences=X,
        outputs_info=[None, h0], non_sequences=params)
    p_y_given_x = p_y_given_x[:, 0, :]
    y_out = T.argmax(p_y_given_x, axis=-1)

    return (X, y, v, cost, params, p_y_given_x, y_out)


#####################
# Section: Training #
#####################
def train_model(dataset, mod_hypers, opt_hypers, dat_hypers,
                model_params=None): 
    """Stochastic gradient descent optimization of a RTDRBM.

    Input
    -----
    dataset : tuple(tuple(np.ndarray))
      Training and validation sets
    mod_hypers : dictionary
      Model hyperparameters
    opt_hypers : dictionary
      Optimization hyperparameters
    model_params : list(np.ndarray)
      A list of model parameter values (optional)

    Output
    ------
    model_params : list(np.ndarray)
      List of all the model parameters (see class definition)
    valid_nll : float
      Validation set negative log-likelihood
    """
#    rng = np.random.RandomState(mod_hypers['seed'])
#
#    # Doing this mainly for clarity in the code that immediately follows
#    n_input = dat_hypers['n_input'] # Note that...
#    n_class = dat_hypers['n_class'] # ... n_visible = n_input + n_class
#    n_hidden = mod_hypers['n_hidden'] # Number of hidden units
#
#    # Pre-train an RBM (this does help quite a lot for most of the datasets).
#    if model_params is None:
#        # Initialise pre-training hyperparameters
#        pretrain_params = pretrain_model(dataset, mod_hypers, dat_hypers) 
#    
#        model_params = [pretrain_params[0],
#                        rng.normal(scale=0.0001, size=(n_hidden, 
#                                                       n_input+n_class))
#                        .astype(theano.config.floatX),
#                        rng.normal(scale=0.0001, size=(n_hidden, n_hidden))
#                        .astype(theano.config.floatX),
#                        pretrain_params[1], pretrain_params[2]]
 
    # The actual learning step
    if opt_hypers['opt_type'] == 'batch-gd':
        learned_params, valid_score = batch_learning(
            dataset, model_params, mod_hypers, opt_hypers, dat_hypers)
    elif opt_hypers['opt_type'] == 'stoch-gd':
        learned_params, valid_score = stochastic_learning(
            dataset, model_params, mod_hypers, opt_hypers, dat_hypers)
    elif opt_hypers['opt_type'] == 'pretrain':
        learned_params = model_params
        valid_score = np.inf # For now.

    return learned_params, valid_score


def pretrain_model(dataset, mod_hypers, dat_hypers):
    """Pretrain the parameters of an DIRTRBM.

    Input
    -----
    X_train : np.ndarray
      Training inputs
    y_train : np.ndarray
      Training labels
    mod_hypers : dict
      Model hyperparameters
    dat_hypers : dict
      Information about training data

    Output
    ------
    model_params : list(np.ndarray)
      A list of pre-trained model parameters
    """
    from rbm import train_model as pretrain_rbm

    # RBM hyperparameters
    rbm_mod_hypers = {}
    rbm_mod_hypers['model_type'] = 'rbm'
    rbm_mod_hypers['n_hidden'] = mod_hypers['n_hidden']
    rbm_mod_hypers['weight_decay'] = 0.0001
    rbm_mod_hypers['activation'] = 'sigmoid' # TODO: Change this to the model's.
    rbm_mod_hypers['n_gibbs'] = 1 
    rbm_mod_hypers['seed'] = 860331
    
    # RBM optimisation hyperparameters
    rbm_opt_hypers = {}
    rbm_opt_hypers['opt_type'] = 'batch-gd'
    rbm_opt_hypers['learning_rate'] = 0.1
    rbm_opt_hypers['schedule'] = 'constant'
    rbm_opt_hypers['rate_param'] = 100
    rbm_opt_hypers['patience'] = 100
    rbm_opt_hypers['max_epoch'] = mod_hypers['n_pretrain']
    rbm_opt_hypers['validation_frequency'] = 20 # Greater than max_epoch
    rbm_opt_hypers['batch_size'] = 100

    issparse = scsp.issparse(dataset[0][0][0])
    if issparse:
        dataset_pretrain = \
            tuple([tuple([scsp.vstack(tuple(dataset[s_idx][t_idx]), 
                                      format='csr')
                          for t_idx in xrange(len(dataset[s_idx]))])
                   for s_idx in xrange(len(dataset))])
    else:
        dataset_pretrain = \
            tuple([tuple([np.concatenate(tuple(dataset[s_idx][t_idx]),
                                         axis=0)
                          for t_idx in xrange(len(dataset[s_idx]))]) 
                   for s_idx in xrange(len(dataset))])

    model_params, _ = pretrain_rbm(dataset_pretrain, rbm_mod_hypers,
                                   rbm_opt_hypers, dat_hypers, None)

    return model_params


def stochastic_learning(dataset, init_params, mod_hypers, opt_hypers, 
                        dat_hypers):
    """Learn the RTDRBM with Stochastic Gradient Descent.

    Input
    -----
    dataset : tuple(tuple(list))
      Each inner tuple contains two lists containing the training inputs and
      labels respectively.
    init_params : list
      A list containing initial values of the different RTDRBM parameters.
    mod_hypers : dict
      Model hyperparameters.
    opt_hypers : dict
      Optimization hyperparameters.
    dat_hypers : dict
      Dataset hyperparameters.

    Output
    ------
    params : list
      A list containing the learned parameters.
    valid_score : np.ndarray (of size 0)
      Best validation score.
    """
    # Read training, and validation sets
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]

    # We need these as well, and note that n_visible = n_input + n_class
    n_input = dat_hypers['n_input']
    n_class = dat_hypers['n_class']

    # I'm training the model using online BPTT, so I convolve sequences to
    # a specified maximum length and update the dataset variable here. Note
    # that if the truncation length is larger than the length of the longest
    # sequence in the dataset, entire sequences are used.
    X_train, y_train = convolve_sequences(X_train, y_train,
                                          mod_hypers['tr_offline'])
    dataset = ((X_train, y_train), (X_valid, y_valid))

    # Learn an RTDRBM initialized with pre-trained RBM parameters with
    # stochastic gradient descent and (truncated) BPTT.
    model = RTDRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                   init_params=init_params) 
    optimizer = stoch_gd(opt_hypers)
    params, valid_score = optimizer.optimize(model, dataset)

    return params, valid_score


def batch_learning(dataset, init_params, mod_hypers, opt_hypers, 
                   dat_hypers):
    """Learn the RTDRBM with Batch Gradient Descent.

    Input
    -----
    dataset : tuple(tuple(list))
      Each inner tuple contains two lists containing the training inputs and
      labels respectively.
    init_params : list
      A list containing initial values of the different RTDRBM parameters.
    mod_hypers : dict
      Model hyperparameters.
    opt_hypers : dict
      Optimization hyperparameters.
    dat_hypers : dict
      Dataset hyperparameters.

    Output
    ------
    params : list
      A list containing the learned parameters.
    valid_score : np.ndarray (of size 0)
      Best validation score.
    """
    # Read training, and validation sets
    X_train, y_train = dataset[0]
    X_valid, y_valid = dataset[1]

    # I'm training the model using truncated BPTT, so I truncate sequences to
    # a specified maximum length and update the dataset variable here. Note
    # that if the truncation length is larger than the length of the longest
    # sequence in the dataset, entire sequences are used.
    X_train, y_train, _ = truncate_sequences(X_train, y_train,
                                             mod_hypers['tr_offline']) 
    dataset = ((X_train, y_train), (X_valid, y_valid))

    # We need these as well, and note that n_visible = n_input + n_class
    n_input = dat_hypers['n_input']
    n_class = dat_hypers['n_class']

    # Batch-learn an RTDRBM initialized with pre-trained RBM parameters with
    # epoch-wise BPTT.
    model = RTDRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                   init_params=init_params) 
    optimizer = sgd(opt_hypers)
    params, valid_score = optimizer.optimize(model, dataset)

    return params, valid_score


#######################
# Section: Evaluation #
#######################
def test_model(model, data, eva_hypers):
    """Evaluate an already trained RTDRBM on given test data.

    Input
    -----
    model : tuple(list(np.ndarray), dict)
      A tuple containing the model parameters, and a dictionary with its
      hyperparameters.
    dataset : tuple(tuple(np.ndarray), dict)
      A tuple containing test inputs and their corresponding labels, and
      dictionary with information about the data. 
    eva_hypers: dict
      A dictionary which specifies how exactly to evaluate the model.
    """
    if eva_hypers['method'] == 'offline':
        return test_offline(model, data, eva_hypers)
    elif eva_hypers['method'] in ('online', 'semi-online'):
        return test_online(model, data, eva_hypers)
    else:
        assert False, 'unknown evaluation method'


def test_offline(model, data, eva_hypers):
    """Test a model offline (by not updating its parameters after prediction).

    Input
    -----
    model : tuple(list(np.ndarray), dict)
      A tuple containing the model parameters, and a dictionary with its
      hyperparameters.
    data : tuple(tuple(np.ndarray), dict)
      A tuple containing test inputs and their corresponding labels, and
      dictionary with information about the data. 
    eva_hypers: dict
      A dictionary which specifies how exactly to evaluate the model.

    Output
    ------
    noname : tuple
      A tuple containing test prediction probabilities, the corresponding cross
      entropy, prediction labels, and the corresponding accuracy.
    """
    test_data, dat_hypers = data
    n_input = dat_hypers['n_input'] 
    n_class = dat_hypers['n_class'] 
    n_test = dat_hypers['n_test']
    X_test = test_data[0]
    y_test = test_data[1]
    issparse = scsp.issparse(X_test[0])
    model_params, mod_hypers = model

    model = RTDRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                    init_params=model_params) 

    # Cross entropy
    y_prob = []
    if issparse:
        for i in xrange(n_test):
#            y_prob.append(model.predict_proba(X_test[i], y_test[i]))
            y_prob.append(model.predict_proba(X_test[i].todense()))
        test_nll = negative_log_likelihood(
            np.concatenate(tuple(y_prob), axis=0),
            np.squeeze(np.asarray(scsp.vstack(tuple(y_test), format='csr'))))
    else:
        for i in xrange(n_test):
#            y_prob.append(model.predict_proba(X_test[i], y_test[i]))
            y_prob.append(model.predict_proba(X_test[i]))
        test_nll = negative_log_likelihood(np.concatenate(tuple(y_prob), axis=0),
                                np.concatenate(tuple(y_test), axis=0))

    # Accuracy
    y_pred = []
    if issparse:
        for i in xrange(n_test):
#            y_pred.append(model.predict_function(X_test[i], y_test[i]))
            y_pred.append(model.predict_function(X_test[i].todense()))
        test_acc = accuracy(
            np.concatenate(tuple(y_pred), axis=0),
            np.squeeze(np.asarray(scsp.vstack(tuple(y_test), format='csr'))))
    else:
        for i in xrange(n_test):
#            y_pred.append(model.predict_function(X_test[i], y_test[i]))
            y_pred.append(model.predict_function(X_test[i]))
        test_acc = accuracy(np.concatenate(tuple(y_pred), axis=0),
                            np.concatenate(tuple(y_test), axis=0))

    print "Test negative log-likelihood (offline): %.3f\n" % (test_nll)
    print "Test accuracy (offline): %.3f\n" % (test_acc)
    
    return np.concatenate(tuple(y_prob), axis=0), test_nll, \
        np.concatenate(tuple(y_pred), axis=0), test_acc


def test_online(model, data, eva_hypers):
    """Test a model online (by updating its parameters after each prediction).
    This version does not make a distinction between individual sequences, and
    updates the same model after each prediction.

    Input
    -----
    model : tuple(list(np.ndarray), dict)
      A tuple containing the model parameters, and a dictionary with its
      hyperparameters.
    data : tuple(tuple(np.ndarray), dict)
      A tuple containing test inputs and their corresponding labels, and
      dictionary with information about the data. 
    eva_hypers: dict
      A dictionary which specifies how exactly to evaluate the model.

    Output
    ------
    _ : tuple
      A tuple containing test prediction probabilities, the corresponding cross
      entropy, prediction labels, and the corresponding accuracy.
    """
    validtest_data, dat_hypers = data
    
    y_prob_valid, cr_ent_valid, y_pred_valid, acc_valid = _test_online(
        model, validtest_data[0], dat_hypers, eva_hypers) 
    y_prob_test, cr_ent_test, y_pred_test, acc_test = _test_online(
        model, validtest_data[1], dat_hypers, eva_hypers)

    return ((y_prob_valid, y_prob_test), (cr_ent_valid, cr_ent_test), 
            (y_pred_valid, y_pred_test), (acc_valid, acc_test))


def _test_online(model, test_data, dat_hypers, eva_hypers):
    """Carry out online evaluation on given test data. This is only an internal
    function I wrote because I'd have to do the same thing twice (on validation
    and test data in the function that calls this one).

    Input
    -----
    test_data : tuple
      Test inputs and corresponding labels.
    dat_hypers : dict
      Information about data.

    Output
    ------
    """
    n_input = dat_hypers['n_input'] 
    n_class = dat_hypers['n_class'] 
    X_test = test_data[0]
    y_test = test_data[1]
    model_params, mod_hypers = model

    # Convolved subsequences
    X_test, y_test, is_beg = convolve_sequences(X_test, y_test,
                                                mod_hypers['tr_online'])
    n_test = len(X_test)
    
    # Initialize model and optimizer
    model = RTDRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                    init_params=model_params) 
    optimizer = stoch_gd(eva_hypers)

    # Cross entropy and accuracy computation
    y_prob = []
    y_pred = []
    t_step = 0
    for d_step in xrange(n_test):
        if eva_hypers['method'] == 'semi-online' and is_beg[d_step]:
            print ("Beginning of a new melody at data-point %d. "
                   "Re-initializing online model..." % (d_step))
            model = RTDRBM(n_input=n_input, n_class=n_class, hypers=mod_hypers,
                           init_params=model_params)
            t_step = 0
        else:
            t_step+=1

        y_prob.append(
            model.predict_proba(X_test[d_step].T, 
                                y_test[d_step])[-1, :][np.newaxis, :])
        y_pred.append(
            model.predict_function(X_test[d_step].T,
                                   y_test[d_step])[-1])
        model, _ = optimizer.one_step(model, (X_test[d_step],
                                              y_test[d_step]), t_step)
    test_nll = np.mean(negative_log_likelihood(
        np.concatenate(tuple(y_prob), axis=0),
        np.asarray([y_el[-1] for y_el in y_test], dtype=int)))
    test_acc = np.float(
        np.sum(np.asarray(y_pred, dtype=np.int) ==
               np.asarray([y_el[-1] for y_el in y_test], dtype=int))) / \
        np.shape(np.asarray(y_pred, dtype=np.int))[0]

    return np.concatenate(tuple(y_prob), axis=0), test_nll, \
        np.asarray(y_pred, dtype=int), test_acc


if __name__ == '__main__':
    print "Did not implement a main function here."
