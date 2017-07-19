"""Optimizers for gradient-based learning of model parameters.
"""

import cPickle
import numpy as np
import os
import sys
import theano
import theano.tensor as T
import time

from IO import generate_file_name

sys.setrecursionlimit(10000)
RNG = np.random.RandomState(860331)


def initialise_metric(metric_name):
    """Initialise the evaluation metric for validation and testing.

    Input
    -----
    metric_name : str
      Name of an evaluation metric.

    Output
    ------
    metric : Python function
      A function implementing the metric.
    """
    if metric_name == 'ce':
        from evaluate import cross_entropy
        return cross_entropy
    elif metric_name == 'er':
        from evaluate import error 
        return error
    elif metric_name == 'll':
        from evaluate import negative_log_likelihood
        return negative_log_likelihood
    else:
        raise NotImplementedError


class gradient_descent(object):
    """Batch gradient descent optimizer class definition.
    
    Optimization hyperparameters in use at the moment:
      learning_rate : float
        Step-size for model parameter updates (suggested value: 0.01).
      threshold : int
        Threshold for early-stopping (suggested value 9).
      patience : int
        Patience for early-stopping.
      max_epoch: int
        Maximum number of epochs (suggested value: 100).
      initial_momentum : float
        Nesterov momentum to speed up learning (suggested value: 0.5).
      final_momentum : float
        Nesterov momentum to speed up learning (suggested value: 0.9).
      momentum_switchover : int 
        Iteration in which to switch from initial to final momentum
        (suggested value: 5)
    """
    def __init__(self, model, opt_hypers):
        """Constructs a gradient_descent class with the given hyperparameters.

        Input
        -----
        self : class instance
          Optimiser class
        model : class instance
          Model class
        opt_hypers : dictionary
          Optimization hyperparameters. 
        """
        self.hypers=opt_hypers
        self.uid = model.uid + generate_file_name('', opt_hypers, '', '')
        self.model = model

        # Compute parameter gradients w.r.t cost.
        lr = T.scalar('learning_rate', dtype=theano.config.floatX)
        mom = T.scalar('momentum', dtype=theano.config.floatX)

        # Most recent updates to use with momentum
        last_updates = [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                      dtype=theano.config.floatX)) 
                        for param in model.params]

#        try:
#            grads = T.grad(model.cost, model.params,
#                               consider_constant=[model.v_sample])
#        except AttributeError:
#            grads = T.grad(model.cost, model.params)
        grads = model.grads

        updates_train = []
        for (update, param, grad) in zip(last_updates, \
                model.params, grads):
            updates_train.append((param, param + mom*update - lr*grad))
            updates_train.append((update, mom*update - lr*grad))

        # Functions for training the model.
        try:
            self.train_function = theano.function(
                    [model.x, model.y, lr, mom], model.cost, 
                    updates=updates_train, allow_input_downcast=True)
        except TypeError:
            self.train_function = theano.function(
                    [model.x, lr, mom], model.cost, 
                    updates=updates_train, allow_input_downcast=True)

        # Evaluation metrics
        self.validation_metric = initialise_metric(
                self.hypers['validation_metric'])


    def learn_supervised(self, X_train, y_train, learning_rate,
                         effective_momentum):
        """Compute cost for a model that does supervised learning.

        Input
        -----
        self : Theano optimiser class
          A supervised learning model.
        X_train : list(np.ndarray)
          Training inputs.
        y_train : list(np.ndarray)
          Training targets.
        learning_rate : float
          Learning rate.
        effective_momentum : float
          Momentum.

        Output
        ------
        np.mean(costs) : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        RNG.seed(0xbeef); RNG.shuffle(y_train)
        costs = []

        # Son' debug
        stepc = 0
        for X, y in zip(X_train, y_train):
            #Son' debug
            stepc = stepc + 1

            isnan = False
            print ('Step',stepc)
            ens  = self.model.get_energies(X)

            logexp = self.model.get_logexp(X)

    
   #         if np.isnan(logexp).any():
   #             print 'NaN found in log_exp'
   #             isnan = True
            if np.isnan(ens[0]).any():
                print 'NaN found in energies'
                isnan = True
            if np.isnan(ens[1]).any():
                print 'NaN found in max energies'
                isnan = True
            logp = self.model.get_logp(X)
            if np.isnan(logp).any():
                print 'NaN found in logp'
                isnan = True
            if isnan:
                exit()
            
            costs.append(self.train_function(X, y, learning_rate,
                                             effective_momentum))
        
        return np.mean(costs)

    def learn_unsupervised(self, X_train, learning_rate,
                           effective_momentum):
        """Compute cost for a model that does unsupervised learning.

        Input
        -----
        model : Theano model class
          A supervised learning model.
        X_train : list(np.ndarray)
          Training inputs.
        learning_rate : float
          Learning rate.
        effective_momentum : float
          Momentum.

        Output
        ------
        np.mean(costs : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        costs = []

        for X in X_train:
            costs.append(self.train_function(X, learning_rate,
                                             effective_momentum))
            
            return np.mean(costs)


    def optimize(self, dataset):
        """Learn a model using batch gradient descent.

        Input
        -----
        self : Python optimiser class 
          Definition of model.
        dataset : tuple(tuple(np.ndarray))
          Each tuple contains inputs and targets for training, validation and
          test data.

        Output
        ------
        best_model_params: list(np.ndarray)
          List of all the model parameters
        best_valid_score: float
          Best validation set score
        """
        # Load training and validation data, and check for sparsity
        try:
            X_train, y_train = dataset[0]
            X_valid, y_valid = dataset[1]
            learning_type = 'supervised'
        except ValueError:
            X_train, = dataset[0]
            X_valid, = dataset[1]
            learning_type = 'unsupervised'

        # Generate file name to save intermediate training models
        os.mkdir('.' + self.uid)
        temp_file_name = os.path.join('.' + self.uid, 'best_model.pkl')

        max_epoch = self.hypers['max_epoch']
        n_valid = len(X_valid)

        # Initialize learning rate and schedule
        learning_rate = self.hypers['learning_rate']
        threshold = self.hypers['threshold']
        if self.hypers['schedule'] == 'constant':
            rate_update = lambda coeff: self.hypers['learning_rate']
        elif self.hypers['schedule'] == 'linear':
            rate_update = lambda coeff: self.hypers['learning_rate'] \
                    / (1+coeff)
        elif self.hypers['schedule'] == 'exponential':
            rate_update = lambda coeff: self.hypers['learning_rate'] \
                    / 10**(coeff/self.threshold)
        elif self.hypers['schedule'] == 'power':
            rate_update = lambda coeff: self.hypers['learning_rate'] \
                    / (1 + coeff/self.threshold)
        elif self.hypers['schedule'] == 'inv-log':
            rate_update = lambda coeff: self.hypers['learning_rate'] \
                    / (1+np.log(coeff+1)) 
        else:
            raise NotImplementedError

        validation_frequency = self.hypers['validation_frequency']
        best_valid_score = np.inf

        cPickle.dump(self.model, open(temp_file_name, 'wb'))
        best_params = self.model.get_model_parameters()
        
        # Early stopping parameters and other checks
        nan_check_frequency = self.hypers['nan_check_frequency']
        patience = self.hypers['patience']
        pissed_off = 0

#        start_time = time.clock()
        epoch_time = []
        for epoch in xrange(max_epoch):
            epoch_start_time = time.clock()
            # Check if any of the model parameters are NaN. If so, exit.
            if (epoch+1) % nan_check_frequency == 0:
                cur_params = self.model.get_model_parameters()
                nan_params = np.any(np.asarray(
                        [np.any(np.isnan(p)) 
                         for p in cur_params], dtype=bool))
                if nan_params:
                    print "NaN detected in parameters! Exiting..."
                    break
                
            # Set effective momentum for the current epoch.
            effective_momentum = self.hypers['final_momentum'] \
                    if epoch > self.hypers['momentum_switchover'] \
                    else self.hypers['initial_momentum']
            
            # Check if it's time to stop learning.
            if threshold == 0:
                if pissed_off == patience: # Exit and return best model
                    self.model = cPickle.load(open(temp_file_name, 'rb'))
                    best_params = self.model.get_model_parameters()
                    
                    print('Learning terminated after %d epochs.\n' % (epoch+1))
                    break
                else: # Reload previous best model and continue
                    pissed_off+=1
                    learning_rate = rate_update(pissed_off)
                    threshold = self.hypers['threshold']
                    self.model = cPickle.load(open(temp_file_name, 'rb'))

                    print('Re-initialising to previous best model with '
                          'validation prediction score %.3f.\n'
                          '\tCurrent pissed off level: %d/%d.\n'
                          '\tCurrent learning rate: %.4f.\n'
                          % (best_valid_score, pissed_off, patience, 
                             learning_rate))

            if learning_type == 'supervised':
                mean_train_score = self.learn_supervised(X_train, y_train, 
                                                         learning_rate,
                                                         effective_momentum)
            elif learning_type == 'unsupervised':
                mean_train_score = self.learn_unsupervised(X_train, 
                                                           learning_rate,
                                                           effective_momentum)
            else:
                raise NotImplementedError
            
            print('Epoch %i/%i, train score: %.3f' %
                  (epoch+1, max_epoch, mean_train_score))

            if (epoch + 1) % validation_frequency == 0:
                # Compute validation negative log-likelihood
                valid_pred = []
                for i in xrange(n_valid):
                    valid_pred.append(
                        self.model.predict_function(X_valid[i]))

                if learning_type == 'supervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(y_valid), axis=0))
                elif learning_type == 'unsupervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(X_valid), axis=0))

                print("\tValidation score: %.3f (previous best: %.3f)" %
                      (this_valid_score, best_valid_score))
                print "\tCurrent learning rate: %.4f" % (learning_rate)

                # TODO: For some scores, a lower value is better, address this.
                if this_valid_score < best_valid_score:
                    best_valid_score = this_valid_score
                    best_params = self.model.get_model_parameters()
                    cPickle.dump(self.model, open(temp_file_name, 'wb'))
                    threshold = self.hypers['threshold']
                else:
                    threshold-=1
            epoch_end_time = time.clock()
            epoch_time.append(epoch_end_time-epoch_start_time)
#        end_time = time.clock()
        print('\nAverage time per epoch = %.3f\n' % (np.mean(epoch_time)))
        print('\nStd. dev. of time per epoch = %.3f\n' % (np.std(epoch_time)))
#        print('\nTime taken to train model for %d epochs: %.3f\n' %
#              (max_epoch, end_time-start_time))
#        print('\nTime per epoch: %.3f\n' % ((end_time-start_time)/(epoch+1)))

        # Clean temporary files and folder
        os.remove(temp_file_name)
        os.rmdir('.' + self.uid)

        return best_params, best_valid_score


class adadelta(object):
    """Adadelta optimizer class definition.
    
    Optimization hyperparameters in use at the moment:
      threshold : int
        Threshold for early-stopping (suggested value 9).
      patience : int
        Patience for early-stopping.
      max_epoch: int
        Maximum number of epochs (suggested value: 100).
    """
    def __init__(self, model, opt_hypers):
        """Constructs an adadelta class with the given hyperparameters.

        Input
        -----
        self : class instance
          Optimiser class
        model : class instance
          Model class
        opt_hypers : dictionary
          Optimization hyperparameters. 
        """
        self.hypers=opt_hypers
        self.uid = model.uid + generate_file_name('', opt_hypers, '', '')
        self.model = model

        # Compute parameter gradients w.r.t cost.
#        grads = T.grad(model.cost, model.params)
        grads = model.grads

        zipped_grads = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]
        running_up2 = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]
        running_grads2 = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(model.params, updir)]

        updates_train = zgup + rg2up + ru2up + param_up

        # Functions for training the model.
        try:
            self.train_function = theano.function(
                    [model.x, model.y], model.cost, 
                    updates=updates_train, allow_input_downcast=True)
        except:
            self.train_function = theano.function(
                    [model.x], model.cost, 
                    updates=updates_train, allow_input_downcast=True)

        # Evaluation metrics
        self.validation_metric = initialise_metric(
                self.hypers['validation_metric'])


    def learn_supervised(self, X_train, y_train):
        """Compute cost for a model that does supervised learning.

        Input
        -----
        self : class instance
          Optimiser class.
        X_train : list(np.ndarray)
          Training inputs.
        y_train : list(np.ndarray)
          Training targets.

        Output
        ------
        np.mean(costs) : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        RNG.seed(0xbeef); RNG.shuffle(y_train)
        costs = []

        for X, y in zip(X_train, y_train):
            costs.append(self.train_function(X, y))
        
        return np.mean(costs)

    def learn_unsupervised(self, X_train):
        """Compute cost for a model that does unsupervised learning.

        Input
        -----
        self : class instance
          A optimiser class instance.
        X_train : list(np.ndarray)
          Training inputs.

        Output
        ------
        np.mean(costs : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        costs = []

        for X in X_train:
            logp = self.get_logp
            print logp.shape
            costs.append(self.train_function(X))
            
        return np.mean(costs)


    def optimize(self, dataset):
        """Learn a model using batch gradient descent.

        Input
        -----
        model : Python class 
          Definition of model.
        dataset : tuple(tuple(np.ndarray))
          Each tuple contains inputs and targets for training, validation and
          test data.

        Output
        ------
        best_model_params: list(np.ndarray)
          List of all the model parameters
        best_valid_score: float
          Best validation set score
        """
        # Load training and validation data, and check for sparsity
        try:
            X_train, y_train = dataset[0]
            X_valid, y_valid = dataset[1]
            learning_type = 'supervised'
        except ValueError:
            X_train, = dataset[0]
            X_valid, = dataset[1]
            learning_type = 'unsupervised'

        # Generate file name to save intermediate training models
        os.mkdir('.' + self.uid)
        temp_file_name = os.path.join('.' + self.uid, 'best_model.pkl')

        max_epoch = self.hypers['max_epoch']
        n_valid = len(X_valid)

        validation_frequency = self.hypers['validation_frequency']
        best_valid_score = np.inf

        cPickle.dump(self.model, open(temp_file_name, 'wb'))
        best_params = self.model.get_model_parameters()
        
        # Early stopping parameters and other checks
        nan_check_frequency = self.hypers['nan_check_frequency']
        patience = self.hypers['patience']
        threshold = self.hypers['threshold']
        pissed_off = 0

#        start_time = time.clock()
        epoch_time = []
        for epoch in xrange(max_epoch):
            epoch_start_time = time.clock()
            # Check if any of the model parameters are NaN. If so, exit.
            if (epoch+1) % nan_check_frequency == 0:
                cur_params = self.model.get_model_parameters()
                nan_params = np.any(np.asarray(
                        [np.any(np.isnan(p)) 
                         for p in cur_params], dtype=bool))
                if nan_params:
                    print "NaN detected in parameters! Exiting..."
                    break
                
            # Check if it's time to stop learning.
            if threshold == 0:
                if pissed_off == patience: # Exit and return best model
                    self.model = cPickle.load(open(temp_file_name, 'rb'))
                    best_params = self.model.get_model_parameters()
                    
                    print('Learning terminated after %d epochs.\n' % (epoch+1))
                    break
                else: # Reload previous best model and continue
                    pissed_off+=1
                    threshold = self.hypers['threshold']
                    self.model = cPickle.load(open(temp_file_name, 'rb'))

                    print('Re-initialising to previous best model with '
                          'validation prediction score %.3f.\n'
                          '\tCurrent pissed off level: %d/%d.\n'
                          % (best_valid_score, pissed_off, patience))

            if learning_type == 'supervised':
                mean_train_score = self.learn_supervised(X_train, y_train)
            elif learning_type == 'unsupervised':
                mean_train_score = self.learn_unsupervised(X_train)
            else:
                raise NotImplementedError
            
            print('Epoch %i/%i, train score: %.3f' %
                  (epoch+1, max_epoch, mean_train_score))

            if (epoch + 1) % validation_frequency == 0:
                # Compute validation negative log-likelihood
                valid_pred = []
                for i in xrange(n_valid):
                    valid_pred.append(
                        self.model.predict_function(X_valid[i]))

                if learning_type == 'supervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(y_valid), axis=0))
                elif learning_type == 'unsupervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(X_valid), axis=0))

                print("\tValidation score: %.3f (previous best: %.3f)" %
                      (this_valid_score, best_valid_score))

                # TODO: For some scores, a lower value is better, address this.
                if this_valid_score < best_valid_score:
                    best_valid_score = this_valid_score
                    best_params = self.model.get_model_parameters()
                    cPickle.dump(self.model, open(temp_file_name, 'wb'))
                    threshold = self.hypers['threshold']
                else:
                    threshold-=1
            epoch_end_time = time.clock()
            epoch_time.append(epoch_end_time-epoch_start_time)
#        end_time = time.clock()
        print('\nAverage time per epoch = %.3f\n' % (np.mean(epoch_time)))
        print('\nStd. dev. of time per epoch = %.3f\n' % (np.std(epoch_time)))
#        print('\nTime taken to train model for %d epochs: %.3f\n' %
#              (max_epoch, end_time-start_time))
#        print('\nTime per epoch: %.3f\n' % ((end_time-start_time)/(epoch+1)))

        # Clean temporary files and folder
        os.remove(temp_file_name)
        os.rmdir('.' + self.uid)

        return best_params, best_valid_score


class rmsprop(object):
    """RMS-prop optimizer class definition.
    
    Optimization hyperparameters in use at the moment:
      threshold : int
        Threshold for early-stopping (suggested value 9).
      patience : int
        Patience for early-stopping.
      max_epoch: int
        Maximum number of epochs (suggested value: 100).
    """
    def __init__(self, model, opt_hypers):
        """Constructs an adadelta class with the given hyperparameters.

        Input
        -----
        self : class instance
          Optimiser class
        model : class instance
          Model class
        opt_hypers : dictionary
          Optimization hyperparameters. 
        """
        self.hypers=opt_hypers
        self.uid = model.uid + generate_file_name('', opt_hypers, '', '')
        self.model = model

        # Compute parameter gradients w.r.t cost.
#        grads = T.grad(model.cost, model.params)
        grads = model.grads

        zipped_grads = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]
        running_grads = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]
        running_grads2 = \
                [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g) 
                for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        updir = [theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                        dtype=theano.config.floatX))
                 for param in model.params]
        updir_new = [(ud, 0.9*ud - 1e-4*zg / T.sqrt(rg2 - rg**2 + 1e-4))
                     for ud, zg, rg, rg2 in zip(updir, zipped_grads, 
                                                running_grads, running_grads2)]
        param_up = [(p, p + udn[1])
                    for p, udn in zip(model.params, updir_new)]

        updates_train = zgup + rgup + rg2up + updir_new + param_up

        # Functions for training the model.
        try:
            self.train_function = theano.function(
                    [model.x, model.y], model.cost, 
                    updates=updates_train, allow_input_downcast=True)
        except TypeError:
            self.train_function = theano.function(
                    [model.x], model.cost, 
                    updates=updates_train, allow_input_downcast=True)

        # Evaluation metrics
        self.validation_metric = initialise_metric(
                self.hypers['validation_metric'])


    def learn_supervised(self, X_train, y_train):
        """Compute cost for a model that does supervised learning.

        Input
        -----
        self : class instance
          Optimiser class.
        X_train : list(np.ndarray)
          Training inputs.
        y_train : list(np.ndarray)
          Training targets.

        Output
        ------
        np.mean(costs) : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        RNG.seed(0xbeef); RNG.shuffle(y_train)
        costs = []

        for X, y in zip(X_train, y_train):
            costs.append(self.train_function(X, y))
        
        return np.mean(costs)

    def learn_unsupervised(self, X_train):
        """Compute cost for a model that does unsupervised learning.

        Input
        -----
        self : class instance
          A optimiser class instance.
        X_train : list(np.ndarray)
          Training inputs.

        Output
        ------
        np.mean(costs : float
          Mean cost over batch.
        """
        RNG.seed(0xbeef); RNG.shuffle(X_train)
        costs = []

        for X in X_train:
            costs.append(self.train_function(X))
        return np.mean(costs)


    def optimize(self, dataset):
        """Learn a model using batch gradient descent.

        Input
        -----
        model : Python class 
          Definition of model.
        dataset : tuple(tuple(np.ndarray))
          Each tuple contains inputs and targets for training, validation and
          test data.

        Output
        ------
        best_model_params: list(np.ndarray)
          List of all the model parameters
        best_valid_score: float
          Best validation set score
        """
        # Load training and validation data, and check for sparsity
        try:
            X_train, y_train = dataset[0]
            X_valid, y_valid = dataset[1]
            learning_type = 'supervised'
        except ValueError:
            X_train, = dataset[0]
            X_valid, = dataset[1]
            learning_type = 'unsupervised'

        # Generate file name to save intermediate training models
        os.mkdir('.' + self.uid)
        temp_file_name = os.path.join('.' + self.uid, 'best_model.pkl')

        max_epoch = self.hypers['max_epoch']
        n_valid = len(X_valid)

        validation_frequency = self.hypers['validation_frequency']
        best_valid_score = np.inf

        cPickle.dump(self.model, open(temp_file_name, 'wb'))
        best_params = self.model.get_model_parameters()
        
        # Early stopping parameters and other checks
        nan_check_frequency = self.hypers['nan_check_frequency']
        patience = self.hypers['patience']
        threshold = self.hypers['threshold']
        pissed_off = 0

#        start_time = time.clock()
        epoch_time = []
        for epoch in xrange(max_epoch):
            epoch_start_time = time.clock()
            # Check if any of the model parameters are NaN. If so, exit.
            if (epoch+1) % nan_check_frequency == 0:
                cur_params = self.model.get_model_parameters()
                nan_params = np.any(np.asarray(
                        [np.any(np.isnan(p)) 
                         for p in cur_params], dtype=bool))
                if nan_params:
                    print "NaN detected in parameters! Exiting..."
                    break
                
            # Check if it's time to stop learning.
            if threshold == 0:
                if pissed_off == patience: # Exit and return best model
                    self.model = cPickle.load(open(temp_file_name, 'rb'))
                    best_params = self.model.get_model_parameters()
                    
                    print('Learning terminated after %d epochs.\n' % (epoch+1))
                    break
                else: # Reload previous best model and continue
                    pissed_off+=1
                    threshold = self.hypers['threshold']
                    self.model = cPickle.load(open(temp_file_name, 'rb'))

                    print('Re-initialising to previous best model with '
                          'validation prediction score %.3f.\n'
                          '\tCurrent pissed off level: %d/%d.\n'
                          % (best_valid_score, pissed_off, patience))

            if learning_type == 'supervised':
                mean_train_score = self.learn_supervised(X_train, y_train)
            elif learning_type == 'unsupervised':
                mean_train_score = self.learn_unsupervised(X_train)
            else:
                raise NotImplementedError
            
            print('Epoch %i/%i, train score: %.3f' %
                  (epoch+1, max_epoch, mean_train_score))

            if (epoch + 1) % validation_frequency == 0:
                # Compute validation negative log-likelihood
                valid_pred = []
                for i in xrange(n_valid):
                    valid_pred.append(
                        self.model.predict_function(X_valid[i]))

                if learning_type == 'supervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(y_valid), axis=0))
                elif learning_type == 'unsupervised':
                    this_valid_score = self.validation_metric( 
                        np.concatenate(tuple(valid_pred), axis=0),
                        np.concatenate(tuple(X_valid), axis=0))

                print("\tValidation score: %.3f (previous best: %.3f)" %
                      (this_valid_score, best_valid_score))

                # TODO: For some scores, a lower value is better, address this.
                if this_valid_score < best_valid_score:
                    best_valid_score = this_valid_score
                    best_params = self.model.get_model_parameters()
                    cPickle.dump(self.model, open(temp_file_name, 'wb'))
                    threshold = self.hypers['threshold']
                else:
                    threshold-=1
            epoch_end_time = time.clock()
            epoch_time.append(epoch_end_time-epoch_start_time)
#        end_time = time.clock()
        print('\nAverage time per epoch = %.3f\n' % (np.mean(epoch_time)))
        print('\nStd. dev. of time per epoch = %.3f\n' % (np.std(epoch_time)))
#        print('\nTime taken to train model for %d epochs: %.3f\n' %
#              (max_epoch, end_time-start_time))
#        print('\nTime per epoch: %.3f\n' % ((end_time-start_time)/(epoch+1)))

        # Clean temporary files and folder
        os.remove(temp_file_name)
        os.rmdir('.' + self.uid)

        return best_params, best_valid_score
