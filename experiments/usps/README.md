# Summary of Results #

## Introduction ##
This document summarises the result on the digit image classification task 
involving the USPS dataset and the DRBM model. We want to evaluate the DRBM 
with different types of hidden layers (Hyperbolic Tangent, Binomial, Rectified 
Linear).

## Summary ##
Before going further into the details of each run, it would be good to go 
through the following summary of all the runs:

### Sanity Check ###

* **run-1:** The first run was a sort of sanity check to see how well the
  various models do on the dataset for a start. A fairly broad grid search is
  carried out by varying learning rate, hidden layer size, and hidden layer 
  activation.

### Reportable Result Generation ###

* **run-2:** This run generates (reportable) results of the {-1,+1}-DRBM 
  (Bipolar DRBM) with different numbers of hidden units with all other 
  hyperparameters being kept the same. Four different hidden layer sizes have 
  been considered. Note that so far only one learning rate value has been used 
  going by the observations on MNIST.

* **run-3:** As a continuation of run-2, this run generates reportable results 
  for the original DRBM ({0,1}-DRBM). Once again, only the number of hidden 
  units is varied owing to the first run clearly determining the best learning 
  rate.

* **run-4:** As a continuation of run-3, this run generates reportable results 
  for the Binomial DRBM with two states. Once again, only the number of hidden 
  units is varied owing to the first run clearly determining the best learning 
  rate.

* **run-5:** As a continuation of run-4, this run generates reportable results 
  for the Binomial DRBM with 4 states. Once again, only the number of hidden 
  units is varied owing to the first run clearly determining the best learning 
  rate.

* **run-6:** As a continuation of run-5, this run generates reportable results 
  for the Binomial DRBM with 8 states. Here, both the number of hidden units as 
  well as the learning rate are varied owing to the NaN problems with this 
  model, and the lack of a clear indication that any one learning rate is the 
  best.


## Notes on Methodology ##
The USPS dataset contains a single fold of training/validation/test data. For
this reason, in order get an estimate of the deviation of performance about the
mean, multiple runs (typically *10*, unless specified as something else) are 
carried out, each with a different seed for initialising the model parameters.
It can be viewed as a smaller version of the MNIST dataset. However, unlike the
MNIST this version of the USPS dataset contains real-values in the range [0,1].

The general methodology employed is very similar for all the runs with minor
variations occasionally. This involves a grid search over different values of a 
set of hyperparameters. In the case of the DRBM, the key model hyperparameters 
to consider are *learning rate*, *hidden layer size* and *hidden layer 
activation*. It was observed in some initial runs that stochastic gradient 
descent resulted in better performance than batch gradient descent and the 
former has been adhered to. Weight decay was initially experimented with, and
found to negatively influence the overall result.

Early stopping has been enabled. The upper-limit on the number of training
epochs is set to 2000, but it is often the case that learning ends well before
this limit. A linear schedule was used with the early-stopping mechanism, where 
a counter (starting at 0 and going to a maximum of 2 i.e, in three steps) is 
incremented every time the validation set score is worse than the previous best 
one for 10 (the *threshold* hyperparemeter) consecutive epochs. Each time this 
counter is incremented, the model's parameters are re-initialised to the values 
corresponding to the previous best model and the learning rate is scaled down 
such that the scaling factor increases linearly at each step (1/2, 1/3, 1/4, 
etc.). This process is repeated 5 (the *patience* hyperparameter) times before 
terminating the training procedure. The performance of the model on the
validation set is determined every epoch, unless specified otherwise.



## run-1##
The first run was a sort of sanity check to see how well the various models 
do on the dataset for a start.

### Methodology ###
The initial learning rate was set to 0.01 (based on previous experience with 
them model on MNIST) when it came to all models. Additionally, a learning rate 
of 0.001 was also evaluated with the BinU models given their requirement for 
smaller learning rates as the number of bins increases. Early-stopping is 
enabled. Below are more details of the grid search:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['sigmoid', 'tanh', 'binomial']
    bin_size: [2, 4, 8]
    seed: [0]

* Optimisation:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01] # 0.001 only for 'binomial' activation
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [False]

### Observations ###
Here are some key observations from the results of theis experiment:

* The performance of both the LogSigU-DRBM and the BinU-2-DRBM is identical
  given the same settings of all other hyperparameters.

* When averaged over learning rates and number of hidden units (whose ranges 
  are the same for all the BinU-DRBMs), an increase in the number of bins 
  results in better performance, as below:
    
    |Model |Performance|
    |      |(Error (%))|
    |:----:|:---------:|
    |BinU-2|   7.6%    |
    |BinU-4|   6.9%    |
    |BinU-8|   6.5%    |

* A similar average computed over different number of hidden units for all the
  models can be summarised in the below table:

    |Model  |Performance|
    |       |(Error (%))|
    |:-----:|:---------:|
    |LogSigU|   6.8%    |
    |TanHU  |   6.6%    |
    |BinU-2 |   6.8%    |
    |BinU-4 |   6.3%    |
    |BinU-8 |   6.3%    |

* In terms of the number of hidden units, no clear preference could be
  observed.

### Next Steps ###
* Try larger bin size.
* Get more clarity on best hidden layer sizes.


## run-2 ##
This run generates (reportable) results of the {-1,+1}-DRBM (Bipolar DRBM) with
different numbers of hidden units with all other hyperparameters being kept the
same. Four different hidden layer sizes have been considered. Note that so far
only one learning rate value has been used going by the observations on MNIST.


### Methodology ###
The following hyperparameter grid was used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['tanh']
    bin_size: [-1]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimiser:
    opt_type: ['batch-gd']
    learning_rate: [0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [True]

### Results ###
Below is a table with the list of results from this run:

    |  Model  |Performance|Performance|
    |(Hiddens)|(Test  (%))|(Valid (%))|
    |:-------:|:---------:|:---------:|
    |   500   |   6.49%   |   3.30%   |
    |   100   |   6.35%   |   3.41%   |
    |    50   |   6.46%   |   3.42%   |
    |  1000   |   6.44%   |   3.43%   |

* There seems to be this issue that the validation set performance does not
  exactly reflect the test set performance.

### Next Steps ###
* Carry out similar reportable runs with the DRBM and Binomial DRBM.


## run-3 ##
As a continuation of run-2, this run generates reportable results for the
original DRBM ({0,1}-DRBM). Once again, only the number of hidden units is
varied owing to the first run clearly determining the best learning rate.

### Methodology ###
The following hyperparameter grid was used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['sigmoid']
    bin_size: [-1]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimiser:
    opt_type: ['batch-gd']
    learning_rate: [0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [True]

### Results ###
Below is a table with the list of results from this run:

    |  Model  |Performance|Performance|
    |(Hiddens)|(Test  (%))|(Valid (%))|
    |:-------:|:---------:|:---------:|
    |    50   |   6.90%   |   3.87%   |
    |   100   |   6.88%   |   3.97%   |
    |   500   |   7.28%   |   4.02%   |
    |  1000   |   7.10%   |   4.11%   |

* Once again, the performance on the validation set does not reflect the same
  trend as the test set.
* Performance generally seems worse than that of the {-1,+1}-DRBM.

### Next Steps ###
* Carry out similar reportable runs with the Binomial DRBM.


## run-4 ##
As a continuation of run-3, this run generates reportable results for the
Binomial DRBM with two states). Once again, only the number of hidden units is
varied owing to the first run clearly determining the best learning rate.

### Methodology ###
The following hyperparameter grid was used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [2]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimiser:
    opt_type: ['batch-gd']
    learning_rate: [0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [True]

### Results ###
Below is a table with the list of results from this run:

    |  Model  |Performance|Performance|
    |(Hiddens)|(Test  (%))|(Valid (%))|
    |:-------:|:---------:|:---------:|
    |    50   |   6.90%   |   3.87%   |
    |   100   |   6.88%   |   3.97%   |
    |   500   |   7.28%   |   4.02%   |
    |  1000   |   7.10%   |   4.11%   |

* Once again, the performance on the validation set does not reflect the same
  trend as the test set.
* The results are identical to those of the {0,1}-DRBM, which is expected.

### Next Steps ###
* Carry out similar reportable runs with the Binomial DRBM with more states.


## run-5 ##
As a continuation of run-4, this run generates reportable results for the
Binomial DRBM with 4 states. Once again, only the number of hidden units is
varied owing to the first run clearly determining the best learning rate.

### Methodology ###
The following hyperparameter grid was used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [4]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimiser:
    opt_type: ['batch-gd']
    learning_rate: [0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [True]

### Results ###
Below is a table with the list of results from this run:

    |  Model  |Performance|Performance|
    |(Hiddens)|(Test  (%))|(Valid (%))|
    |:-------:|:---------:|:---------:|
    |  1000   |   6.48%   |   3.24%   |
    |   500   |   6.45%   |   4.25%   |
    |   100   |   6.33%   |   3.25%   |
    |    50   |   6.36%   |   3.33%   |

* Once again, the performance on the validation set does not reflect the same
  trend as the test set. So while the best validation set model contains 1000
  hidden units, the best test set model contains 100 units.

### Next Steps ###
* Carry out similar reportable runs with the Binomial DRBM with more states.


## run-6 ##
As a continuation of run-5, this run generates reportable results for the
Binomial DRBM with 8 states. Here, both the number of hidden units as well as
the learning rate are varied owing to the NaN problems with this model, and
the lack of a clear indication that any one learning rate is the best.

### Methodology ###
The following hyperparameter grid was used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [8]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimiser:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1]
    nan_check_frequency: [2001]
    eval_test: [True]
    
### Results ###
TO BE COMPLETED
Below is a table with the list of results from this run:

    |  Model  |Performance|Performance|
    |(Hiddens)|(Test  (%))|(Valid (%))|
    |:-------:|:---------:|:---------:|
    |  1000   |   6.48%   |   3.24%   |
    |   500   |   6.45%   |   4.25%   |
    |   100   |   6.33%   |   3.25%   |
    |    50   |   6.36%   |   3.33%   |

* Once again, the performance on the validation set does not reflect the same
  trend as the test set. So while the best validation set model contains 1000
  hidden units, the best test set model contains 100 units.

### Next Steps ###
* Carry out similar reportable runs with the Binomial DRBM with more states.
