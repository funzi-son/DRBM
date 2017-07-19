# Summary of Results #


## Introduction ##
This document summarises the result on the digit image classification task 
involving the MNIST dataset and the DRBM model. The first step in working with
this dataset and model is to try and reproduce the results in [1] which first
introduced the DRBM. Following this, we also want to evaluate the DRBM with
other types of hidden layers (Hyperbolic Tangent, Binomial, Rectified Linear).


## Highlights ##
Below is a list of the key details of all runs, and their respective goals.

### Sanity Check and Result Reproduction ###

* **run-1:** The first run was a sort of sanity check to see if the code was 
  working, and to get an initial sense of the prediction performance of the 
  model on the dataset.

* **run-2:** The goal of this second run was to try to reproduce the 
  classification result on the MNIST obtained in [1]. For this I used 
  (hopefully) identical parameter settings as those in the paper and learned 
  two models. The performance is still short of that in the paper, but matches 
  earlier runs with SGD optimisation which was used here.

* **run-3:** The previous run confirmed that the code is alright, since Son had 
  very similar results on the MNIST dataset with the DRBM as well. In this run, 
  the intention was to test the performance of the DRBM with hyperbolic tangent 
  (TanH) units on the dataset. This would serve as a comparison between the two 
  types of units.  This is only the first run and based on the success/failure 
  of this, further optimisations will be determined.

* **run-4:** It was observed that, while the experiments carried out here (runs 
  1-3) used mini-batch gradient descent, the error-rate reported in [1] was 
  when the model was trained using stochastic gradient descent (SGD). This is 
  exactly what the present run does.

### {-1,+1}-DRBM/Bipolar DRBM/TanH DRBM Experiments ###

* **run-5:** With the previous run successfully reproducing the classification 
  results of [1], this run was aimed at carrying out SGD learning with the TanH 
  DRBM.  However, during the run it was found that the weights of the models 
  were going to NaN due to improper scale. As the first step to tackle this, 
  weight-decay (which generally tends to worsen the performance in the case of 
  this model and this dataset) was introduced. The results here correspond to 
  this setup.

* **run-6:** Since the idea of introducing weight-decay to SGD with the TanH 
  DRBM to prevent NaN issues did not prove very useful, we simply re-ran the 
  previous experiment with a range of smaller learning rates, no weight-decay 
  and not using momentum while keeping everything else the same. Note that I 
  terminated this run before learning and evaluating the last model could 
  finish. The reason will be explained in the next run.

* **run-7:** Since the models trained in the previous run were optimised with 
  different test and validation set losses, I am re-running the above to see if 
  it is better to do so with the same loss function. I am also reducing the 
  size of the search here. So the main difference between this run and the 
  previous one is that there the validation set loss and test set loss were 
  different and here they are the same.

* **run-8:** Having carried out the last run, and determined hyperparameters
  corresponding to the best models (learning rates: {0.01, 0.001}; hidden
  units: {500, 1000}), this run contains 10 results corresponding to each of
  the four hyperparameter combinations whose average is the performance that
  will be reported for that combination. This run gives the mean and standard
  deviation around the result for each hyperparameter combination.

* **run-9** As an extension of run-8 this run was carried out in order to test
  the same four models from that run, but with different loss functions, namely
  average cross-entropy (ACE) and mean squared-error (MSE). Just for reference,
  all the previous runs optimised negative log-likelihood (NLL).

### {0,...,N}-DRBM/Binomial DRBM ###

* **run-10** This is the first run with Binomial hidden layer activations in
  the DRBM, after finding out that the NaN errors could be bypassed for now by
  using 64-bit floating point precision as opposed to 32-bit. On the other
  hand, the GPU cannot be used as Theano does not support 64-bit precision.

* **run-11** This run is to test the performance of batch learning in the TanH
  DRBMS. In previuos runs, it was found that stochastic learning worked better 
  for the LogSig DRBMs. The same has not yet been examined with other hidden
  layer activations, which is what this run takes a first step towards.

* **run-12** The purpose of this run is to generate the final (reportable)
  results with the Binomial DRBMs on the MNIST dataset. The best four
  hyperparameter combinations from run-10 were used here. It was found that
  there was no significant advantage of using a greater number of bins (or vice
  versa). The average performance seemed better for lesser number of bins,
  however the difference was not significant w.r.t this dataset.

### Comparative Experiments and Filling Gaps ###

* **run-13** The purpose of this run is to generate results with the 
  LogSigU-DRBM which are comparable to the other types of DRBMs. This involves 
  10 seeded runs on the MNIST dataset.

* **run-14** This run sort of fills the gap in previous experiments wherein no
  grid search was carried out on the LogSigU-DRBM in addition to the efforts to
  reproduce the results of previous work.

* **note-1** This note compares the average number of training iterations until 
  convervence for four previously trained models - BinU2-DRBM, BinU4-DRBM, 
  TanHU-DRBM and LogSigU-DRBM. It makes use of the result from run-14 for 
  LogSigU-DRBM.

* **run-15** 


## Notes on Methodology ##
The MNIST dataset contains a single fold of training/validation/test data. For
this reason, in order get an estimate of the deviation of performance about the
mean, multiple runs (typically *10*, unless specified as something else) are 
carried out, each with a different seed for initialising the model parameters. 

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
The first run was a sort of sanity check to see if the code was working, and to
get an initial sense of the prediction performance of the model on the dataset.

### Methodology ###
For early stopping, the performance of the model on a validation set was 
determined every 10 epochs. If this performance happened to be worse than the 
previous best one for 3 consecutive checks, we reverted back to the 
previous best and model and continued training with a reduced learning rate
(see the code in optimize.py for the schedule in use). And if this happened 
three times, training was stopped. The maximum number of training epochs was 
set to 1000, but it was found that learning did not exceed around 500 epochs in 
the case of the model with a Logistic Sigmoid hidden layer, and around 200 for 
the model with the Hyperbolic Tangent hidden layer. Mini-batch gradient descent 
was used. Below are the full details of learning:

* Model:
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0001]
    activation: ['sigmoid', 'tanh']
    output_type: ['softmax']
    seed: [860331]
    initial_momentum: [0.5]  # Unused 
    final_momentum: [0.9]    # Unused 
    momentum_switchover: [5] # Unused 

* Optimisation:
    opt_type: ['batch-gd']
    learning_rate: [0.1]
    schedule: ['power']
    rate_param: [2]
    patience: [2]
    max_epoch: [1000]
    validation_frequency: [10]
    sparse_data: [False]
    batch_size: [400]

### Results ###
* Below is a summary of the results (both models are the DRBM):

       Model        Performance
    (Activation)    (Error (%))
    -----------     -----------
      sigmoid          2.93%
       tanh            2.35%

* And below are results for the best the benchmark models from [1]:

       Model        Performance
    (Activation)    (Error (%))
    ------------    -----------
      sigmoid          1.81%

### Next Steps ###
The results obtained in our run differ from those in the paper, which is not a
good thing since we're using the same model. However, the set of
hyperparameters we use here are not identical to those from the paper so there
is still hope. Furthermore, an older implementation of the model which
contains essentially the same code wrapped in a different structure had
comparable results to the benchmark, so it would be worth re-running the
experiment.

Here are some suggestions:

* It might help to wait longer (more iterations of worse performance) before 
  incrementing the counter. This is because validation error tends to fluctuate 
  quite a bit during learning.

* Evaluate on validation set after every epoch so that we don't miss any good
  model along the way.

* The early exit from learning could be because of the small step checks above.
  It would help to have a minimum number of epochs before which exit would not
  be possible.


## run-2 ##
The goal of this second run was to try to reproduce the classification result
on the MNIST obtained in [1]. For this I used (hopefully) identical parameter
settings as those in the paper and learned two models. The performance is still 
short of that in the paper, but matches earlier runs with SGD optimisation
which was used here.

### Methodology ###
I tried to replicate the settings in [1] for the run. And since there wasn't 
anything said about weight decay, I learned two models, one in which there was 
no weight decay at all, and another in which L1 and L2 coefficients were both 
set to 0.0001. Here are more details on the learning hyperparameters:

* Model:
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0000, 0.0001]
    activation: ['sigmoid']
    output_type: ['softmax']
    seed: [860331]
    initial_momentum: [0.5]  # Unused 
    final_momentum: [0.9]    # Unused 
    momentum_switchover: [5] # Unused 

* Optimisation:
    opt_type: ['batch-gd']
    learning_rate: [0.05]
    schedule: ['linear']
    threshold: [5]
    patience: [2]
    max_epoch: [2000]
    validation_frequency: [10]
    sparse_data: [False]
    batch_size: [400]

### Results ###
* Below is a summary of the results from this run:

   |   Model   |Performance|
   |(Wt. Decay)|(Error (%))|
   |:---------:|:---------:|
   |  0.0001   |   2.65%   |
   |  0.0000   |   2.22%   |

* The error rate of the best DRBM from [1] is 1.81% (+/-0.2%). And that of a 
  previous run with my code using batch gradient descent was 2.25%.

### Next Steps ###
I think we can move on to doing a more comprehensive grid search with the
LogSig and TanH hidden layer models. The best performance can be further
optimised later with other tricks, or even using conjugate gradients instead of
vanilla batch gradient descent (with which I was able to reproduce the results
in [1] previously.

* Try online learning as in Hugo's paper.


## run-3 ##
The previous run confirmed that the code is alright, since Son had very similar
results on the MNIST dataset with the DRBM as well. In this run, the intention
was to test the performance of the DRBM with hyperbolic tangent (TanH) units on
the dataset. This would serve as a comparison between the two types of units.
This is only the first run and based on the success/failure of this, further
optimisations will be determined.

### Methodology ###
Details of the hyperparameter grid are given below:

* Model:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000, 5000]
    weight_decay: [0.0000, 0.0001]
    activation: ['tanh']
    seed: [860331]
    initial_momentum: [0.5]
    final_momentum: [0.9]
    momentum_switchover: [5]

* Optimisation:
    opt_type: ['batch-gd']
    learning_rate: [0.1]
    schedule: ['linear']
    threshold: [5]
    patience: [2]
    max_epoch: [2000]
    validation_frequency: [10]
    sparse_data: [False]
    batch_size: [400]

### Results ###
Below is the summary of results from this run:

    |    Model     |Performance|
    |              |(Error (%))|
    |:------------:|:---------:|
    |(500, 0.0000) |   1.86%   |
    |(100, 0.0000) |   2.19%   |
    |(1000, 0.0000)|   2.06%   |
    |(1000, 0.0001)|   2.38%   |
    |(500, 0.0001) |   2.38%   |
    |(5000, 0.0000)|   2.34%   |
    |(100, 0.0001) |   2.55%   |
    |(5000, 0.0001)|   2.63%   |

* The tuple in the first column represents (hidden layer size, weigth decay).

* The best model (with 500 hidden units and no weight decay) has a performance 
  of 1.86%. This result is better than what we obtained with the DRBM with
  logistic sigmoid units.

* In general, it seems like weight decay is not that useful with this dataset
  and the models of sizes that we've tried so far.

### Next Steps ###
Based on these results, the following can be tried:
* Further optimise the model with other tricks like momentum and dropout.

* Carry out a finer grid search.

* Try other hidden units like binomial and rectified linear.


## run-4 ##
It was observed that, while the experiments carried out here (runs 1-3) used
mini-batch gradient descent, the error-rate reported in [1] was when the model
was trained using stochastic gradient descent (SGD). This is exactly what the
present run does.

### Methodology ###
Most importantly, learning was carried out using Stochastic Gradient Descent 
(SGD), as opposed to Mini-Batch Gradient Descent (MBGD). This was done by 
running the MBGD code with a batch size of 1. The details of the learning 
hyperparameters are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0, 0.0001]
    activation: ['sigmoid']
    seed: [860331]
    initial_momentum: [0.5]
    final_momentum: [0.9]
    momentum_switchover: [5]

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.05]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
The model with weight-decay had an error-rate of 3.98% which is significantly 
worse than the best result reported in [1]. However, on removing weight-decay
the error-rate dropped to 1.65% which is slightly (but not significantly) 
better than the result reported in [1].

### Next Steps ###
Having reproduced the result in [1], and given that the result of the TanH DRBM 
from the previous runs with batch GD is better than the corresponding LogSig
DRBM, it makes sense to test the TanH DRBM on the dataset with SGD.


## run-5 ##
With the previous run successfully reproducing the classification results of 
[1], this run was aimed at carrying out SGD learning with the TanH DRBM.
However, during the run it was found that the weights of the models were going
to NaN due to improper scale. As the first step to tackle this, weight-decay
(which generally tends to worsen the performance in the case of this model and
this dataset) was introduced. The results here correspond to this setup.

### Methodology ###
Details of the hyperparameter grid are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000, 5000]
    weight_decay: [0.0001]
    activation: ['sigmoid']
    seed: [860331]
    initial_momentum: [0.5]
    final_momentum: [0.9]
    momentum_switchover: [5]

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.05]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |Model|Performance|
    |     |(Error (%))|
    |:----|:---------:|
    | 500 |   2.43%   |
    |1000 |   2.48%   |
    |5000 |   2.59%   |
    | 100 |   2.42%   |

### Next Steps ###
So the performance here is obviously not satisfactory, as much as we got rid of
the NaN problem. After some thought and discussion, it was decided that some
options to try are:

* Use a smaller learning rate.
* Use a smaller weight-decay.
* Initialise weights to smaller values.
* Detect NaNs early enough and re-initialise these weights to random values.


## run-6 ##
Since the idea of introducing weight-decay to SGD with the TanH DRBM to prevent
NaN issues did not prove very useful, we simply re-ran the previous experiment 
with a range of smaller learning rates, no weight-decay and not using momentum 
while keeping everything else the same. Note that I terminated this run before
learning and evaluating the last model could finish. The reason will be
explained in the next run.

### Methodology ###
Details of the hyperparameter grid are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000, 5000]
    weight_decay: [0.0]
    activation: ['sigmoid']
    seed: [860331]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.0001, 0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |    Model     |Performance|
    |              |(Error (%))|
    |:------------:|:---------:|
    |(500, 0.01)   |   2.01%   |
    |(500, 0.001)  |   1.93%   |
    |(500, 0.0001) |   2.06%   |
    |(1000, 0.001) |   1.93%   |
    |(1000, 0.01)  |   1.80%   |
    |(100, 0.001)  |   2.25%   |
    |(1000, 0.0001)|   2.12%   |
    |(100, 0.0001) |   2.22%   |
    |(100, 0.01)   |   2.15%   |
    |(5000, 0.001) |   2.38%   |
    |(5000, 0.01)  |   2.44%   |
    |(5000, 0.0001)|   ?.??%   |

* We are indeed able to match the best performing LogSig DRBM with the TanH
  DRBM, however, this does not correspond to the model with the best validation
  set performance. Nevertheless, this is just a single run and we should have
  more conclusive results following multiple runs with different randomisation
  seeds.

* The models with 500 and 1000 hidden units seem most promising so this range
  should be kept in mind.

### Next Steps ###
Ok, so we are actually getting somewhere with this. Carrying out multiple runs
with some of these models to get a more robust estimate of the performance
might just result in something usable. However, we noticed that the validation
loss being computed is negative log-likelihood which is not the same as the
test loss (mean prediction error). So in the next run we will optimise the
model with validation loss and test loss being the same.


## run-7 ##
Since the models trained in the previous run were optimised with different test
and validation set losses, I am re-running the above to see if it is better to
do so with the same loss function. I am also reducing the size of the search
here. So the main difference between this run and the previous one is that
there the validation set loss and test set loss were different and here they 
are the same.

### Methodology ###
Details of the hyperparameter grid are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000, 5000]
    weight_decay: [0.0]
    activation: ['tanh']
    seed: [860331]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |    Model    |Performance|
    |             |(Error (%))|
    |:-----------:|:---------:|
    |(1000, 0.01) |   1.87%   |
    |(500, 0.01)  |   1.74%   |
    |(500, 0.001) |   1.96%   |
    |(1000, 0.001)|   2.01%   |
    |(100, 0.01)  |   2.25%   |
    |(100, 0.001) |   2.23%   |
    |(5000, 0.001)|   2.38%   |
    |(5000, 0.01) |   2.29%   |

* So the best result is more or less the same. We cannot say yet if the
  difference between these and the results of the previous run are significant.

* Again, the models with 500 and 1000 hidden units seem most promising so this 
  range should be kept in mind.

### Next Steps ###
Carrying out multiple runs with some of these models to get a more robust 
estimate of the performance might just result in something usable.


## run-8 ##
Having carried out the last run, and determined hyperparameters corresponding
to the best models (learning rates: {0.01, 0.001}; hidden units: {500, 1000}),
this run contains 10 results corresponding to each of the four hyperparameter
combinations whose average is the performance that will be reported for that
combination. This run gives the mean and standard deviation around the result
for each hyperparameter combination.

### Methodology ###
Details of the hyperparameter grid are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [500, 1000]
    weight_decay: [0.0]
    activation: ['tanh']
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    loss: ['ll']

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |    Model    |  Performance  |
    |             |Mean(%)|Std.(%)|
    |:-----------:|:-----:|:-----:|
    |(500, 0.01)  | 1.84% |0.0007%|
    |(1000, 0.01) | 1.90% |0.0010%|
    |(500, 0.001) | 2.05% |0.0009%|
    |(1000, 0.001)| 2.08% |0.0007%|

* So the best result is as good as the sigmoid DRBM in [1].

* These are the results to be reported.

### Next Steps ###
* Try different optimisation criteria (squared-error, cross-entropy) with the
  above hyperparameters to see if results improve in any way. These were
  discussed with Son.


## run-9 ##
As an extension of run-8 this run was carried out in order to test the same
four models from that run, but with different loss functions, namely average
cross-entropy (ACE) and mean squared-error (MSE). Just for reference, all the
previous runs optimised negative log-likelihood (NLL).

### Methodology ###
This run finalises the results of the previous run by carrying out multiple
runs corresponding to the four best models from the previous run. Further 
details of the run are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [500, 1000]
    weight_decay: [0.0]
    activation: ['tanh']
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    loss: ['ce', 'se']

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |      Model      |  Performance  |
    |                 |Mean(%)|Std.(%)|
    |:---------------:|:-----:|:-----:|
    |(500, 0.01, se)  | 1.91% |0.0007%|
    |(1000, 0.001, ce)| 1.92% |0.0007%|
    |(1000, 0.01, se) | 1.90% |0.0008%|
    |(500, 0.001, ce) | 1.93% |0.0005%|
    |(500, 0.01, ce)  | 1.83% |0.0011%|
    |(1000, 0.01, ce) | 1.92% |0.0008%|
    |(500, 0.001, se) | 2.33% |0.0013%|
    |(1000, 0.001, se)| 2.35% |0.0011%|

* So the best result is still as good as the sigmoid DRBM in [1], and that tanh
  DRBM of the previous run. No improvement to report.

### Next Steps ###
* Try other hidden layer activations to see if the performance can be further
  improved. Discuss with Son.


## run-10 ##
This is the first run with Binomial hidden layer activations in the DRBM, 
after finding out that the NaN errors could be bypassed for now by using 
64-bit floating point precision as opposed to 32-bit. On the other hand, 
the GPU cannot be used as Theano does not support 64-bit precision.

### Methodology ###
An additional hyperparameter that comes into play with the Binomial hidden 
units is the bin size. Ideally, increasing the bin size should result in 
better modelling capabilities. This hyperparameter was varied as {2, 4, 8, 16, 
32, 64}. Details of the learning process are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [2, 4, 8, 16, 32, 64]]
    seed: [860331]

* Optimisation hyperparameters:
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
    batch_size: [1] # This makes it SGD
    nan_check_frequency: [2001]

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |           Model            |Performance|
    |                            |(Error (%))|
    |:--------------------------:|:---------:|
    | (16,  500, 0.0010, 0.0001) |  98.22%   |
    | (16,  500, 0.0010, 0.0000) |  98.11%   |
    | ( 8,  500, 0.0010, 0.0000) |  98.20%   |
    | (32,  500, 0.0001, 0.0001) |  98.03%   |
    | ( 4,  500, 0.0100, 0.0000) |  98.21%   |
    | ( 2,  500, 0.0100, 0.0000) |  98.20%   |
    | ( 4, 1000, 0.0100, 0.0000) |  98.11%   |
    | (32,  500, 0.0010, 0.0001) |  98.02%   |
    | (32, 1000, 0.0010, 0.0001) |  97.97%   |
    | ( 8,  500, 0.0010, 0.0001) |  98.02%   |
    | ( 2, 1000, 0.0100, 0.0000) |  98.09%   |
    | (16,  500, 0.0001, 0.0001) |  98.22%   |
    | ( 8, 1000, 0.0010, 0.0000) |  98.13%   |
    | ( 8, 1000, 0.0010, 0.0001) |  97.96%   |
    | (32, 1000, 0.0001, 0.0001) |  98.12%   |
    | (16,  100, 0.0010, 0.0001) |  98.13%   |
    | ( 8, 1000, 0.0100, 0.0000) |  97.85%   |
    | (64,  500, 0.0001, 0.0001) |  97.82%   |
    | (16, 1000, 0.0010, 0.0000) |  97.96%   |
    | (64, 1000, 0.0001, 0.0001) |  98.11%   |
    | ( 8,  500, 0.0100, 0.0000) |  97.94%   |
    | (16, 1000, 0.0001, 0.0000) |  98.04%   |
    | (16,  500, 0.0001, 0.0000) |  97.95%   |
    | (32,  500, 0.0001, 0.0000) |  97.81%   |
    | (32, 1000, 0.0010, 0.0000) |  98.03%   |
    | (16,  100, 0.0001, 0.0001) |  97.90%   |
    | ( 8,  100, 0.0010, 0.0000) |  97.92%   |
    | (32, 1000, 0.0001, 0.0000) |  98.01%   |
    | ( 4, 1000, 0.0010, 0.0000) |  97.85%   |
    | (32,  100, 0.0010, 0.0001) |  97.73%   |
    | (16, 1000, 0.0001, 0.0001) |  97.96%   |
    | ( 8,  100, 0.0010, 0.0001) |  97.93%   |
    | ( 4,  500, 0.0010, 0.0000) |  97.97%   |
    | ( 4,  100, 0.0010, 0.0000) |  97.54%   |
    | (16,  100, 0.0010, 0.0000) |  97.94%   |
    | (32,  500, 0.0010, 0.0000) |  97.83%   |
    | (16, 1000, 0.0010, 0.0001) |  98.17%   |
    | ( 2,  100, 0.0100, 0.0000) |  97.90%   |
    | ( 8,  100, 0.0001, 0.0000) |  97.81%   |
    | ( 8, 1000, 0.0001, 0.0000) |  97.75%   |
    | ( 8,  500, 0.0100, 0.0001) |  97.70%   |
    | ( 8, 1000, 0.0100, 0.0001) |  97.76%   |
    | ( 4,  100, 0.0100, 0.0000) |  97.83%   |
    | ( 8,  100, 0.0100, 0.0001) |  97.75%   |
    | ( 4, 1000, 0.0100, 0.0001) |  97.75%   |
    | ( 4,  100, 0.0100, 0.0001) |  97.79%   |
    | (64,  500, 0.0001, 0.0000) |  97.60%   |
    | (32,  100, 0.0001, 0.0001) |  97.73%   |
    | ( 8,  100, 0.0001, 0.0001) |  97.58%   |
    | (64, 1000, 0.0001, 0.0000) |  97.94%   |
    | ( 4,  500, 0.0100, 0.0001) |  97.61%   |
    | ( 2,  500, 0.0010, 0.0000) |  97.50%   |
    | ( 8,  500, 0.0001, 0.0000) |  97.67%   |
    | ( 8, 1000, 0.0001, 0.0001) |  97.59%   |
    | ( 2,  100, 0.0010, 0.0000) |  97.56%   |
    | ( 8,  100, 0.0100, 0.0000) |  97.60%   |
    | ( 2, 1000, 0.0010, 0.0000) |  97.60%   |
    | ( 4,  500, 0.0010, 0.0001) |  97.58%   |
    | ( 4, 1000, 0.0010, 0.0001) |  97.53%   |
    | (32,  100, 0.0001, 0.0000) |  97.49%   |
    | (16,  100, 0.0001, 0.0000) |  97.74%   |
    | ( 8,  500, 0.0001, 0.0001) |  97.34%   |
    | (64,  100, 0.0001, 0.0000) |  97.03%   |
    | (64,  100, 0.0001, 0.0001) |  97.64%   |
    | ( 4,  100, 0.0010, 0.0001) |  97.13%   |
    | (32,  100, 0.0010, 0.0000) |  97.21%   |
    | ( 4,  500, 0.0001, 0.0000) |  97.37%   |
    | ( 4,  100, 0.0001, 0.0000) |  96.85%   |
    | (16,  100, 0.0100, 0.0001) |  96.57%   |
    | ( 4,  100, 0.0001, 0.0001) |  96.52%   |
    | ( 2,  100, 0.0100, 0.0001) |  96.57%   |
    | ( 2,  500, 0.0100, 0.0001) |  96.45%   |
    | ( 4,  500, 0.0001, 0.0001) |  96.24%   |
    | (16,  100, 0.0100, 0.0000) |  96.27%   |
    | ( 2,  100, 0.0010, 0.0001) |  95.77%   |
    | (64,  100, 0.0010, 0.0000) |  95.98%   |
    | ( 2,  500, 0.0001, 0.0000) |  95.38%   |
    | ( 4, 1000, 0.0001, 0.0000) |  95.12%   |
    | ( 2, 1000, 0.0010, 0.0001) |  95.03%   |
    | ( 2,  500, 0.0010, 0.0001) |  95.04%   |
    | ( 2,  100, 0.0001, 0.0000) |  95.01%   |
    | ( 2,  100, 0.0001, 0.0001) |  93.53%   |
    | ( 2, 1000, 0.0001, 0.0000) |  92.24%   |
    | ( 4, 1000, 0.0001, 0.0001) |  91.86%   |
    | ( 2,  500, 0.0001, 0.0001) |  91.20%   |
    | ( 2, 1000, 0.0001, 0.0001) |  90.86%   |
    |----------------------------|-----------|
    | (16,  500, 0.0100, 0.0001) |   9.80%   |
    | (16, 1000, 0.0100, 0.0000) |   9.80%   |
    | (16, 1000, 0.0100, 0.0001) |   9.80%   |
    | (64,  100, 0.0010, 0.0001) |   9.80%   |
    | (16,  500, 0.0100, 0.0000) |   9.80%   |
    | (32,  500, 0.0100, 0.0001) |   9.80%   |
    | (64, 1000, 0.0100, 0.0001) |   9.80%   |
    | (64, 1000, 0.0010, 0.0001) |   9.80%   |
    | (64,  100, 0.0100, 0.0001) |   9.80%   |
    | (64, 1000, 0.0010, 0.0000) |   9.80%   |
    | (64, 1000, 0.0100, 0.0000) |   9.80%   |
    | (32,  500, 0.0100, 0.0000) |   9.80%   |
    | (64,  100, 0.0100, 0.0000) |   9.80%   |
    | (64,  500, 0.0100, 0.0000) |   9.80%   |
    | (64,  500, 0.0010, 0.0001) |   9.80%   |
    | (32,  100, 0.0100, 0.0000) |   9.80%   |
    | (32, 1000, 0.0100, 0.0000) |   9.80%   |
    | (64,  500, 0.0100, 0.0001) |   9.80%   |
    | (32,  100, 0.0100, 0.0001) |   9.80%   |
    | (64,  500, 0.0010, 0.0000) |   9.80%   |
    | (32, 1000, 0.0100, 0.0001) |   9.80%   |

* The larger the bin-size, the smaller the preferred learning rate. This is
  understandable as a larger bin-size with a larger learning rate can lead to
  NaN weights very easily. This is becuase the weight-updates will be larger,
  and so they need to be scaled down by a larger factor. And this is actually 
  what happens. 

* One interesting observation is that the best performance is about the same
  irrespective of the number of bins N used. This is barring the case where
  N=64, as this ran into underflow/overflow errors.

* Once again, 500 hidden units seems to be ideal. 8 out of the 10 top models
  use 500 hidden units, and the other two 1000 units.

* Could it be said that one of the advantages of this approach is that there is
  no loss in computational efficiency on increasing bin-size, which should
  ideally result in a better model.

* Models with fewer bins perform better with a larger learning rates, whereas
  those with greater number of bins perform better with relatively smaller 
  learning rates.

* So the best result is more or less the same as the LogSig and TanH DRBMs. We 
  cannot say yet if the difference between these and the results of the 
  previous run are significant.

### Next Steps ###
* The next step is to carry out a run over a subset of the initial grid search
  of this run. For this, since each model will be evaluated 10 times with
  different seeds, it is a good idea to narrow things down based on the
  existing results. This will be based on the following observations regarding
  those hyperparameters which were actually varied in this run:
    - The number of hidden units will be set to only one value,namely 500. This
      is because 8/10 top models used 500 hidden units, and the number of
      hidden units corresponding to the best model for each value of N (number
      of bins) is 500.
    - Number of bins N will be varied as {2, 4, 8, 16} since {32, 64} tended to
      often run into overflow/underflow errors and did not show any significant
      improvement in performance when compared to the smaller values of N.
    - Weight decay will be set to {0.0} as the only purpose it seems to have
      served is to keep SGD for leading to overflow/underflow errors. Moreover,
      a weight decay of 0.0001 was only used in the bese cases of N=32 which we
      decided to exclude from the run anyway.
    - Learning rate will be varied as {0.01, 0.001} as both these learning
      rates were equally likely to result in good models. However, it might
      help to further reduce the number of grid points if 0.01 learning rate is
      used only with N = {2, 4} and 0.001 used only with N = {8, 16} as
      other combinations were not among the top 10.
* So the final grid search (in *run-12*) would be carried out in two sets:
    - Set 1:
        n_hidden: [500]
        learning_rate: [0.01]
        weight_decay: [0.0]
        bin_size: [2, 4]
        seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    - Set 2:
        n_hidden: [500]
        learning_rate: [0.001]
        weight_decay: [0.0]
        bin_size: [8, 16]
        seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  which is a total of 40 points.


## run-11 ##
This run is to test the performance of batch learning in the TanH DRBMS. In 
previuos runs, it was found that stochastic learning worked better for the 
LogSig DRBMs. The same has not yet been examined with other hidden layer 
activations, which is what this run takes a first step towards.

### Methodology ###
Details of the run are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [100, 500, 1000, 5000]
    weight_decay: [0.0000, 0.0001]
    loss: ['ll', 'ce', 'se']
    activation: ['tanh']
    bin_size: [-1]
    seed: [860331]

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.01, 0.001]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [100] # NOTE that this is batch learning
    nan_check_frequency: [2001]


### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |       Model       |Performance|
    |(n_hid, w_dec, los)|(Error (%))|
    |:-----------------:|:---------:|
    |( 100, 0.0000, ll) |   97.26   |
    |( 500, 0.0001, ce) |   97.36   |
    |( 100, 0.0000, ce) |   97.12   |
    |( 500, 0.0000, ce) |   97.23   |
    |(1000, 0.0000, ll) |   97.09   |
    |( 100, 0.0001, ce) |   96.97   |
    |( 500, 0.0000, ll) |   96.61   |
    |(1000, 0.0001, ce) |   96.77   |
    |( 100, 0.0000, se) |   96.52   |
    |( 500, 0.0001, ll) |   96.47   |
    |-------------------------------|
    |( 500, 0.0000, se) |   96.32   |
    |( 100, 0.0001, ll) |   96.15   |
    |(1000, 0.0000, se) |   95.26   |
    |( 500, 0.0000, ce) |   95.05   |
    |( 100, 0.0000, ce) |   95.03   |
    |( 100, 0.0001, ce) |   95.07   |
    |( 100, 0.0000, ll) |   94.85   |
    |( 100, 0.0001, se) |   94.46   |
    |( 100, 0.0001, ll) |   93.48   |
    |(5000, 0.0000, se) |   93.05   |
    |(5000, 0.0000, ll) |   92.47   |
    |(1000, 0.0000, ce) |   92.98   |
    |(5000, 0.0000, ce) |   92.34   |
    |( 500, 0.0001, ce) |   92.39   |
    |(1000, 0.0001, ll) |   91.85   |
    |( 100, 0.0001, se) |   92.01   |
    |(1000, 0.0000, ll) |   91.98   |
    |( 100, 0.0000, se) |   91.80   |
    |(5000, 0.0000, ce) |   91.89   |
    |(1000, 0.0001, ce) |   91.96   |
    |( 500, 0.0001, se) |   91.82   |
    |(1000, 0.0000, ce) |   92.06   |
    |( 500, 0.0001, ll) |   91.71   |
    |(5000, 0.0000, ll) |   91.76   |
    |( 500, 0.0000, ll) |   91.84   |
    |(5000, 0.0001, ce) |   91.58   |
    |(1000, 0.0001, ll) |   91.71   |
    |(5000, 0.0001, ce) |   91.54   |
    |(5000, 0.0000, se) |   91.43   |
    |(1000, 0.0001, se) |   91.62   |
    |(5000, 0.0001, ll) |   91.17   |
    |(5000, 0.0001, ll) |   91.09   |
    |( 500, 0.0001, se) |   90.88   |
    |( 500, 0.0000, se) |   90.86   |
    |(1000, 0.0000, se) |   90.92   |
    |(1000, 0.0001, se) |   90.54   |
    |(5000, 0.0001, se) |   90.11   |
    |(5000, 0.0001, se) |   90.08   |
    |-------------------------------|

* The best error-rate is not better than that obtained with stochastic
  learning.
* Log-likelihood and cross-entropy criteria are more effective than mean
  squared-error.
* Hidden layer sizes of 100 and 500 units are better than 1000, and 5000 is the
  worst.
* I suppose this experiment further supports previous observations that
  stochastic learning works better than mini-batch learning.

### Next Steps ###
I cannot think of anything that can extend this for the moment.


## run-12 ##
The purpose of this run is to generate the final (reportable) results with the 
Binomial DRBMs on the MNIST dataset. The best four hyperparameter combinations 
from run-10 were used here. It was found that there was no significant benefit 
of using a greater number of bins (or vice versa). The average performance 
seemed better for lesser number of bins, however the difference was not 
significant w.r.t this dataset. 

### Methodology ###
This run finalises the results of run-10 by carrying out multiple runs for each 
of the the four best models from that run. Another important hyperparameter 
here is the number of bins (*bin_size*) which was varied as {2, 4, 8, 16} based 
on the results of run-10. The learning rate was set to 0.01 for when *bin_size* 
was {2, 4}, and to 0.001 when it was {8, 16} just to minimise the number of 
runs. Thus this experiment was carried out in two sets. Further details of the 
learning process are given below:

* Model hyperparameters:
    Set 1
    -----
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [2, 4]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Set 2
    -----
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['binomial']
    bin_size: [8, 16]
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

* Optimisation hyperparameters: 
    Set 1
    -----
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

    Set 2
    -----
    opt_type: ['batch-gd']
    learning_rate: [0.001]
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


### Results ###
* Below are the results of this run. Note that all models have 500 hidden units 
  and hence this hyperparameter has not been mentioned. The number of bins and
  learning rate are the ones listed in the table below and the list is sorted 
  according to validation error:

    |    Model  |  Test error   |  Valid error  |
    |           |Mean(%)|Std.(%)|Mean(%)|Std.(%)|
    |:---------:|:-----:|:-----:|:-----:|:-----:|
    |(4, 0.01)  | 1.88% |0.0009%| 1.77% |0.0006%|
    |(2, 0.01)  | 1.86% |0.0008%| 1.78% |0.0016%|
    |(8, 0.001) | 1.90% |0.0006%| 1.80% |0.0005%|
    |(16, 0.001)| 1.92% |0.0006%| 1.83% |0.0007%|

  Given that the significant difference on the MNIST dataset is 0.2%, this is
  just as good as the Hyperbolic Tangent and Logistic Sigmoid DRBMs.

### Next Steps ###
* Evaluate the performance of the ReLU DRBMs and see how they do in comparison
  to the other variants that have been evaluated so far.

* Re-check the experiments and code to make sure there are no trivial mistakes.

* Look into whether this lack of improvement in performance and the similarity
  between the different variants of the DRBM (TanH, LogSig & Bin) is expected.


## run-13 ##
The purpose of this run is to generate results with the LogSigU-DRBM which are
comparable to the other types of DRBMs. This involves 10 seeded runs on the
MNIST dataset.

### Methodology ###
Details of the run are given below:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0]
    activation: ['sigmoid']
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    loss: ['ll']

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.05]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |    Model    |  Performance  |
    |             |Mean(%)|Std.(%)|
    |:-----------:|:-----:|:-----:|
    |(500, 0.05)  | 1.78% |0.0012%|

* So the best result is slightly better than the sigmoid DRBM in [1].

### Next Steps ###
Can't think of any.


## run-14 ## 
This run sort of fills the gap in previous experiments wherein no grid search 
was carried out on the LogSigU-DRBM in addition to the efforts to reproduce the 
results of previous work.

### Methodology ###
Here are the details of the grid search:

* Model hyperparameters:
    model_type: ['drbm']
    n_hidden: [500]
    weight_decay: [0.0]
    activation: ['sigmoid']
    seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    initial_momentum: [0.0]
    final_momentum: [0.0]
    momentum_switchover: [0]
    loss: ['ll']

* Optimisation hyperparameters:
    opt_type: ['batch-gd']
    learning_rate: [0.001, 0.01]
    schedule: ['linear']
    threshold: [9]
    patience: [4]
    max_epoch: [2000]
    validation_frequency: [1]
    sparse_data: [False]
    batch_size: [1] # This makes it SGD

### Results ###
Below is the summary of results from this run (sorted by validation set
performance):

    |    Model    |  Performance  |
    |             |Mean(%)|Std.(%)|
    |:-----------:|:-----:|:-----:|
    |(500, 0.01)  | 1.87% |0.0007%|
    |(500, 0.001) | 2.52% |0.0009%|

* So the best result is slightly worse than sigmoid DRBM in [1].

### Next Steps ###
* Will add these results to note-1.


## note-1 ##
Given that the performance of the TanHU-DRBM and the BinU-DRBM is not
significantly different from the LogSigU-DRBM, it was decided to compare their
respective computational efficiencies. The following table shows the details of
this comparison.

Based on previous runs, the following are the best models

    |    Model    |Hiddens| Rate  |W. Decay|Iter| Test  |
    |             |       |       |        |    |Mean(%)|
    |:-----------:|:-----:|:-----:|:------:|:--:|:-----:|
    |LogSigU-DRBM |  500  | 0.01  | 0.0000 | 88 | 1.87% |
    |TanHU-DRBM   |  500  | 0.01  | 0.0000 | 66 | 1.86% |
    |BinU-DRBM (2)|  500  | 0.01  | 0.0000 | 86 | 1.86% |
    |BinU-DRBM (4)|  500  | 0.01  | 0.0000 | 67 | 1.88% |
    |BinU-DRBM (8)|  500  | 0.001 | 0.0000 | 80 | 1.90% |
    |BinU-DRBM(16)|  500  | 0.001 | 0.0000 | 67 | 1.92% |


## run-15 ##


## Reference ##
[1] Larochelle, H., & Bengio, Y. (2008, July). Classification using 
    Discriminative Restricted Boltzmann Machines. In Proceedings of the 25th
    International Conference on Machine Learning (pp. 536-543). ACM.
