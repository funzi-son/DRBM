# 20newsgroups dataset experiments #

This folder contains data, results and conclusions of the experiments carried
out on the 20newsgroups dataset as a part of the work related to the DRBM.


## Summary of runs ##

* **run-1:** The first run is a sanity check to see if everything is working ok
  with this new dataset.

* **run-2:** This run attempts to reproduce the result in [1] with the original
  LogSigU-DRBM.

* **run-3:** This run involves the first one-seed grid search with the
  TanHU-DRBM and the BinU-DRBM of 2, 4 and 8 bins.

* **run-4:** Based on the observations of *run-3*, this run carries out a
  10-seed evaluation of the four most promising models of the TanHU-DRBM.

* **run-5:** Based on the observations of *run-3*, this run carries out a
  10-seed evaluation of the four most promising models of the BinU-DRBM of 2
  bins.

* **run-6:** Based on the observations of *run-3*, this run carries out a
  10-seed evaluation of the four most promising models of the BinU-DRBM of 4
  bins.

* **run-7:** Based on the observations of *run-3*, this run carries out a
  10-seed evaluation of the four most promising models of the BinU-DRBM of 8
  bins.

* **run-8:** For the sake of completion, a grid search was carried out on the
  LogSigU-DRBM as well in this experiment, and this is also a seeded run with
  10 different initialisations for each grid point.


## Notes on Dataset ##

## Notes on Methodology ##

## Details of Runs ##

### run-1 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-2 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-3 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-4 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-5 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-6 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-7 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### run-8 ###

#### Grid Search ####

#### Observations ####

#### Future Work ####


### Methodology ###
In order to determine the best model in any particular experiment, a grid 
search was carried out. Three types of hidden layer activations were evaluated 
- logistic sigmoid (LogSigU), hyperbolic tangent (TanHU) and binomial (BinU). 
Furthermore, three variants of the binomial activations were included, namely 
those with 2 (BinU-2), 4 (BinU-4) and 8 (BinU-8) bins. The initial learning 
rate was varied as {0.0001, 0.001, 0.01}. Early-stopping is enabled. For this, 
the performance of the model on a validation set is determined after every 
epoch. A counter known as *patience* (starting at 0 and going to a maximum of 4 
i.e, in five steps) is incremented every time the validation set performance 
was worse than the previous best for 10 (the *threshold* number) consecutive 
epochs. Each time the counter is incremented, the model is re-initialised to 
the previous best one and the initial learning rate is scaled down such that 
the scaling factor is increased *linearly* at each step (1/2, 1/3, 1/4, etc.). 
And if the counter reaches its upper-limit of 4, learning is terminated. The maximum number of 
training epochs was set to 2000. No weight-decay was used, as this seemed to
only worsen the performance in previous runs on the MNIST task. Stochastic
gradient descent optimisation was used as opposed to its mini-batch variant. 
The number of hidden units was varied as {50, 100, 500, 1000} for all models. 
Momentum was disabled.
Below is the detailed list of model and optimiser hyperparameters used:

* Model:
    model_type: ['drbm']
    n_hidden: [50, 100, 500, 1000]
    weight_decay: [0.0]
    loss: ['ll']
    activation: ['sigmoid', 'tanh', 'binomial']
    bin_size: [2, 4, 8] # Applies only to the binomial case, -1 for the rest
    seed: [0]

* Optimisation:
    opt_type: ['batch-gd']
    learning_rate: [0.0001, 0.001, 0.01]
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
This is only an initial run based on which further runs are to be carried out.
The analysis carried out will thus be coarse and help decide how to refine the
future experiments. It will mainly focus on the learning rate and the number of
hidden units of the various models.

Here are some key observations from the results of theis experiment:

* For the TanHU-DRBM, relatively smaller learning rates of 0.0001 and 0.001 
  seem to work best. And in general, the best performance was also obtained 
  with smaller hidden layer sizes of 50 and 100 units.

* For the BinU-DRBM of 2 bins, relatively larger learning rates of 0.001 and
  0.01 seem to work best. The conclusion regarding the number of hidden units 
  is the same as that for the TanHU-DRBM.

* For the BinU-DRBM of 4 bins, once again relatively smaller learning rates of 
  0.001 and 0.0001 seem more promising while once again smaller hidden layers 
  seem to be a better choice.

* For the BinU-DRBM of 8 bins, the conclusion regarding the learning rate and
  the number of hidden units is the same as that of 4 bins above.

### Next Steps ###
Based on the above observations, the following can be recommended for future
runs with each of the four models evaluated here.

* *TanHU-DRBM:*

* *BinU-DRBM (2):*

* *BinU-DRBM (4):*

* *BinU-DRBM (8):*


## run-4 ##
Seeded run for TanHU-DRBM

## run-5 ##
Seeded run for BinU-2-DRBM

## run-6 ##
Seeded run for BinU-4-DRBM
