"""Functions to carry out different types of evaluation on the models.

negative_log_likelihood:
  Compute negative log-likelihood of the probability distribution predicted by
  a model, given the target values for certain input.

error:
  Compute prediction error of the predictions of a model.
"""


import numpy as np


def negative_log_likelihood(probs, tgts):
    """Compute negative log-likelihood of the probability distribution
    predicted by a model, given the target values for certain input.

    Input:
    ------
    probs: np.ndarray
      The predicted distributions for each sample of input data.
    tgts: np.ndarray
      Target values corresponding to each input.

    Output:
    -------
    Negative log-likelihood.
    """
    return -np.mean(np.log(probs)[np.arange(tgts.shape[0]), tgts])


def error(y_pred, y_test):
    """Compute the prediction error of a model given the target.

    Input:
    ------
    y_pred: np.ndarray
      The predictions for each sample of input data.
    tgts: np.ndarray
      Target values corresponding to each input.

    Output:
    -------
    Accuracy.
    """
    if len(y_pred.shape) == 2: # Predictions are probabilities
        y_pred = np.argmax(y_pred, axis=-1)
    return np.float(np.sum(y_pred != y_test)) / np.shape(y_pred)[0]

def cross_entropy(y_pred, y_test):
    """Compute the cross-entropy of predictions made by a model given the
    target values.

    Input
    -----
    y_pred: np.ndarray
      The predictions for each sample of input data.
    tgts: np.ndarray
      Target values corresponding to each input.

    Output
    ------
    Cross entropy
    """
    return np.sum(-y_pred*np.log(y_test + np.finfo(float).eps) - \
            (1-y_pred)*np.log(1-y_test + np.finfo(float).eps)) / \
            y_test.shape[0]
