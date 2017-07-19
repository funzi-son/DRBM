"""Various utility functions.

dict_product:
  Convert dict(list) to list(dict) by taking the Cartesian product.

index_to_onehot:
  Convert a sequence of integer indices into a one-hot vector sequence.

onehot_to_index:
  Convert a sequence of one-hot vectors into a sequence of integer indices.
"""


from collections import OrderedDict
from itertools import product
from itertools import izip
import numpy as np
from sklearn.utils import gen_even_slices
import sys

RNG = np.random.RandomState(860331)


def dict_combine(dict1, dict2):
    """Combine the keys of two dictionaries into a single dictionary. It is
    important here that the order of keys is the same each time the same types
    of dictionaries are combined, as they are traversed sequentially to
    generate file names, etc.

    Input
    -----
    dict1 : dict
      First dictionary
    dict2 : dict
      Second dictionary

    Output
    ------
    dictc : dict
      Combined dictionary
    """
    dictc = OrderedDict()
    for k, e in dict1.items() + dict2.items():
        dictc.setdefault(k, e)

    return dictc


def dict_product(dicts):
    """Convert dict(list) to list(dict) by taking the Cartesian product.

    Input
    -----
    dicts: dict
      A dictionary where each key is associated with a list of values.

    Output
    ------
    -----: list
      A list where each element is obtained after the Cartesian product over
      the values associated with different keys.
    """
    return (OrderedDict(izip(dicts, e)) 
            for e in product(*dicts.itervalues()))


def dict_prettyprint(dic, title):
    """Pretty-print a dictionary with a given title.

    Input
    -----
    dic : dict
      Input dictionary.
    title : str
      Title of the print.

    Output
    ------
    Pretty-prints the dictionary
    """
    print title
    for k in dic.keys():
        print "\t%s : %s" % (k, dic[k])
    print ""

def index_to_onehot(idx_seq, n_class):
    """
    Convert a sequence of integer indices into a one-hot vector sequence.

    Input:
    ------
    idx_seq: Sequence of integer indices(positive, and smallest index as 1).
    n_class: Length of one-hot vector.

    Output:
    -------
    onehot_seq: A numpy array of size n_class x m, where m is the length of
                 the sequence.

    Note that this function only works for a single sequence. If multiple
    sequences are input simultaneously, as in multiple viewpoint sequences, it
    won't do the conversion correctly. There may even be an error. I haven't
    checked. I'll do it later if required.
    """
    n_pts, = idx_seq.shape
    onehot_seq = np.zeros((n_pts, n_class))

    assert ~np.any(idx_seq > n_class) == True, \
           "Invalid class-label index to convert to one-hot."

    for i in range(n_pts):
        onehot_seq[i, idx_seq[i]] = 1

    return onehot_seq


def onehot_to_index(onehot_seq):
    """Convert a sequence of one-hot vectors into a sequence of integer
    indices. 

    Input
    -----
    onehot_seq : np.ndarray
      A sequence of one-hot vectors.

    Output
    ------
    np.ndarray
      A vector of integer indices corresponding to the input sequence.
    """
    return np.where(onehot_seq)[1]


def truncate_sequences(x, y=None, seq_len=10):
    """Split sequences into subsequences of fixed length.

    Input
    -----
    x: list(np.ndarray)
      Input sequences
    y: list(np.ndarray)
      Label sequences
    seq_len: int
      Truncation length

    Output
    ------
    x_sub: list(np.ndarray)
      Input sequences
    y_sub: list(np.ndarray)
      Label sequences
    is_beg: list(bool)
      Indicates the beginning of a sequence
    """
    n_sequences = len(x) # y should also have the same length

    x_sub = []
    if y is not None:
        y_sub = []
    is_beg = []
    for s in xrange(n_sequences):
        in_seq = x[s]
        if y is not None:
            lab_seq = y[s]
        len_seq = in_seq.shape[0]
        l = 0
        while l < len_seq:
            x_sub.append(in_seq[l:np.min((l+seq_len, len_seq)), :])
            if y is not None:
                y_sub.append(lab_seq[l:np.min((l+seq_len, len_seq))])

            
            if l == 1:
                is_beg.append(True)
            else:
                is_beg.append(False)
            l += seq_len

    assert np.sum([x_el.shape[0] for x_el in x]) == \
        np.sum([x_el.shape[0] for x_el in x_sub]), \
        "Mismatch between sums of subsequence lengths before and after" \
        "truncation in x and x_sub."
    
    if y is not None:
        assert np.sum([y_el.shape[0] for y_el in y]) == \
            np.sum([y_el.shape[0] for y_el in y_sub]), \
            "Mismatch between sums of subsequence lengths before and after" \
            "truncation in y and y_sub."

    if y is not None:
        assert len(y_sub) == len(x_sub) == len(is_beg), \
            "List length mismatch between x_sub, y_sub and is_beg."
    else:
        assert len(x_sub) == len(is_beg), \
            "List length mismatch between x_sub, y_sub and is_beg."

    if y is not None:
        return x_sub, y_sub, is_beg
    else:
        return x_sub, is_beg

def convolve_sequences(x, y, len_subseq):
    """Convolve over sequences to generate subsequences of fixed length.

    Input
    -----
    x: list(np.ndarray)
      Input sequences
    y: list(np.ndarray)
      Label sequences
    len_subseq: int
      Truncation length

    Output
    ------
    x_sub: list(np.ndarray(float))
      Input subsequences
    y_sub: list(np.ndarray(int))
      Label subsequences
    is_beg: list(bool)
      Indicates the beginning of a sequence
    """
    n_sequences = len(x)

    x_sub = []
    y_sub = []
    is_beg = []
    for s in xrange(n_sequences):
        in_seq = x[s]
        lab_seq = y[s]
        len_seq = in_seq.shape[1]
        l = 1
        while l <= len_seq:
            x_sub.append(in_seq[:, np.max((0, l-len_subseq)):l])
            y_sub.append(lab_seq[np.max((0, l-len_subseq)):l])

            if l == 1:
                is_beg.append(True)
            else:
                is_beg.append(False)
            l += 1

    assert len(x_sub) == np.sum([x_el.shape[1] for x_el in x]), \
        "Event count mismatch between x and x_sub before and after " \
        "convolution." 
    assert len(y_sub) == np.sum([y_el.shape[0] for y_el in y]), \
        "Event count mismatch between y and y_sub before and after " \
        "convolution." 
    assert len(y_sub) == len(x_sub) == len(is_beg), \
        "List length mismatch between x_sub, y_sub and is_beg."

    return x_sub, y_sub, is_beg


def make_batches(data_x, data_y, batch_size):
    """Split dataset into batches.
    """
    n_data = data_x.shape[0]
    # Compute number of minibatches for training, validation and testing.
    n_batches = np.int(np.ceil(np.float(n_data) / batch_size))
    batch_slices = list(gen_even_slices(n_data, n_batches))

    batches_x = [data_x[batch_slice] for batch_slice in batch_slices]
    batches_y = [data_y[batch_slice] for batch_slice in batch_slices]

    return batches_x, batches_y


def resample_data(data_x, data_y, fold_labels):
    """Resample a given dataset (inputs and targets) into folds.

    Input
    -----
    X : list(np.ndarray)
      Data inputs
    y : list(np.ndarray)
      Data labels
    f : list(np.ndarray)
      Fold labels
    """
    n_folds = np.unique(fold_labels).shape[0]
    data_folds = []
    for fld in xrange(n_folds):
        # Prepare training and validation data for this fold
        n_seqs = len(data_x)
        x_trainvalid = [data_x[seq_idx] for seq_idx in xrange(n_seqs)
                        if fold_labels[seq_idx] != fld]
        y_trainvalid = [data_y[seq_idx] for seq_idx in xrange(n_seqs)
                        if fold_labels[seq_idx] != fld]
        x_test = [data_x[seq_idx] for seq_idx in xrange(n_seqs)
                  if fold_labels[seq_idx] == fld]
        y_test = [data_y[seq_idx] for seq_idx in xrange(n_seqs)
                  if fold_labels[seq_idx] == fld]

        test_data = (x_test, y_test)

        n_trainvalid = len(x_trainvalid)
        n_valid = np.max((np.int32(np.ceil(n_trainvalid/20)), 5))
        n_train = n_trainvalid - n_valid

        idxs = RNG.permutation(n_trainvalid)
        x_train = [x_trainvalid[idx] for idx in idxs[:n_train]]
        x_valid = [x_trainvalid[idx] for idx in idxs[n_train:]]
        y_train = [y_trainvalid[idx] for idx in idxs[:n_train]]
        y_valid = [y_trainvalid[idx] for idx in idxs[n_train:]]

        dataset = ((x_train, y_train), (x_valid, y_valid), (x_test, y_test))

        data_folds.append(dataset)

    return data_folds

