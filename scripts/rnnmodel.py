#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module providing wrapper class around an RNN classifier.

Constants:

Classes:
RNNModel - wrapper class around an RNN classifier

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from evaluate import PRDCT_IDX, TOTAL_IDX
from cPickle import dump
from collections import defaultdict, OrderedDict, Counter
from copy import deepcopy
from datetime import datetime
from itertools import chain

from lasagne.init import HeNormal, HeUniform, Orthogonal
from theano import config, printing, tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import sys
import theano

##################################################################
# Variables and Constants
INF = float("inf")
SEED = 1
RELU_ALPHA = 0.5
HE_NORMAL = HeNormal()
HE_UNIFORM = HeUniform()
HE_UNIFORM_RELU = HeUniform(gain = np.sqrt(2))
HE_UNIFORM_LEAKY_RELU = HeUniform(gain = np.sqrt(2./(1+ (RELU_ALPHA or 1e-6)**2)))
ORTHOGONAL = Orthogonal()
CORPUS_PROPORTION_MAX = 0.95
CORPUS_PROPORTION_MIN = 0.47
INCR_SAMPLE_AFTER = 40
BINOMI = 0.93

# default training parameters
ALPHA = 5e-3
EPSILON = 1e-5
MAX_ITERS = 1000
ADADELTA = 0
SGD = 1
SVM_C = 1.5
L1 = 1e-4
L2 = 1e-5

# default dimension of input vectors
VEC_DIM = 64
# default context window

# initial parameters for uniform distribution
UMIN = -5.
UMAX = 5.

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

# custom function for generating random vectors
np.random.seed(SEED)
RND_VEC = lambda a_dim = VEC_DIM: \
          np.random.uniform(UMIN, UMAX, a_dim).astype(config.floatX)

# symbolic codes for auxiliary vectors
EMP = "___%EMP%___"
UNK = "___%UNK%___"
BEG = "___%BEG%___"
END = "___%END%___"
AUX_VEC_KEYS = [EMP, UNK, BEG, END]

##################################################################
# Methods
def _debug_conv(a_seq, a_seq_len, a_emb, a_conv, a_conv_bias, \
                a_nconv, a_conv_width):
    """Debug character convolutions.

    Args:
    -----
    a_seq: list
      input characters
    a_emb: theano.shared(fmatrix)
      character embeddings

    Returns:
    --------
     (void)

    """
    emb = TT.tensor4(name = "iemb")
    conv = TT.reshape(TT.nnet.conv.conv2d(emb, a_conv), \
                      (a_nconv, a_seq_len - a_conv_width + 1)).T
    conv_max = conv.max(axis = 0) + a_conv_bias
    conv_amax = conv.argmax(axis = 0)
    get_conv = theano.function([emb], [conv, conv_max, conv_amax], \
                               name = "get_conv")
    conv_out, max_out, amax_out = get_conv(a_emb)
    for i, j in enumerate(amax_out):
        print("conv{:d}[{:d}]: {:s} ({:.5f})".format(a_conv_width, i, \
                                                     ''.join(a_seq[j:j + a_conv_width]),
                                                                   max_out[0][i]))
    return max_out[0]

def _floatX(data, a_dtype = config.floatX):
    """Return numpy array populated with the given data.

    @param data - input tensor
    @param dtype - digit type

    @return numpy array populated with the given data

    """
    return np.asarray(data, dtype = a_dtype)

def _rnd_orth_mtx(a_dim):
    """Return orthogonal matrix with random weights.

    @param a_dim - dimensionality of square matrix

    @return orthogonal Theano matrix with random weights

    """
    W = np.random.randn(a_dim, a_dim)
    u, _, _ = np.linalg.svd(W)
    return u.astype(config.floatX)

def _balance_ts(a_ts, a_min, a_class2idx, icnt, a_binom = False):
    """Obtained samples from trainset with balanced classes.

    Args:
    -----
      (list of 2-tuples) a_ts: original (unbalanced) training set
      (int) a_min: minimum number of instances pertaining to one class
      (set of indices) a_class2idx: mapping from classes to particular indices
      (int) a_icnt: iteration counter (used to increase sample sizes)
      (bool) a_binom: use Bernoulli subsampling of training instances

    Yields:
    -------
      2-tuples sampled from a balanced set

    """
    ibinom = iseq = None
    samples = [i for v in a_class2idx.itervalues() for i in v]
    # clever sub-sampling (would probably work with SGD but does not work with AdaDelta)
    # icnt // INCR_SAMPLE_AFTER + 1
    # [i for v in a_class2idx.itervalues() for i in \
        # np.random.choice(v, min(float((icnt + 1) * 0.5) * CORPUS_PROPORTION_MAX * a_min, \
        #                               (float(icnt)/(icnt + 1.)) * len(v)), \
        #                           replace = False)]
    np.random.shuffle(samples)
    for i in samples:
        if a_binom:
            iseq = []
            for iword in a_ts[i][0]:
                ibinom = np.random.binomial(1, BINOMI, len(iword))
                iseq.append([x for x, b in zip(iword, ibinom) if b])
            yield (np.asarray(iseq, dtype = "int32"), a_ts[i][-1])
        else:
            yield a_ts[i]

def _get_minibatches_idx(idx_list, n, mb_size, shuffle = True):
    """Shuffle the dataset at each iteration.

    Args:
    -----
      idx_list (np.list): source list of indices to sample from
      n (int): length of the idx_list
      mb_size (int): size of a minibatch
      shuffle (bool): perform random permutation of source list

    Returns:
    --------
      iterator: 2-tuple with minibatch index and shards of single minibatch indices

    """

    if shuffle:
        np.random.shuffle(idx_list)

    mb_start = 0
    for _ in xrange(n // mb_size):
        yield idx_list[mb_start:mb_start + mb_size]
        mb_start += mb_size

    if (mb_start != n):
        # Make a minibatch out of what is left
        yield idx_list[mb_start:]

def adadelta(tparams, grads, x, y, cost):
    """An adaptive learning rate optimizer

    Args:
    -----
    tpramas: model parameters
    grads: gradients of cost w.r.t to parameres
    x: model inputs
    y: targets
    cost: objective function to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(value = np.asarray(p.get_value() * 0., config.floatX), \
                                  name = '%s_grad' % str(p)) for p in tparams]
    running_up2 = [theano.shared(np.asarray(p.get_value() * 0., config.floatX),
                                 name='%s_rup2' % str(p)) for p in tparams]
    running_grads2 = [theano.shared(value = np.asarray(p.get_value() * 0., config.floatX),
                                    name = '%s_rgrad2' % str(p)) \
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates = zgup + rg2up,
                                    name = "adadelta_f_grad_shared")

    updir = [-TT.sqrt(ru2 + 1e-6) / TT.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams, updir)]

    f_update = theano.function([], [], updates = ru2up + param_up,
                               on_unused_input = "ignore",
                               name = "adadelta_f_update")
    return f_grad_shared, f_update

##################################################################
# Class
class RNNModel(object):
    """Wrapper class around an RNN classifier.

    Instance variables:
    vdim - default dimensionality of embedding vectors
    V - size of fature vocabulary
    n_labels - total number of distinct target labels
    lbl2int - mapping from string classes to integers
    int2lbl - reverse mapping from integers to string classes
    int2coeff - mapping from label integer to its integral coefficient
    feat2idx - mapping from string features to the ondex of their
                learned representations

    Methods:
    train - extract features and adjust parameters of the model
    _reset - clear instance variables

    """

    def __init__(self, a_vdim = VEC_DIM):
        """Class constructor.

        @param a_vdim - default dimensionality of embedding vectors

        """
        self.V = 0              # vocabulary size
        self.n_labels = 0
        self.vdim = a_vdim
        self.max_len = 0        # maximum length of an input item
        # maximum width of a convolution stride
        self.max_conv_len = theano.shared(value = 5, name = "MAX_CONV_LEN")
        # mapping from symbolic representations to indices
        self.lbl2int = dict()
        self.int2lbl = dict()
        self.int2coeff = dict()
        self.feat2idx = dict()
        # NN parameters to be learned
        self._params = []
        # private prediction function
        self._predict = None
        # the parameters below will be initialized during training
        # matrix of items' embeddings (either words or characters) that serve as input to RNN
        self.EMB = self.EMB_I = self.CONV_IN = None
        # output of convolutional layers
        self.CONV2_OUT = self.CONV3_OUT = self.CONV4_OUT = None
        # max-sample output of convolutional layers
        self.CONV2_MAX_OUT = self.CONV3_MAX_OUT = self.CONV4_MAX_OUT = self.CONV_MAX_OUT = None
        # map from hidden layer to output and bias for the output layer
        self.CMO2Y = self.Y_BIAS = self.Y = None
        # the remianing parameters will be initialized immediately
        self._init_params()

    def fit(self, a_trainset, a_path, a_devset = None, \
            a_batch_size = 16, a_optimizer = ADADELTA):
        """Train RNN model on the training set.

        Args:
          set: a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
          str: a_path - path for storing the best model
          set: a_devset - development set as a list of 2-tuples with
                            training instances and classes
          int: a_batch_size - size of single training batch
          method: a_optimizer - optimizer to use (ADADELTA or SGD)

        Returns:
          void:

        """
        if len(a_trainset) == 0:
            return
        # estimate the number of distinct features and the longest sequence
        labels = set()
        featset = set()
        self.max_len = 0
        for w, lbl in a_trainset:
            labels.add(lbl)
            featset.update(w)
            # append auxiliary items to training instances
            w[:0] = [BEG]; w.append(END)
            self.max_len = max(self.max_len, len(w))
        self.V = len(AUX_VEC_KEYS) + len(featset)
        del featset
        self.n_labels = len(labels)

        # initialize embedding matrix for features
        self._init_emb()
        # convert symbolic features in the trainset to indices
        a_trainset = self._digitize_feats(a_trainset, a_add = True)
        # store indices of particular classes in the training set to ease the
        # sampling
        class2indices = defaultdict(list)
        for i, (_, y_i) in enumerate(a_trainset):
            class2indices[y_i].append(i)
        # order instances according to their class indices
        n_items = [len(i) for _, i in sorted(class2indices.iteritems(), \
                                             key = lambda k: k[0])]
        total_items = float(sum(n_items))
        # get the minimum number of instances per class
        min_items = min(n_items)
        # due to unbalanced classes, we have to introduce error coefficients
        err_coeff = [(1. - float(c_i)/total_items) for c_i in n_items]
        self.ERR_COEFF = theano.shared(value = _floatX(err_coeff) , name = "CONV_COEFF")

        # initialize LSTM layer
        cmo_out, hw2_carry_out = self._emb2conv(self.CHAR_INDICES)
        # initialize dropout layer
        # dropout_out = self._init_dropout(lstm_out[-1])
        cmo = self.CMO_COEFF * cmo_out + self.CONV_COEFF * TT.nnet.sigmoid(hw2_carry_out)

        # self.LSTM2I = theano.shared(value = HE_NORMAL.sample((self.n_cmo, self.n_cmo)), \
        #                             name = "LSTM2I")
        # # output bias
        # self.I_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_cmo)).flatten(), \
        #                             name = "I_BIAS")
        # # output layer
        # intermediate = TT.tanh(TT.dot(dropout_out, self.LSTM2I) + self.I_BIAS)

        # # add newly initialized weights to the parameters to be trained
        # self._params += [self.LSTM2I, self.I_BIAS]

        # LSTM debug function
        # lstm_debug = theano.function([self.W_INDICES], lstm_out, name = "lstm_debug")

        # mapping from the LSTM layer to output
        self.CMO2Y = theano.shared(value = HE_NORMAL.sample((self.n_cmo, self.n_labels)), \
                                    name = "CMO2Y")
        # output bias
        self.Y_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_labels)).flatten(), \
                                    name = "Y_BIAS")
        # output layer
        self.Y = TT.nnet.softmax(TT.dot(cmo, self.CMO2Y) + self.Y_BIAS)

        # add newly initialized weights to the parameters to be trained
        self._params += [self.CMO2Y, self.Y_BIAS]

        # correct label
        y = TT.scalar('y', dtype = "int32")

        # cost gradients and updates
        cost = -TT.log(self.Y[0, y]) \
               + L2 * TT.sum([TT.sum(p ** 2) for p in self._params])

        # Alternative cost functions:
        # cost = self.Y.sum() - 2 * self.Y[0, y]
        # cost = 0.5 * ((self.LSTM2Y ** 2).sum() + (self.Y_BIAS ** 2).sum()) + \
        #        SVM_C * (TT.max(1 - self.Y * y, 0) ** 2).sum()

        # AdaDelta:
        gradients = TT.grad(cost, wrt = self._params)
        # define training function and let the training begin
        f_grad_shared, f_update = adadelta(self._params, gradients, \
                                           self.CHAR_INDICES, y, cost)
        # SGD:
        # alpha = TT.scalar("alpha")
        # f_grad = theano.function([self.W_INDICES, y], gradients, name = "f_grad")
        # updates = OrderedDict((p, p - alpha * g) for p, g in zip(self._params , gradients))
        # train = theano.function(inputs  = [self.W_INDICES, y, alpha], \
        #                             outputs = cost, updates = updates)

        # Predictions:
        y_pred = TT.argmax(self.Y, axis = 1)
        # prediction function
        _predict = theano.function([self.CHAR_INDICES], [y_pred], name = "predict")

        if a_devset is not None:
            for w, _ in a_devset:
                # append auxiliary items to training instances
                w[:0] = [BEG]; w.append(END)
            a_devset = self._digitize_feats(a_devset)
            # initialize and populate rho statistics
            rhostat = defaultdict(lambda: [0, 0])
            for _, y in a_devset:
                rhostat[y][TOTAL_IDX] += 1

        time_delta = 0.
        start_time = end_time = None
        N = len(a_trainset)
        idx_list = np.arange(N, dtype="int32")
        mb_size = max(N // 10, 1)

        dev_stat = None
        dev_score = max_dev_score = -INF
        if a_devset is None:
            max_dev_score = dev_score = 0.
        icost = prev_cost = min_train_cost = INF
        print("lbl2int =", repr(self.lbl2int), file = sys.stderr)
        for i in xrange(MAX_ITERS):
            icost = 0.
            start_time = datetime.utcnow()
            # iterate over the training instances
            for x_i, y_i in _balance_ts(a_trainset, min_items, class2indices, i):
                try:
                    icost += f_grad_shared(x_i, y_i)
                    f_update()
                except:
                    print("self.feat2idx =", repr(self.feat2idx), file = sys.stderr)
                    print("x_i =", repr(x_i), file = sys.stderr)
                    print("y_i =", repr(y_i), file = sys.stderr)
                    raise
            # dump the best model
            if icost < min_train_cost:
                # dump trained model to disc, if no development set was supplied
                # sys.setrecursionlimit(1500) # model might be large to save
                if a_devset is None:
                    self._dump(a_path)
                min_train_cost = icost
            if a_devset is not None:
                # reset rho statistics
                for k in rhostat:
                    rhostat[k][PRDCT_IDX] = 0
                # compute rho statistics anew
                for x_i, y_i in a_devset:
                    rhostat[y_i][PRDCT_IDX] += (_predict(x_i)[0] == y_i)
                dev_stat = [float(v[PRDCT_IDX])/float(v[TOTAL_IDX] or 1.) \
                            for v in rhostat.itervalues()]
                dev_score = sum(dev_stat) / float(len(rhostat) or 1)
                print("dev_stat =", repr(dev_stat), file = sys.stderr)
                if dev_score > max_dev_score:
                    self._dump(a_path)
                    max_dev_score = dev_score
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print("Iteration #{:d}: train_cost = {:.10f}, dev_score = {:.5f} ({:.2f} sec);".format(i, \
                                                                                                  icost, \
                                                                                                  dev_score, \
                                                                                                  time_delta), \
                  file = sys.stderr)
            if prev_cost != INF and 0. <= (prev_cost - icost) < EPSILON:
                break
            prev_cost = icost
        print("Minimum train cost = {:.10f}, maximum dev score = {:.10f}".format(min_train_cost, \
                                                                                 max_dev_score))
        return icost

    def predict(self, a_seq):
        """Prediction function

        @param a_seq - input sequence whose class should be predicted

        @return 2-tuple with predicted label and its assigned score

        """
        a_seq[:0] = [BEG]; a_seq.append(END)
        self._activate_predict()
        y, score = self._predict(self._feat2idcs(a_seq))
        return (self.int2lbl[int(y)], score)

    def debug(self, a_seq):
        """Output verbose information in prediction.

        Args:
        -----
        a_seq (matrix): row matrix of words represented as char index vectors

        Returns:
        --------
        (TT.vector): convolution vector

        """
        a_seq[:0] = [BEG]; a_seq.append(END)
        a_dseq = self._feat2idcs(a_seq)
        # output embeddings
        in_len = len(a_dseq)
        emb = self.EMB[self.CHAR_INDICES]
        d_emb = printing.Print("EMB =")(emb)
        debug_emb = theano.function([self.CHAR_INDICES], [d_emb], name = "debug_emb")
        print("*** EMBEDDINGS ***")
        debug_emb(a_dseq)

        # output convolutions
        ee = emb.reshape((1, 1, in_len, self.vdim))
        get_ee = theano.function([self.CHAR_INDICES], [ee], name = "get_ee")
        eemb = get_ee(a_dseq)[0]
        print("*** CONVOLUTIONS(2) ***")
        conv2_out = _debug_conv(a_seq, in_len, eemb, self.CONV2, self.CONV2_BIAS, \
                                self.n_conv2, self.conv2_width)
        print("*** CONVOLUTIONS(3) ***")
        conv3_out = _debug_conv(a_seq, in_len, eemb, self.CONV3, self.CONV3_BIAS, \
                                self.n_conv3, self.conv3_width)
        print("*** CONVOLUTIONS(4) ***")
        conv4_out = _debug_conv(a_seq, in_len, eemb, self.CONV4, self.CONV4_BIAS, \
                                self.n_conv4, self.conv4_width)
        print("*** CONVOLUTIONS(5) ***")
        conv5_out = _debug_conv(a_seq, in_len, eemb, self.CONV5, self.CONV5_BIAS, \
                                self.n_conv5, self.conv5_width)

        # output highways
        print("conv2_out = ", repr(conv2_out), file = sys.stderr)
        print("conv3_out = ", repr(conv3_out), file = sys.stderr)
        conv_max_out = np.concatenate((conv2_out, conv3_out, conv4_out, conv5_out))
        print("conv_max_out = ", repr(conv_max_out), file = sys.stderr)
        cmo = TT.vector(name = "cmo")
        hw2_trans = TT.nnet.hard_sigmoid(TT.dot(cmo, self.HW2_TRANS) + self.HW2_TRANS_BIAS)
        hw2_carry = TT.nnet.relu(cmo * (1. - hw2_trans), alpha = RELU_ALPHA)
        d_hw2_trans = printing.Print("HW2_TRANS =")(hw2_trans)
        debug_hw2_trans = theano.function([cmo], [d_hw2_trans], name = "debug_hw2_trans")
        print("*** HW2 TRANS ***")
        debug_hw2_trans(conv_max_out)
        d_hw2_carry = printing.Print("HW2_CARRY =")(hw2_carry)
        debug_hw2_carry = theano.function([cmo], [d_hw2_carry], name = "debug_hw2_carry")
        print("*** HW2 CARRY ***")
        debug_hw2_carry(conv_max_out)
        print("*** HW OUT ***")
        hw_out = TT.dot(hw2_trans, self.CMO_W) + self.CMO_BIAS
        d_hw_out = printing.Print("HW_OUT =")(hw_out)
        debug_hw_out = theano.function([cmo], [d_hw_out], name = "debug_hw_out")
        debug_hw_out(conv_max_out)
        print("*** PRE_Y ***")
        pre_y = self.CMO_COEFF * hw_out + self.CONV_COEFF * TT.nnet.sigmoid(hw2_carry)
        d_pre_y = printing.Print("HW_OUT =")(pre_y)
        debug_pre_y = theano.function([cmo], [d_pre_y], name = "debug_pre_y")
        debug_pre_y(conv_max_out)

        # output predictions
        self._activate_predict()
        y, score = self._predict(self._feat2idcs(a_seq))
        print("y =", repr(self.int2lbl[int(y)]), file = sys.stderr)
        print("score =", repr(score), file = sys.stderr)


    def _digitize_feats(self, a_trainset, a_add = False):
        """Convert features and target classes to vectors and ints.

        Args:
        -----
        a_trainset (set): training set as a list of 2-tuples with training
                            instances and classes
        a_add (bool): create indices for new features

        Returns:
        --------
          (list): new list of digitized features and classes

        """
        assert self.n_labels > 0, "Invalid number of labels."
        assert self.V > 0, "Invalid size of feature vocabulary."
        ret = []
        clabels = 0
        dlabel = -1; dint = coeff = 1.
        # create a vector for unknown words
        for iseq, ilabel in a_trainset:
            # digitize label and convert it to a vector
            if ilabel is not None:
                if ilabel not in self.lbl2int:
                    try:
                        coeff = int(ilabel)
                    except (AssertionError, ValueError):
                        coeff = 1.
                    self.lbl2int[ilabel] = clabels
                    self.int2lbl[clabels] = ilabel
                    self.int2coeff[clabels] = abs(coeff) + 1
                    clabels += 1
                dint = self.lbl2int[ilabel]
                # dlabel = np.zeros(self.n_labels)
                # dlabel[dint] = 1 * self.int2coeff[dint]
                # for SVM
                # dlabel = np.ones(self.n_labels).astype("int32") * -1
                # dlabel[dint] = 1
                dlabel = dint
            # convert features to indices and append new training
            # instance
            ret.append((np.asarray(self._feat2idcs(iseq, a_add = a_add), \
                                   dtype = "int32"), dlabel))
        return ret

    def _feat2idcs(self, a_seq, a_add = False):
        """Convert features to their indices.

        @param a_seq - sequence of features to be converted
        @param a_add - boolean flag indicating whether features should
                       be added to an internal dictionary

        @return list of lists of feature indices within a given context

        """
        # initialize matrix of feature indices
        findices = []
        # convert features to vector indices
        cfeats = len(self.feat2idx)
        # determine maximum word length in sequence
        ilen = len(a_seq)
        max_len = max(ilen, self.conv2_width, self.conv3_width, \
                      self.conv4_width, self.conv5_width)
        for ichar in a_seq:
            if a_add and ichar not in self.feat2idx:
                self.feat2idx[ichar] = cfeats
                cfeats += 1
            findices.append(self.feat2idx.get(ichar, UNK))
        if ilen < max_len:
            findices += [self.feat2idx[EMP]] * (max_len - ilen)
        return findices

    def _init_params(self):
        """Initialize parameters which are independent of the training data.

        @return \c void

        """
        # auxiliary zero matrix used for padding the input
        self._subzero = TT.zeros((self.max_conv_len, self.vdim))
        # # matrix of char vectors, corresponding to single word
        # self.W_INDICES = TT.imatrix(name = "W_INDICES")
        # matrix of char vectors, corresponding to single word
        self.CHAR_INDICES = TT.ivector(name = "CHAR_INDICES")
        # number of embeddings per training item
        self.M_EMB = TT.shape(self.CHAR_INDICES)[0]
        # number of padding rows
        self.M_PAD = TT.max([self.max_conv_len - self.M_EMB, 0])
        # length of input
        self.IN_LEN = self.M_EMB + self.M_PAD

        ################
        # CONVOLUTIONS #
        ################
        # three convolutional filters for strides of width 2
        self.n_conv2 = 5 # number of filters
        self.conv2_width = 2 # width of stride
        self.CONV2 = theano.shared(value = HE_NORMAL.sample((self.n_conv2, 1, \
                                                             self.conv2_width, self.vdim)), \
                                   name = "CONV2")
        self.CONV2_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_conv2)), \
                                        name = "CONV2_BIAS")
        # four convolutional filters for strides of width 3
        self.n_conv3 = 12 # number of filters
        self.conv3_width = 3 # width of stride
        self.CONV3 = theano.shared(value = HE_NORMAL.sample((self.n_conv3, 1, \
                                                             self.conv3_width, self.vdim)), \
                                   name = "CONV3")
        self.CONV3_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_conv3)), \
                                        name = "CONV3_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv4 = 15 # number of filters
        self.conv4_width = 4 # width of stride
        self.CONV4 = theano.shared(value = HE_NORMAL.sample((self.n_conv4, 1, \
                                                             self.conv4_width, self.vdim)), \
                                   name = "CONV4")
        self.CONV4_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_conv4)), \
                                        name = "CONV4_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv5 = 15 # number of filters
        self.conv5_width = 5 # width of stride
        self.CONV5 = theano.shared(value = HE_NORMAL.sample((self.n_conv5, 1, \
                                                             self.conv5_width, self.vdim)), \
                                   name = "CONV5")
        self.CONV5_BIAS = theano.shared(value = HE_NORMAL.sample((1, self.n_conv5)), \
                                        name = "CONV5_BIAS")
        # remember parameters to be learned
        self._params += [self.CONV2, self.CONV3, self.CONV4, self.CONV5, \
                         self.CONV2_BIAS, self.CONV3_BIAS, self.CONV4_BIAS, self.CONV5_BIAS]
        ############
        # Highways #
        ############
        self.n_cmo = self.n_conv2 + self.n_conv3 + self.n_conv4 + self.n_conv5
        # the 1-st highway links character embeddings to the output
        # self.HW1_TRANS_COEFF = theano.shared(value = _floatX(0.75) , name = "HW1_TRANS_COEFF")
        # self.HW1_TRANS = theano.shared(value = HE_UNIFORM_RELU.sample((self.vdim, self.n_cmo)), \
        #                                name = "HW1_TRANS")
        # self.HW1_TRANS_BIAS = theano.shared(value = \
        #                                     HE_UNIFORM_RELU.sample((1, self.n_cmo)).flatten(), \
        #                                     name = "HW1_TRANS_BIAS")
        # self._params += [self.HW1_TRANS_COEFF, self.HW1_TRANS, self.HW1_TRANS_BIAS]
        # the 2-nd highway links convolutions to the output
        self.HW2_TRANS = theano.shared(value = HE_UNIFORM.sample((self.n_cmo, self.n_cmo)), \
                                       name = "HW2_TRANS")
        self.HW2_TRANS_BIAS = theano.shared(value = \
                                            HE_UNIFORM.sample((1, self.n_cmo)).flatten(), \
                                            name = "HW2_TRANS_BIAS")
        self._params += [self.HW2_TRANS, self.HW2_TRANS_BIAS]
        #######
        # CMO #
        #######
        self.CMO_W = theano.shared(value = _rnd_orth_mtx(self.n_cmo), name = "CMO_W")

        self.CMO_BIAS = theano.shared(HE_NORMAL.sample((1, self.n_cmo)).flatten(), \
                                       name = "CMO_BIAS")
        self._params += [self.CMO_W, self.CMO_BIAS]
        # ########
        # # LSTM #
        # ########
        # self.LSTM_W = theano.shared(value = np.hstack((_rnd_orth_mtx(self.n_lstm) \
        #                                                         for _ in xrange(4))), \
        #                             name = "LSTM_W")
        # self.LSTM_U = theano.shared(value = np.hstack([_rnd_orth_mtx(self.n_lstm) \
        #                                                     for _ in xrange(4)]), \
        #                             name = "LSTM_U")

        # self.LSTM_BIAS = theano.shared(HE_NORMAL.sample((1, self.n_lstm * 4)).flatten(), \
        #                                name = "LSTM_BIAS")
        # self._params += [self.LSTM_W, self.LSTM_U, self.LSTM_BIAS]
        ################
        # Coefficients #
        ################
        # self.EMB_COEFF = theano.shared(value = _floatX(0.25) , name = "EMB_COEFF")
        self.CONV_COEFF = theano.shared(value = _floatX(0.5) , name = "CONV_COEFF")
        self.CMO_COEFF = theano.shared(value = _floatX(1.) , name = "CMO_COEFF")

        self._params += [self.CONV_COEFF, self.CMO_COEFF]
        ###########
        # DROPOUT #
        ###########
        self.use_dropout = theano.shared(_floatX(1.))

    def _init_emb(self):
        """Initialize embeddings.

        @return \c void

        """
        # obtain indices for special embeddings (BEGINNING, END, UNKNOWN)
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
        # create embeddings
        emb = HE_NORMAL.sample((self.V, self.vdim))
        # set EMPTY units to 0
        emb[self.feat2idx[EMP],:] = 0
        self.EMB = theano.shared(value = emb, name = "EMB")
        # add embeddings to the parameters to be trained
        self._params.append(self.EMB)

    def _emb2conv(self, a_x):
        """Compute convolutions from indices

        @param a_x - indices of embeddings

        @return max convolutions computed from these indices

        """
        # length of character input
        in_len = a_x.shape[0]
        # input to convolutional layer
        conv_in = self.EMB[a_x].reshape((1, 1, in_len, self.vdim))
        # first highway passes embeddings to convolutions and directly to the output layer
        # hw1_carry = (1 - self.HW1_TRANS_COEFF) * \
        #             TT.nnet.relu(TT.dot(self.EMB[a_x].mean(axis = 0).reshape((self.vdim,)), \
        #                                 self.HW1_TRANS) + self.HW1_TRANS_BIAS, alpha = RELU_ALPHA)
        # width-2 convolutions
        conv2_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV2), \
                               (self.n_conv2, in_len - self.conv2_width + 1)).T
        conv2_max_out = conv2_out.max(axis = 0) + self.CONV2_BIAS
        # width-3 convolutions
        conv3_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV3), \
                               (self.n_conv3, in_len - self.conv3_width + 1)).T
        conv3_max_out = conv3_out.max(axis = 0) + self.CONV3_BIAS
        # width-4 convolutions
        conv4_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV4), \
                               (self.n_conv4, in_len - self.conv4_width + 1)).T
        conv4_max_out = conv4_out.max(axis = 0) + self.CONV4_BIAS
        # width-5 convolutions
        conv5_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV5), \
                               (self.n_conv5, in_len - self.conv5_width + 1)).T
        conv5_max_out = conv5_out.max(axis = 0) + self.CONV5_BIAS
        # output convolutions
        conv_max_out = TT.concatenate([conv2_max_out, conv3_max_out, \
                                       conv4_max_out, conv5_max_out], axis = 1)
        hw2_trans = TT.nnet.hard_sigmoid(TT.dot(conv_max_out[0,:], self.HW2_TRANS) + \
                                        self.HW2_TRANS_BIAS)
        hw2_carry = TT.nnet.relu(conv_max_out[0,:] * (1. - hw2_trans), alpha = RELU_ALPHA)
        # lstm_in = TT.dot(conv_max_out[0,:], self.LSTM_W) + self.LSTM_BIAS
        cmo = TT.dot(hw2_trans, self.CMO_W) + self.CMO_BIAS
        return cmo, hw2_carry

    # def _init_lstm(self):
    #     """Initialize parameters of LSTM layer.

    #     @return 2-tuple with result and updates of LSTM scan

    #     """
    #     # single LSTM recurrence step function
    #     def _lstm_step(x_, o_, m_, c2_):
    #         """Single LSTM recurrence step.

    #         @param x_ - indices of input characters
    #         @param o_ - previous output
    #         @param m_ - previous state of memory cell
    #         @param c1_ - highway carry (from embeddings)
    #         @param c2_ - highway carry (from convolutions)

    #         @return 2-tuple (with the output and memory cells)

    #         """
    #         # obtain character convolutions for input indices
    #         lstm_in, hw2_carry = self._emb2conv(x_)
    #         # compute common term for all LSTM components
    #         proxy = TT.nnet.sigmoid(lstm_in + TT.dot(o_, self.LSTM_U))
    #         # input
    #         i = proxy[:self.n_lstm]
    #         # forget
    #         f = proxy[self.n_lstm:2*self.n_lstm]
    #         # output
    #         o = proxy[2*self.n_lstm:3*self.n_lstm]
    #         # new state of memory cell (input * current + forget * previous)
    #         m = i * TT.tanh(proxy[3*self.n_lstm:]) + f * m_
    #         # new outout state
    #         o = o * TT.tanh(m)
    #         # return new output and memory state
    #         return o, m, hw2_carry
    #     # `scan' function
    #     res, _ = theano.scan(_lstm_step,
    #                          sequences = [self.W_INDICES],
    #                          outputs_info = [np.zeros(self.n_lstm).astype(config.floatX),
    #                                          np.zeros(self.n_lstm).astype(config.floatX),
    #                                          np.zeros(self.n_lstm).astype(config.floatX)],
    #                                name = "_lstm_layers")
    #     # lstm_in, hw2_carry = self._emb2conv(self.W_INDICES[0])
    #     return (res[0], res[-1])

    def _init_dropout(self, a_input):
        """Create a dropout layer.

        Args:
          a_input (theano.vector): input layer

        Returns:
          theano.vector: dropout layer

        """
        # generator of random numbers
        trng = RandomStreams(SEED)
        # the dropout layer itself
        output = TT.switch(self.use_dropout,
                           (a_input * trng.binomial(a_input.shape,
                                                    p = 0.5, n=1,
                                                    dtype = a_input.dtype)),
                           a_input * 0.5)
        return output

    def _activate_predict(self):
        """Activate prediction function.

        Returns:
        --------
        void

        """
        if self._predict is None:
            # deactivate dropout when using the model
            self.use_dropout.set_value(0.)
            y_pred = TT.argmax(self.Y, axis = 1)
            self._predict = theano.function([self.CHAR_INDICES], \
                                            [y_pred, self.Y[0, y_pred]], \
                                            name = "predict")

    def _dump(self, a_path):
        """Dump this model to disc at the given path.

        @param a_path - path to file in which to store the model

        @return \c void

        """
        with open(a_path, "wb") as ofile:
            dump(self, ofile)

    def _reset(self):
        """Reset instance variables.

        @return \c void

        """
        self._predict = None
