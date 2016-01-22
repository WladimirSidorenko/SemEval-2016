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
RELU_ALPHA = 0.
HE_NORMAL = HeNormal()
HE_UNIFORM = HeUniform()
HE_UNIFORM_RELU = HeUniform(gain = np.sqrt(2))
HE_UNIFORM_LEAKY_RELU = HeUniform(gain = np.sqrt(2./(1+ (RELU_ALPHA or 1e-6)**2)))
ORTHOGONAL = Orthogonal()

# corpus sampling
BINOMI = 0.4
CORPUS_PROPORTION_MAX = 0.95
CORPUS_PROPORTION_MIN = 0.47
RESAMPLE_AFTER = 40
INCR_SAMPLE_AFTER = 150
MAX_ITERS = 2000

# default training parameters
ALPHA = 5e-3
EPSILON = 1e-5
ADADELTA = 0
SGD = 1
SVM_C = 0.5
L1 = 1e-4
L2 = 1e-5

# default dimension of input vectors
VEC_DIM = 32
# default context window

# initial parameters for uniform distribution
UMIN = -5.
UMAX = 5.

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

# custom function for generating random vectors
# np.random.seed(SEED)
np.random.seed()
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
def _debug_conv(a_seq, a_conv_width, a_conv, a_conv_max):
    """Debug character convolutions.

    Args:
    -----
    a_seq: list
      input characters
    a_conv_width: int
      width of the convolution
    a_conv: theano.shared(fmatrix)
      matrix of character convolutions
    a_conv_max: theano.shared(fvector)
      vector of maximum character convolutions

    Returns:
    --------
     (void)

    """
    conv = TT.matrix(name = "conv")
    amax = conv.argmax(axis = 0)
    get_amax = theano.function([conv], [amax], name = "get_amax")
    amax_out = get_amax(a_conv)
    for i, j in enumerate(amax_out[0]):
        print("conv{:d}[{:d}]: {:s} ({:.5f})".format(a_conv_width, i, \
                                                     ''.join(a_seq[j:j + a_conv_width]),
                                                                   a_conv_max[0][i]))

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

def _resample_words(a_rnn, a_dseq, a_seq, a_re, a_words):
    """Randomly replace words in a sequence with other words from the same set.

    Args:
    -----
      a_dseq (list(int)):
        the same string converted to ist of feature indices
      a_seq (str):
        original string used for matching
      a_re (re):
        regexp that matches words from the set
      a_words (list(str)):
        list of words to sample from

    Returns:
    --------
      (2-tuple): modified a_seq and a_dseq

    """
    ldseq = len(a_dseq)
    lseq = len(a_seq)
    assert ldseq == lseq or (ldseq > lseq and a_dseq[lseq] == 0), \
        "Unequal number of elements in dseq '{:s}' and seq '{:s}'".format(repr(a_dseq), \
                                                                           repr(a_seq))
    iword = ""
    seq = a_seq
    istart = iend = delta = 0
    # binomial and random throws and word index from set
    bthrow = uthrow = w_i = 0
    for imatch in a_re.finditer(a_seq):
        # replace word if Bernoulli tells us to do so
        if np.random.binomial(1, BINOMI, 1):
            # choose new word from a set using Uniform
            w_i = np.random.randint(0, len(a_words), 1)
            iword = a_words[w_i]
            # estimate the indices for replacement
            istart, iend = imatch.start(1), imatch.end(1)
            istart += delta; iend += delta
            # do the actual replacement
            seq = seq[:istart] + iword + seq[iend:]
            a_dseq[istart:iend] = a_rnn._feat2idcs(iword, a_fill = False)
            # update the offsets
            delta += len(iword) - (iend - istart)
    if seq != a_seq:
        print("*** a_seq =", repr(a_seq), file = sys.stderr)
        print("*** seq =", repr(seq), file = sys.stderr)
        print("*** len(seq) =", repr(len(seq)), file = sys.stderr)
        print("*** len(a_dseq) =", repr(len(a_dseq)), file = sys.stderr)
        ldseq = len(a_dseq)
        lseq = len(seq)
        assert ldseq == lseq or (ldseq > lseq and a_dseq[lseq] == 0), \
            "Got unequal number of elements in dseq '{:s}' and seq '{:s}'".format(repr(a_dseq), \
                                                                                  repr(seq))
    #     sys.exit(66)
    return a_dseq, seq

def _balance_ts(a_ts, a_min, a_class2idx, icnt, a_binom = False, \
                a_rnn = None, a_ts_orig = None, \
                a_pos_re = None, a_pos = None, \
                a_neg_re = None, a_neg = None):
    """Obtained samples from trainset with balanced classes.

    Args:
    -----
      (list of 2-tuples) a_ts: original (unbalanced) training set
      (int) a_min: minimum number of instances pertaining to one class
      (set of indices) a_class2idx: mapping from classes to particular indices
      (int) a_icnt: iteration counter (used to increase sample sizes)
      (bool) a_binom: use Bernoulli subsampling of training instances
      (RNNModel) a_rnn: RNNModel
      (set) a_ts_orig: original trainset (not converted to features)
      (re) a_pos_re: regexp matching positive terms
      (set) a_pos: set of positive terms
      (re) a_pos_re: regexp matching negative terms
      (set) a_neg: set of negative terms

    Yields:
    -------
      2-tuples sampled from a balanced set

    """
    ibinom = dseq = iseq = None
    # clever sub-sampling (would probably work with SGD but does not work with AdaDelta)
    # icnt = icnt // INCR_SAMPLE_AFTER + 1
    # samples= [i for v in a_class2idx.itervalues() for i in \
    #     np.random.choice(v, min(float((icnt + 1) * 0.5) * CORPUS_PROPORTION_MAX * a_min, \
    #                                   (float(icnt)/(icnt + 1.)) * len(v)), \
    #                               replace = False)]
    # less clever sub-sampling but ir works in practice
    icnt = 1
    samples= [i for v in a_class2idx.itervalues() for i in \
        np.random.choice(v, min(float((icnt + 1) * 0.5) * CORPUS_PROPORTION_MAX * a_min, \
                                len(v)), \
                         replace = False)]
    # np.random.shuffle(samples)
    for i in samples:
        if a_binom:
            dseq = deepcopy(a_ts[i][0])
            iseq = deepcopy(a_ts_orig[i][0])
            dseq, iseq = _resample_words(a_rnn, dseq, iseq, a_pos_re, a_pos)
            dseq, _ = _resample_words(a_rnn, dseq, iseq, a_neg_re, a_neg)
            yield (np.asarray(dseq, dtype = "int32"), a_ts[i][-1])
        else:
            yield (np.asarray(a_ts[i][0], dtype = "int32"), a_ts[i][-1])

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
        self.n_cmo = self.n_i0_conv = 0 # size of output convolutions
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
        self.CONV3_OUT = self.CONV4_OUT = self.CONV5_OUT = None
        # max-sample output of convolutional layers
        self.CONV3_MAX_OUT = self.CONV4_MAX_OUT = self.CONV5_MAX_OUT = self.CONV_MAX_OUT = None
        # output of intermediate convolutional layers
        self.I0_CONV3_OUT = self.I0_CONV4_OUT = self.I0_CONV5_OUT = None
        # max-sample output of intermediate convolutional layers
        self.I0_CONV3_MAX_OUT = self.I0_CONV4_MAXOUT = self.I0_CONV5_MAXOUT = None
        # map from hidden layer to output and bias for the output layer
        self.I0 = self.Y = None
        self.I0_BIAS = self.Y_BIAS = None
        # the remianing parameters will be initialized immediately
        self._init_params()

    def fit(self, a_trainset, a_path, a_devset = None, \
            a_batch_size = 16, a_optimizer = ADADELTA, **a_kwargs):
        """Train RNN model on the training set.

        Args:
          set: a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
          str: a_path - path for storing the best model
          set: a_devset - development set as a list of 2-tuples with
                            training instances and classes
          int: a_batch_size - size of single training batch
          method: a_optimizer - optimizer to use (ADADELTA or SGD)
          dict: a_kwargs - additional keyword arguments

        Returns:
          void:

        """
        if len(a_trainset) == 0:
            return
        # estimate the number of distinct features and the longest sequence
        labels = set()
        featset = set()
        for w, lbl in a_trainset:
            labels.add(lbl)
            featset.update(w)
            # append auxiliary items to training instances
            # w[:0] = [BEG]; w.append(END)
        pos_re, pos = a_kwargs["a_pos"]
        neg_re, neg  = a_kwargs["a_neg"]
        featset |= set([c for w in (pos | neg) for c in w])
        pos = list(pos); neg = list(neg)
        self.V = len(AUX_VEC_KEYS) + len(featset)
        del featset
        self.n_labels = len(labels)
        # initialize embedding matrix for features
        self._init_emb()
        # convert symbolic features in the trainset to indices
        N = len(a_trainset)
        trainset = self._digitize_feats(a_trainset, a_add = True)
        # convert raw character lists back to strings in the original trainset
        a_trainset = [(''.join(inst), lbl) for inst, lbl in a_trainset]
        # store indices of particular classes in the training set to ease the
        # sampling
        class2indices = defaultdict(list)
        for i, (_, y_i) in enumerate(trainset):
            class2indices[y_i].append(i)
        # order instances according to their class indices
        n_items = [len(i) for _, i in sorted(class2indices.iteritems(), \
                                             key = lambda k: k[0])]
        total_items = float(sum(n_items))
        # get the minimum number of instances per class
        min_items = min(n_items)

        # initialize convolution layer
        self._emb2conv(self.CHAR_INDICES)

        # initialize dropout layer
        # dropout_out = self._init_dropout(lstm_out[-1])

        # convolution mapping from the CMO layer
        self._conv2conv()

        # mapping from convolutions to output
        self.I02Y = theano.shared(value = HE_UNIFORM.sample((self.n_i0_conv, self.n_labels)), \
                                    name = "I02Y")
        # output bias
        self.Y_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_labels)).flatten(), \
                                    name = "Y_BIAS")
        # add newly initialized weights to the parameters to be trained
        self._params += [self.I02Y, self.Y_BIAS]
        self.Y = TT.nnet.softmax(TT.dot(self.I0[0,:], self.I02Y) + self.Y_BIAS)

        # correct label
        y = TT.scalar('y', dtype = "int32")
        # predicted label
        self._activate_predict()

        # cost gradients and updates
        cost = -TT.log(self.Y[0, y]) \
               + L2 * TT.sum([TT.sum(p ** 2) for p in self._params])

        # Alternative cost functions:
        # cost = self.Y.sum() - 2 * self.Y[0, y]
        # ones = theano.shared(value = _floatX(np.ones((1, self.n_labels))), name = "ones")
        # zeros = theano.shared(value = _floatX(np.zeros((1, self.n_labels))), name = "ones")
        # cost = SVM_C * (TT.max([1 - self.Y[y], 0]) ** 2 + TT.max([1 + self.Y[1-y], 0]) ** 2) \
        #        + L2 * TT.sum([TT.sum(p ** 2) for p in self._params])
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

        if a_devset is not None:
            # for w, _ in a_devset:
            #     # append auxiliary items to training instances
            #     w[:0] = [BEG]; w.append(END)
            a_devset = self._digitize_feats(a_devset)
            # initialize and populate rho statistics
            rhostat = defaultdict(lambda: [0, 0])
            for _, y in a_devset:
                rhostat[y][TOTAL_IDX] += 1

        time_delta = 0.
        start_time = end_time = None
        idx_list = np.arange(N, dtype="int32")
        mb_size = max(N // 10, 1)

        ts = None
        dev_stat = None
        dev_score = max_dev_score = -INF
        if a_devset is None:
            max_dev_score = dev_score = 0.
        icost = prev_cost = min_train_cost = INF
        print("lbl2int =", repr(self.lbl2int), file = sys.stderr)
        for i in xrange(MAX_ITERS):
            icost = 0.
            if (i % RESAMPLE_AFTER) == 0:
                ts = [ti for ti in _balance_ts(trainset, min_items, class2indices, i, \
                                               True, self, a_trainset, pos_re, pos, neg_re, neg)]
            start_time = datetime.utcnow()
            # iterate over the training instances
            np.random.shuffle(ts)
            for x_i, y_i in ts:
                try:
                    # i0 = debug_i0(x_i)
                    # print("debug_i0 =", repr(i0), file = sys.stderr)
                    # print("debug_i02y =", repr(debug_i02y(i0[0])), file = sys.stderr)
                    # sys.exit(66)
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
                    rhostat[y_i][PRDCT_IDX] += (self._predict(x_i)[0] == y_i)
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
        # a_seq[:0] = [BEG]; a_seq.append(END)
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
        # a_seq[:0] = [BEG]; a_seq.append(END)
        a_dseq = self._feat2idcs(a_seq)
        # output embeddings
        d_emb = printing.Print("EMB =")(self.CONV_IN)
        debug_emb = theano.function([self.CHAR_INDICES], [d_emb], name = "debug_emb")
        print("*** EMBEDDINGS ***")
        debug_emb(a_dseq)

        get_conv_in = theano.function([self.CHAR_INDICES], \
                                      [self.CONV_IN, self.IN_LEN], name = "get_conv_in")
        conv_in, in_len = get_conv_in(a_dseq)
        # output convolutions
        print("*** CONVOLUTIONS(2) ***")
        print("conv_in =", repr(conv_in))
        print("in_len =", repr(in_len))
        print("*** CONVOLUTIONS(3) ***")
        get_conv3 = theano.function([self.CONV_IN, self.IN_LEN], \
                                    [self.CONV3_OUT, self.CONV3_MAX_OUT], \
                                    name = "get_conv3")
        conv3_out, conv3_max_out = get_conv3(conv_in, in_len)
        _debug_conv(a_seq, self.conv3_width, conv3_out, conv3_max_out)
        print("*** CONVOLUTIONS(4) ***")
        get_conv4 = theano.function([self.CONV_IN, self.IN_LEN], \
                                    [self.CONV4_OUT, self.CONV4_MAX_OUT], \
                                    name = "get_conv4")
        conv4_out, conv4_max_out = get_conv4(conv_in, in_len)
        _debug_conv(a_seq, self.conv4_width, conv4_out, conv4_max_out)
        print("*** CONVOLUTIONS(5) ***")
        get_conv5 = theano.function([self.CONV_IN, self.IN_LEN], \
                                    [self.CONV5_OUT, self.CONV5_MAX_OUT], \
                                    name = "get_conv4")
        conv5_out, conv5_max_out = get_conv5(conv_in, in_len)
        _debug_conv(a_seq, self.conv5_width, conv5_out, conv5_max_out)

        # concatenated convolution layer
        get_conv_max_out = theano.function([self.CONV3_MAX_OUT, \
                                            self.CONV4_MAX_OUT, self.CONV5_MAX_OUT], \
                                           [self.CONV_MAX_OUT], \
                                           name = "get_conv_max_out")
        conv_max_out = get_conv_max_out(conv3_max_out, \
                                            conv4_max_out, conv5_max_out)[0]
        print("*** CONV_MAX_OUT ***\n", repr(conv_max_out), file = sys.stderr)

        # output highways
        get_hw_trans = theano.function([self.CONV_MAX_OUT], [self.HW2_TRANS], \
                                       name = "get_hw_trans")
        hw_trans = get_hw_trans(conv_max_out)[0]
        print("*** HW_TRANS ***\n", repr(hw_trans), file = sys.stderr)

        get_hw_carry = theano.function([self.CONV_MAX_OUT], [self.HW2_CARRY], \
                                       name = "get_hw_carry")
        hw_carry = get_hw_carry(conv_max_out)[0]
        print("*** HW_CARRY ***\n", repr(hw_trans), file = sys.stderr)
        # output CMO and final predictions
        get_cmo = theano.function([self.CONV_MAX_OUT, self.HW2_TRANS, self.HW2_CARRY], \
                                  [self.CMO], name = "get_cmo")
        cmo = get_cmo(conv_max_out, hw_trans, hw_carry)
        print("*** CMO ***\n", repr(cmo), file = sys.stderr)

        # output CMO2Y and Y_BIAS
        print("*** I0_BIAS ***\n", repr(self.I0_BIAS.get_value()), file = sys.stderr)
        get_i0 = theano.function([self.CMO], [self.I0], name = "get_i0")
        i0 = get_i0(cmo[0])
        print("*** I0 ***\n", repr(i0), file = sys.stderr)

        # output final predictions
        print("*** I02Y ***\n", repr(self.I02Y.get_value()), file = sys.stderr)
        print("*** Y_BIAS ***\n", repr(self.Y_BIAS.get_value()), file = sys.stderr)
        get_y = theano.function([self.I0], [self.Y], name = "get_y")
        print("*** Y ***\n", repr(get_y(i0)), file = sys.stderr)

        # re-check predictions
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
            ret.append((self._feat2idcs(iseq, a_add = a_add), dlabel))
        return ret

    def _feat2idcs(self, a_seq, a_add = False, a_fill = True):
        """Convert features to their indices.

        @param a_seq - sequence of features to be converted
        @param a_add - boolean flag indicating whether features should
                       be added to an internal dictionary
        @param a_fill - add empty characters to fit maximum convolution width

        @return list of lists of feature indices within a given context

        """
        # initialize matrix of feature indices
        findices = []
        # convert features to vector indices
        cfeats = len(self.feat2idx)
        # determine maximum word length in sequence
        ilen = len(a_seq)
        if a_fill:
            max_len = max(ilen, self.conv3_width, \
                          self.conv4_width, self.conv5_width)
        else:
            max_len = ilen
        for ichar in a_seq:
            if a_add and ichar not in self.feat2idx:
                self.feat2idx[ichar] = cfeats
                cfeats += 1
            findices.append(self.feat2idx.get(ichar, self.feat2idx[UNK]))
        if ilen < max_len:
            findices += [self.feat2idx[EMP]] * (max_len - ilen)
        return findices

    def _init_params(self):
        """Initialize parameters which are independent of the training data.

        @return \c void

        """
        # matrix of char vectors, corresponding to single word
        self.CHAR_INDICES = TT.ivector(name = "CHAR_INDICES")

        ################
        # CONVOLUTIONS #
        ################
        # three convolutional filters for strides of width 2
        # four convolutional filters for strides of width 3
        self.n_conv3 = 6 # number of filters
        self.conv3_width = 3 # width of stride
        self.CONV3 = theano.shared(value = HE_UNIFORM.sample((self.n_conv3, 1, \
                                                             self.conv3_width, self.vdim)), \
                                   name = "CONV3")
        self.CONV3_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_conv3)), \
                                        name = "CONV3_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv4 = 15 # number of filters
        self.conv4_width = 4 # width of stride
        self.CONV4 = theano.shared(value = HE_UNIFORM.sample((self.n_conv4, 1, \
                                                             self.conv4_width, self.vdim)), \
                                   name = "CONV4")
        self.CONV4_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_conv4)), \
                                        name = "CONV4_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv5 = 20 # number of filters
        self.conv5_width = 5 # width of stride
        self.CONV5 = theano.shared(value = HE_UNIFORM.sample((self.n_conv5, 1, \
                                                             self.conv5_width, self.vdim)), \
                                   name = "CONV5")
        self.CONV5_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_conv5)), \
                                        name = "CONV5_BIAS")
        # remember parameters to be learned
        self._params += [self.CONV3, self.CONV4, self.CONV5, \
                         self.CONV3_BIAS, self.CONV4_BIAS, self.CONV5_BIAS]
        ############
        # Highways #
        ############
        self.n_cmo = self.n_conv3 + self.n_conv4 + self.n_conv5
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
        self.CMO_W_HRZ = theano.shared(value = _rnd_orth_mtx(self.n_cmo), name = "CMO_W_HRZ")
        self.CMO_W_VRT = theano.shared(value = _rnd_orth_mtx(self.n_cmo), name = "CMO_W_VRT")
        self.CMO_W_DIAG = theano.shared(value = _rnd_orth_mtx(self.n_cmo), name = "CMO_W_DIAG")

        self.CMO_BIAS = theano.shared(HE_UNIFORM.sample((1, self.n_cmo)).flatten(), \
                                       name = "CMO_BIAS")
        self._params += [self.CMO_W_HRZ, self.CMO_W_VRT, self.CMO_W_DIAG, self.CMO_BIAS]
        ########################
        # Intermediate Level 0 #
        ########################
        self.n_i0_conv3 = 6 # number of filters
        self.i0_conv3_width = 2 # width of stride
        self.I0_CONV3 = theano.shared(value = HE_UNIFORM.sample((self.n_i0_conv3, 1, \
                                                                self.i0_conv3_width, 1)), \
                                      name = "I0_CONV3")
        self.I0_CONV3_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_i0_conv3)), \
                                           name = "I0_CONV3_BIAS")
        self._params += [self.I0_CONV3, self.I0_CONV3_BIAS]

        self.n_i0_conv4 = 8 # number of filters
        self.i0_conv4_width = 3 # width of stride
        self.I0_CONV4 = theano.shared(value = HE_UNIFORM.sample((self.n_i0_conv4, 1, \
                                                                self.i0_conv4_width, 1)), \
                                      name = "I0_CONV4")
        self.I0_CONV4_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_i0_conv4)), \
                                           name = "I0_CONV4_BIAS")
        self._params += [self.I0_CONV4, self.I0_CONV4_BIAS]

        self.n_i0_conv5 = 10 # number of filters
        self.i0_conv5_width = 4 # width of stride
        self.I0_CONV5 = theano.shared(value = HE_UNIFORM.sample((self.n_i0_conv5, 1, \
                                                                self.i0_conv5_width, 1)), \
                                      name = "I0_CONV5")
        self.I0_CONV5_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_i0_conv5)), \
                                           name = "I0_CONV5_BIAS")
        self._params += [self.I0_CONV5, self.I0_CONV5_BIAS]

        self.n_i0_conv = self.n_i0_conv3 + self.n_i0_conv4 + self.n_i0_conv5
        self.I0_BIAS = theano.shared(value = HE_UNIFORM.sample((1, self.n_i0_conv)), \
                                     name = "I0_BIAS")
        self._params += [self.I0_BIAS]

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
        emb = HE_UNIFORM.sample((self.V, self.vdim))
        # set EMPTY and UNK units to 0
        emb[self.feat2idx[EMP],:] = emb[self.feat2idx[UNK],:] = 0
        self.EMB = theano.shared(value = emb, name = "EMB")
        # add embeddings to the parameters to be trained
        self._params.append(self.EMB)

    def _emb2conv(self, a_x):
        """Compute convolutions from indices

        @param a_x - indices of embeddings

        @return max convolutions computed from these indices

        """
        # length of character input
        self.IN_LEN = a_x.shape[0]
        # input to convolutional layer
        self.CONV_IN = self.EMB[a_x].reshape((1, 1, self.IN_LEN, self.vdim))
        # first highway passes embeddings to convolutions and directly to the output layer
        # hw1_carry = (1 - self.HW1_TRANS_COEFF) * \
        #             TT.nnet.relu(TT.dot(self.EMB[a_x].mean(axis = 0).reshape((self.vdim,)), \
        #                                 self.HW1_TRANS) + self.HW1_TRANS_BIAS, alpha = RELU_ALPHA)
        # width-3 convolutions
        self.CONV3_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV3), \
                               (self.n_conv3, self.IN_LEN - self.conv3_width + 1)).T
        self.CONV3_MAX_OUT = self.CONV3_OUT.max(axis = 0) + self.CONV3_BIAS
        # width-4 convolutions
        self.CONV4_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV4), \
                               (self.n_conv4, self.IN_LEN - self.conv4_width + 1)).T
        self.CONV4_MAX_OUT = self.CONV4_OUT.max(axis = 0) + self.CONV4_BIAS
        # width-5 convolutions
        self.CONV5_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV5), \
                               (self.n_conv5, self.IN_LEN - self.conv5_width + 1)).T
        self.CONV5_MAX_OUT = self.CONV5_OUT.max(axis = 0) + self.CONV5_BIAS
        # output convolutions
        self.CONV_MAX_OUT = TT.concatenate([self.CONV3_MAX_OUT, \
                                       self.CONV4_MAX_OUT, self.CONV5_MAX_OUT], axis = 1)
        self.HW2_TRANS = TT.nnet.sigmoid(TT.dot(self.CONV_MAX_OUT[0,:], self.HW2_TRANS) + \
                                    self.HW2_TRANS_BIAS)
        self.HW2_CARRY = self.CONV_MAX_OUT[0,:] * (1. - self.HW2_TRANS)
        self.CMO = TT.tanh(TT.dot(self.CONV_MAX_OUT[0,:], self.CMO_W) + self.CMO_BIAS) * \
                   self.HW2_TRANS + self.HW2_CARRY
        # lstm_in = TT.dot(conv_max_out[0,:], self.LSTM_W) + self.LSTM_BIAS
        # return cmo, hw2_carry

    def _conv2conv(self):
        """Compute meaning convolutions from character convolutions

        Args:
        -----

        Returns:
        --------
        (void)

        """
        cmo = self.CMO.reshape((1, 1, self.n_cmo, 1))
        # width-3 convolutions
        self.I0_CONV3_OUT = TT.reshape(TT.nnet.conv.conv2d(cmo, self.I0_CONV3), \
                                       (self.n_i0_conv3, self.n_cmo - self.i0_conv3_width + 1)).T
        self.I0_CONV3_MAX_OUT = self.I0_CONV3_OUT.max(axis = 0) + self.I0_CONV3_BIAS
        # width-4 convolutions
        self.I0_CONV4_OUT = TT.reshape(TT.nnet.conv.conv2d(cmo, self.I0_CONV4), \
                                       (self.n_i0_conv4, self.n_cmo - self.i0_conv4_width + 1)).T
        self.I0_CONV4_MAX_OUT = self.I0_CONV4_OUT.max(axis = 0) + self.I0_CONV4_BIAS
        # width-5 convolutions
        self.I0_CONV5_OUT = TT.reshape(TT.nnet.conv.conv2d(cmo, self.I0_CONV5), \
                                       (self.n_i0_conv5, self.n_cmo - self.i0_conv5_width + 1)).T
        self.I0_CONV5_MAX_OUT = self.I0_CONV5_OUT.max(axis = 0) + self.I0_CONV5_BIAS
        # uniting convolutions
        i0 = TT.concatenate([self.I0_CONV3_MAX_OUT, self.I0_CONV4_MAX_OUT, self.I0_CONV5_MAX_OUT], \
                            axis = 1)
        self.I0 = TT.tanh(i0[0,:] + self.I0_BIAS)

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
