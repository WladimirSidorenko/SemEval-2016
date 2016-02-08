#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module providing wrapper class around an RNN classifier.

Constants:
----------

Classes:
--------
RNNModel - wrapper class around an RNN classifier

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from charstat import VAR_CHAR_IDX, COR_CHAR_IDX, \
    VAR_MSG_IDX, COR_MSG_IDX, VAR_N_IDX, COR_NMSG_IDX, \
    compute_cnt, compute_stat, compute_cov_cor
from preprocessing import NONMATCH_RE
from twokenize import tokenize

from cPickle import dump
from collections import defaultdict, OrderedDict, Counter
from copy import deepcopy
from datetime import datetime
from evaluate import PRDCT_IDX, TOTAL_IDX
from itertools import izip_longest

from lasagne.init import HeNormal, HeUniform, Orthogonal
from theano import config, printing, tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
import sys
import theano

##################################################################
# Variables and Constants
INF = float("inf")
SEED = None
RELU_ALPHA = 0.
HE_NORMAL = HeNormal()
HE_UNIFORM = HeUniform()
HE_UNIFORM_RELU = HeUniform(gain=np.sqrt(2))
HE_UNIFORM_LEAKY_RELU = HeUniform(
    gain=np.sqrt(2. / (1 + (RELU_ALPHA or 1e-6)**2)))
ORTHOGONAL = Orthogonal()

# probability of replacing a sentiment term in corpus
BINOMI_RSMPL = 0.075
# probability of adding a new positive item from the lexicon as training
# instance
BINOMI_SMPL_POS = 0.05
# probability of assigning class `2' to the new positive item
BINOMI_POS_XTRM = 0.10383
# probability of adding a new negative item from the lexicon as training
# instance
BINOMI_SMPL_NEG = 0.05
# probability of assigning class `2' to the new positive item
BINOMI_NEG_XTRM = 0.112337
# probability of adding an exclamation mark to extreme instance
BINOMI_EXCL_MARK = 0.3

CORPUS_PROPORTION_MAX = 0.33
CORPUS_PROPORTION_MIN = 0.2
XCHANGERS_MIN = 3
RESAMPLE_AFTER = 35
INCR_SAMPLE_AFTER = 150
MAX_PRE_ITERS = RESAMPLE_AFTER * 5
MAX_ITERS = RESAMPLE_AFTER * 50
DS_PRCNT = 0.15

# default training parameters
ALPHA = 5e-3
EPSILON = 1e-5
ADADELTA = 0
SGD = 1
RMS = 2
SVM_C = 3e-4
L1 = 1e-4
L2 = 1e-5
L3 = 1e-4
MIN_T_LEN = 5

# default dimension of input vectors
VEC_DIM = 32

# very small number for insignificant vectors
LEAKY_ZERO = 1e-4

# initial parameters for uniform distribution
UMIN = -5.
UMAX = 5.

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

# custom function for generating random vectors
# np.random.seed(SEED)
np.random.seed()
RND_VEC = lambda a_dim=VEC_DIM: \
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
    conv = TT.matrix(name="conv")
    amax = conv.argmax(axis=0)
    get_amax = theano.function([conv], [amax], name="get_amax")
    amax_out = get_amax(a_conv)
    for i, j in enumerate(amax_out[0]):
        print(
            "conv{:d}[{:d}]: {:s} ({:.5f})".format(
                a_conv_width, i, ''.join(a_seq[j:j + a_conv_width]),
                a_conv_max[0][i]))


def _floatX(data, a_dtype=config.floatX):
    """Return numpy array populated with the given data.

    @param data - input tensor
    @param dtype - digit type

    @return numpy array populated with the given data

    """
    return np.asarray(data, dtype=a_dtype)


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
        "Unequal number of elements in dseq '{:s}' and seq '{:s}'".format(
            repr(a_dseq), repr(a_seq))
    iword = ""
    seq = a_seq
    istart = iend = delta = 0
    # binomial and random throws and word index from set
    bthrow = uthrow = w_i = 0
    for imatch in a_re.finditer(a_seq):
        # replace word if Bernoulli tells us to do so
        if np.random.binomial(1, BINOMI_RSMPL, 1):
            # choose new word from a set using Uniform distribution
            w_i = np.random.randint(0, len(a_words), 1)
            iword = a_words[w_i]
            # estimate the indices for replacement
            istart, iend = imatch.start(1), imatch.end(1)
            # make sure we fit the width of minimal filter
            if len(seq) - (iend - istart) + len(iword) < MIN_T_LEN:
                continue
            istart += delta
            iend += delta
            # do the actual replacement
            seq = seq[:istart] + iword + seq[iend:]
            a_dseq[istart:iend] = a_rnn._feat2idcs(iword, a_fill=False)
            # update the offsets
            delta += len(iword) - (iend - istart)
    if seq != a_seq:
        # print("*** a_seq =", repr(a_seq), file=sys.stderr)
        # print("*** seq =", repr(seq), file=sys.stderr)
        ldseq = len(a_dseq)
        lseq = len(seq)
        assert ldseq == lseq or (ldseq > lseq and a_dseq[lseq] == 0), \
            ("Got unequal number of elements in dseq '{:s}'"
             " and seq '{:s}'".format(repr(a_dseq), repr(seq)))
    return a_dseq, seq


def _balance_ts(a_ts, a_min, a_class2idx, a_icnt,
                a_ts_orig=None, a_generate_ds=False,
                a_binom=False, a_rnn=None,
                a_pos_re=None, a_pos=None,
                a_neg_re=None, a_neg=None):
    """Obtained samples from the training set and balance classes.

    Args:
    -----
       a_ts: list((np.array, int))
         original (unbalanced) training set with digitized features
      a_min: int
        minimum number of instances pertaining to one class
      a_class2idx: set of indices
         mapping from classes to the indices of their instances
      a_icnt: int
        iteration counter (used to increase sample size, currently not used)
      a_ts_orig: set((str, int))
        original trainset (not converted to features, used for binomi
        subsampling only)
      a_generate_ds: bool
        return a re-sampled development set as well
      a_binom: bool
        use Bernoulli subsampling of training instances
      a_rnn: RNNModel
        instance of RNNModel (is used to digitize newly introduced samples)
      a_pos_re: re
        regexp matching positive terms
      a_pos: set(str)
        set of positive terms
      a_pos_re: re
        regexp matching negative terms
      a_neg: set(str)
        set of negative terms

    Returns:
    --------
      2-tuple:
        resampled balanced training set and dev set

    """
    # check correctness of arguments
    assert not a_binom or (a_ts_orig and a_pos_re and a_pos
                           and a_neg_re and a_neg), \
        ("Original training set, positive and negative re's are"
         " required for Binomial subsampling.")
    ts_samples = []             # generated training set
    ds_samples = []             # generated development set
    ibinom = dseq = iseq = None
    # 0) clever sub-sampling (would probably work with SGD but does not work
    # with AdaDelta):
    # icnt = icnt // INCR_SAMPLE_AFTER + 1
    # samples = [i for v in a_class2idx.itervalues() for i in
    #            np.random.choice(v, min(float((icnt + 1) * 0.5) *
    #                                    CORPUS_PROPORTION_MAX * a_min,
    #                                    (float(icnt) / (icnt + 1.)) * len(v)),
    #                             replace=False)]

    # 1) a less clever sub-sampling but it works in practice
    a_icnt = 1
    if a_generate_ds:
        ts, ds = _get_ts_ds_samples(a_class2idx, a_icnt, a_min, DS_PRCNT)
        for i in ds:
            ds_samples.append((np.asarray(a_ts[i][0], dtype="int32"),
                               a_ts[i][-1]))
    else:
        ts = _get_ts_samples(a_class2idx, a_icnt, a_min)

    # 2) select the whole corpus
    # samples = [i for v in a_class2idx.itervalues() for i in
    #            np.random.choice(v, len(v), replace=False)]

    y = ""
    x_i = 0
    x = None
    for i in ts:
        if a_binom:
            # randomly add a new positive term from the lexicon as a training
            # instance
            _rndm_add(ts_samples, a_pos, a_rnn, BINOMI_SMPL_POS,
                      BINOMI_POS_XTRM, "positive")
            # randomly add a new negative term from the lexicon as a training
            # instance
            _rndm_add(ts_samples, a_neg, a_rnn, BINOMI_SMPL_NEG,
                      BINOMI_NEG_XTRM, "negative")
            # randomly replace a sentiment term with a phrase from lexicon
            dseq = deepcopy(a_ts[i][0])
            iseq = deepcopy(a_ts_orig[i][0])
            # sample word from the opposite set and change tag
            dseq, iseq = _resample_words(a_rnn, dseq, iseq, a_pos_re,
                                         a_neg)
            # sample word from the opposite set and change tag
            dseq, _ = _resample_words(a_rnn, dseq, iseq, a_neg_re, a_pos)
            # yield modified instance if it was modified
            if dseq != a_ts[i][0]:
                # swap the tag as we are changing the classes
                ts_samples.append((np.asarray(dseq, dtype="int32"),
                                   0 if a_ts[i][-1] == 1 else 1))
                continue
        ts_samples.append((np.asarray(a_ts[i][0], dtype="int32"),
                           a_ts[i][-1]))
    return (ts_samples, ds_samples)


def _rndm_add(a_ts, a_lex, a_rnn, a_p_add, a_p_xtrm, a_tag, a_tag_xtrm=None):
    """Randomly add a subjective term from the lexicon.

    Args:
    -----
    a_ts: list(tuple(np.array, int))
      target training set to which new instances should be added
    a_lex: list(str)
      source lexicon to sample new items from
    a_rnn: RNNModel
      instance of RNNModel (is used to digitize newly introduced samples)
    a_p_add: float
      probability of sampling a new item
    a_p_xtrm: float
      probability that the new item will have an extreme degreee provided that
      it gets sampled
    a_tag: str
      tag of new sampled item
    a_tag_xtrm: str or None
      extreme tag of new sampled item

    Returns:
    --------
    (void)

    """
    xtrm_tag = False
    if np.random.binomial(1, a_p_add, 1):
        x_i = np.random.randint(0, len(a_lex), 1)
        if a_tag_xtrm is not None and np.random.binomial(1, a_p_xtrm, 1):
            xtrm_tag = True
            y = a_tag_xtrm
        else:
            y = a_tag
        # obtain new training instance
        x = a_lex[x_i]
        if xtrm_tag:
            while np.random.binomial(1, BINOMI_EXCL_MARK, 1):
                x += '!'
        # print("*** sampled {:s} with tag {:s}".format(repr(x), y))
        x, y = a_rnn._digitize_feats([(x, y)])[0]
        a_ts.append((np.asarray(x, dtype="int32"),
                     np.asarray(y, dtype="int32")))


def _get_ts_samples(a_class2idx, a_icnt, a_min,
                    a_set_apart=0.):
    """Obtained blanced training set samples.

    Args:
    -----
    a_class2idx: set of indices
      mapping from classes to particular indices
    a_icnt: int
      iteration counter (used to increase sample sizes)
    a_min: int
      minimum number of instances pertaining to one class
    a_set_apart: float
      percentage of instances to exclude from the resulting samples

    Returns:
    --------
    list:
      indices of training set instanced balanced by classes

    """
    samples = []
    n = 0
    nsamples = None
    for v in a_class2idx.itervalues():
        nsamples = np.random.choice(v, CORPUS_PROPORTION_MIN * a_min,
                                    replace=False)
        n = len(nsamples)
        samples.extend(nsamples[n*a_set_apart:])
    return samples


def _get_ts_ds_samples(a_class2idx, a_icnt, a_min,
                       a_set_apart=DS_PRCNT):
    """Obtained blanced training and dev set samples.

    Args:
    -----
    a_class2idx: set of indices
      mapping from classes to particular indices
    a_icnt: int
      iteration counter (used to increase sample sizes)
    a_min: int
      minimum number of instances pertaining to one class
    a_set_apart: float
      percentage of instances to exclude from the resulting samples

    Returns:
    --------
    list:
      indices of training set instanced balanced by classes

    """
    # generate training samples in the usual way, but take the rest of the
    # corpus as development set
    ts_samples = _get_ts_samples(a_class2idx, a_icnt, a_min, a_set_apart)
    used_samples = set(ts_samples)
    ds_samples = [i for v in a_class2idx.itervalues()
                  for i in v if i not in used_samples]
    return (ts_samples, ds_samples)


def rmsprop(tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * _floatX(0.))
                    for p in tparams]
    running_grads = [theano.shared(p.get_value() * _floatX(0.))
                     for p in tparams]
    running_grads2 = [theano.shared(p.get_value() * _floatX(0.))
                      for p in tparams]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * _floatX(0.))
             for p in tparams]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / TT.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams, updir_new)]
    f_update = theano.function([], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')
    return (f_grad_shared, f_update, (zipped_grads, running_grads,
                                      running_grads2, updir))


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

    def __init__(self, a_vdim=VEC_DIM):
        """Class constructor.

        Args:
        -----
        a_vdim: int
        Default dimensionality of embedding vectors.

        """
        self.V = 0              # vocabulary size
        self.n_labels = 0
        self.vdim = a_vdim
        self.n_cmo = 0  # size of output convolutions
        # mapping from symbolic representations to indices
        self.lbl2int = {"positive": 1, "negative": 0}
        self.int2lbl = {v: k for k, v in self.lbl2int.iteritems()}
        self.int2coeff = dict()
        self.feat2idx = dict()
        # NN parameters to be learned
        self._params = []
        # subset of parameters pertaining to convolutional layers
        self._convs = []
        # private prediction function
        self._predict = None
        # the parameters below will be initialized during training
        # matrix of items' embeddings (either words or characters) that
        # serve as input to RNN
        self.EMB = self.CONV_IN = None
        # convolutional layers, their outputs, and max-over-time outputs
        ## convolutions of width 3
        self.CONV3_P = self.CONV3_P_BIAS = self.CONV3_N = self.CONV3_N_BIAS = \
            self.CONV3_X = self.CONV3_X_BIAS = None
        self.CONV3_P_OUT = self.CONV3_P_MAX_OUT = None
        self.CONV3_N_OUT = self.CONV3_N_MAX_OUT = None
        self.CONV3_X_OUT = self.CONV3_X_MAX_OUT = None
        ## convolutions of width 4
        self.CONV4_P = self.CONV4_P_BIAS = self.CONV4_N = self.CONV4_N_BIAS = \
            self.CONV4_X = self.CONV4_X_BIAS = None
        self.CONV4_P_OUT = self.CONV4_P_MAX_OUT = None
        self.CONV4_N_OUT = self.CONV4_N_MAX_OUT = None
        self.CONV4_X_OUT = self.CONV4_X_MAX_OUT = None
        ## convolutions of width 5
        self.CONV5_P = self.CONV5_P_BIAS = self.CONV5_N = self.CONV5_N_BIAS = \
            self.CONV5_X = self.CONV5_X_BIAS = None
        self.CONV5_P_OUT = self.CONV5_P_MAX_OUT = None
        self.CONV5_N_OUT = self.CONV5_N_MAX_OUT = None
        self.CONV5_X_OUT = self.CONV5_X_MAX_OUT = None
        ## final result of convolutions
        self.CONV_P_OUT = self.CONV_N_OUT = self.CONV_X_OUT = \
            self.CONV_MAX_OUT = None
        # mapping from `CONV_MAX_OUT` to the first hidden layer and its bias
        # term
        self.CMO2I0 = self.I0_BIAS = None
        # mapping from the first hidden layer to the second hidden layer and
        # its bias term (could be replaced by dropout)
        self.I02I1 = self.I1_BIAS = None
        # mapping from the second hidden layer to the output
        self.I12Y = self.Y_BIAS = None
        # some parameters will be initialized immediately
        self._init_params()

    def fit(self, a_trainset, a_path, a_devset=None,
            a_pos_re=NONMATCH_RE, a_pos=set(),
            a_neg_re=NONMATCH_RE, a_neg=set()):
        """Train RNN model on the training set.

        Args:
          a_trainset: set
            trainig set as a list of 2-tuples with training instances and
            classes
          a_path: str
            path for storing the best model
          a_devset: set
            development set as a list of 2-tuples with training instances
            and classes
          a_pos_re: re
            regexp matching positive terms
          a_pos: set
            set of positive terms
          a_neg_re: re
            regexp matching negative terms
          a_neg: set
            set of negative terms

        Returns:
          (void)

        """
        if len(a_trainset) == 0:
            return
        # estimate the number of distinct features and classes in the corpus
        labels = set()
        feats = set()
        for w, lbl in a_trainset:
            labels.add(lbl)
            feats.update(w)
        # add features from the provided lexica
        feats |= set([c for w in (a_pos | a_neg) for c in w])
        pos = list(a_pos)
        neg = list(a_neg)

        self.n_labels = len(labels)
        self.V = len(AUX_VEC_KEYS) + len(feats)
        del feats
        # obtain indices for special embeddings (BEGINNING, END, UNKNOWN)
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
        trainset = self._digitize_feats(a_trainset, a_add=True)

        # convert raw character lists back to strings in the original training
        # set
        a_trainset = [(''.join(inst), lbl) for inst, lbl in a_trainset]

        # store indices of class instances present in the training set to ease
        # the sampling
        class2indices = defaultdict(list)
        for i, (_, y_i) in enumerate(trainset):
            class2indices[y_i].append(i)

        # get the minimum number of instances per class
        min_items = min(len(v) for v in class2indices.itervalues())

        # initialize custom function for sampling the corpus
        _balance = self._get_balance(trainset, min_items,
                                     class2indices,
                                     a_trainset, a_devset is None,
                                     True, a_pos_re, pos, a_neg_re, neg
                                     )
        # initialize final output layer (we need to know `n_labels` before we
        # do so)
        self.I12Y = theano.shared(value=HE_UNIFORM.sample((self.n_cmo,
                                                           self.n_labels)),
                                  name="I12Y")
        self.Y_BIAS = theano.shared(value=HE_UNIFORM.sample(
            (1, self.n_labels)).flatten(), name="Y_BIAS")
        # add newly initialized weights to the parameters to be trained
        self._params += [self.I12Y, self.Y_BIAS]

        # initialize embedding matrix for features
        self._init_emb()

        # initialize convolution layer and obtain additional corpus data
        self._emb2conv(self.CHAR_INDICES, _balance, a_trainset=a_trainset,
                       a_pos_re=a_pos_re, a_pos=a_pos,
                       a_neg_re=a_neg_re, a_neg=a_neg
                       )
        #  mapping from the CMO layer to the intermediate layers
        self._conv2i1(_balance)

        #  mapping from the intermediate layer to output
        self._i12y()

        # get prediction function
        self._activate_predict()

        # create symbolic variable for the correct label
        y = TT.scalar('y', dtype="int32")

        # cost gradients and updates
        cost = y * (1 - self.Y) + (1 - y) * self.Y  # + \
               # L2 * TT.sum([TT.sum(p**2) for p in self._params])

        # Alternative cost functions (SVM):
        # cost = SVM_C * (TT.max([1 - self.Y[0, y], 0])**2 +
        #                 TT.max([1 + self.Y[0, self.y_pred], 0]) ** 2) + \
        #     L2 * TT.sum([TT.sum(p**2) for p in self._params])

        gradients = TT.grad(cost, wrt=self._params)
        # define training function and let the training begin
        f_grad_shared, f_update, _ = rmsprop(self._params, gradients,
                                             self.CHAR_INDICES, y, cost)

        # digitize features in the dev set and pre-compute $\rho$ statistics
        rhostat = defaultdict(lambda: [0, 0])
        if a_devset is not None:
            a_devset = self._digitize_feats(a_devset)
            # initialize and populate rho statistics
            for _, y in a_devset:
                rhostat[y][TOTAL_IDX] += 1

        # auxiliary variables used in iterations
        y_pred_i = 0
        time_delta = 0.
        ts = ds = dev_stat = start_time = end_time = None
        # for two-class classification, we try to maximize the average $\rho$
        max_dev_score = -INF
        # for five-class classification, we try to minimize the mean error rate
        dev_score = icost = prev_cost = min_train_cost = INF
        print("lbl2int =", repr(self.lbl2int), file=sys.stderr)
        for i in xrange(MAX_ITERS):
            icost = 0.
            # resample corpus every RESAMPLE_AFTER iteration
            if (i % RESAMPLE_AFTER) == 0:
                ts, ds = _balance(i)
                # reset the $\rho$ statistics, if no dedicated development set
                # was supplied
                if a_devset is None:
                    for k in rhostat:
                        rhostat[k][TOTAL_IDX] = 0
                    for _, y in ds:
                        rhostat[y][TOTAL_IDX] += 1
            start_time = datetime.utcnow()
            # iterate over the training instances and update weights
            np.random.shuffle(ts)
            # activate dropout during training
            self.use_dropout.set_value(1.)
            for x_i, y_i in ts:
                try:
                    icost += f_grad_shared(x_i, y_i)
                    f_update()
                except:
                    print("self.feat2idx =", repr(self.feat2idx),
                          file=sys.stderr)
                    print("x_i =", repr(x_i), file=sys.stderr)
                    print("y_i =", repr(y_i), file=sys.stderr)
                    raise
            # de-activate dropout during prediction
            self.use_dropout.set_value(0.)
            # update train costs
            if icost < min_train_cost:
                min_train_cost = icost
            # reset the $\rho$ statistics
            for k in rhostat:
                rhostat[k][PRDCT_IDX] = 0
            # compute the $\rho$ statistics and cross-validation/dev score anew
            dev_score = 0.
            for x_i, y_i in (a_devset or ds):
                y_pred_i = self._predict(x_i)[0]
                rhostat[y_i][PRDCT_IDX] += (y_pred_i == y_i)
            dev_stat = [float(v[PRDCT_IDX])/float(v[TOTAL_IDX] or 1.)
                        for v in rhostat.itervalues()]
            # (used for two-class prediction)
            dev_score = sum(dev_stat) / float(len(rhostat) or 1)
            if dev_score > max_dev_score:
                self._dump(a_path)
                max_dev_score = dev_score
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print("dev_stat =", repr(dev_stat), file=sys.stderr)
            print("Iteration #{:d}: train_cost = {:.10f}, dev_score = {:.5f}"
                  " ({:.2f} sec);".format(i, icost, dev_score, time_delta),
                  file=sys.stderr)
            if prev_cost != INF and 0. <= (prev_cost - icost) < EPSILON:
                break
            prev_cost = icost
        print("Minimum train cost = {:.10f}, "
              "Maximum dev score = {:.10f}".format(min_train_cost,
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
        self.int2lbl = {1: "positive", 0: "negative"}
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
        debug_emb = theano.function([self.CHAR_INDICES], [d_emb],
                                    name="debug_emb")
        print("*** EMBEDDINGS ***")
        debug_emb(a_dseq)

        get_conv_in = theano.function([self.CHAR_INDICES],
                                      [self.CONV_IN, self.IN_LEN],
                                      name="get_conv_in")
        conv_in, in_len = get_conv_in(a_dseq)
        # output convolutions
        print("*** CONVOLUTIONS(2) ***")
        print("conv_in =", repr(conv_in))
        print("in_len =", repr(in_len))
        print("*** CONVOLUTIONS(3) ***")
        get_conv3 = theano.function([self.CONV_IN, self.IN_LEN],
                                    [self.CONV3_P_OUT, self.CONV3_P_MAX_OUT],
                                    name="get_conv3")
        conv3_out, conv3_max_out = get_conv3(conv_in, in_len)
        _debug_conv(a_seq, self.conv3_width, conv3_out, conv3_max_out)
        print("*** CONVOLUTIONS(4) ***")
        get_conv4 = theano.function([self.CONV_IN, self.IN_LEN],
                                    [self.CONV4_P_OUT, self.CONV4_P_MAX_OUT],
                                    name="get_conv4")
        conv4_out, conv4_max_out = get_conv4(conv_in, in_len)
        _debug_conv(a_seq, self.conv4_width, conv4_out, conv4_max_out)
        print("*** CONVOLUTIONS(5) ***")
        get_conv5 = theano.function([self.CONV_IN, self.IN_LEN],
                                    [self.CONV5_P_OUT, self.CONV5_P_MAX_OUT],
                                    name="get_conv4")
        conv5_out, conv5_max_out = get_conv5(conv_in, in_len)
        _debug_conv(a_seq, self.conv5_width, conv5_out, conv5_max_out)

        # concatenated convolution layer
        get_conv_max_out = theano.function([self.CONV3_P_MAX_OUT,
                                            self.CONV4_P_MAX_OUT,
                                            self.CONV5_P_MAX_OUT],
                                           [self.CONV_MAX_OUT],
                                           name="get_conv_max_out")
        conv_max_out = get_conv_max_out(conv3_max_out,
                                        conv4_max_out,
                                        conv5_max_out)[0]
        print("*** CONV_MAX_OUT ***\n", repr(conv_max_out), file=sys.stderr)

        # output highways
        get_hw2_trans = theano.function([self.CONV_MAX_OUT],
                                        [self.HW2_TRANS], name="get_hw2_trans")
        hw2_trans = get_hw2_trans(conv_max_out)[0]
        print("*** HW2_TRANS ***\n", repr(hw2_trans), file=sys.stderr)

        get_hw2_carry = theano.function([self.CONV_MAX_OUT],
                                        [self.HW2_CARRY], name="get_hw2_carry")
        hw2_carry = get_hw2_carry(conv_max_out)[0]
        print("*** HW2_CARRY ***\n", repr(hw2_carry), file=sys.stderr)
        # output CMO and final predictions
        get_cmo = theano.function([self.CONV_MAX_OUT, self.HW2_TRANS,
                                   self.HW2_CARRY], [self.CMO], name="get_cmo")
        cmo = get_cmo(conv_max_out, hw2_trans, hw2_carry)
        print("*** CMO ***\n", repr(cmo), file=sys.stderr)

        # output CMO2Y and Y_BIAS
        print("*** CMO2I0 ***\n", repr(self.CMO2I0.get_value()),
              file=sys.stderr)
        print("*** I0_BIAS ***\n", repr(self.I0_BIAS.get_value()),
              file=sys.stderr)
        get_i0 = theano.function([self.CMO], [self.I0], name="get_i0")
        i0 = get_i0(cmo[0])
        print("*** I0 ***\n", repr(i0), file=sys.stderr)

        # output final predictions
        print("*** I02Y ***\n", repr(self.I02Y.get_value()), file=sys.stderr)
        print("*** Y_BIAS ***\n", repr(self.Y_BIAS.get_value()),
              file=sys.stderr)
        get_y = theano.function([self.I0], [self.Y], name="get_y")
        print("*** Y ***\n", repr(get_y(i0)), file=sys.stderr)

        # re-check predictions
        self._activate_predict()
        y, score = self._predict(self._feat2idcs(a_seq))
        print("y =", repr(self.int2lbl[int(y)]), file=sys.stderr)
        print("score =", repr(score), file=sys.stderr)

    def _digitize_feats(self, a_trainset, a_add=False):
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
        dlabel = -1
        dint = coeff = 1.
        # create a vector for unknown words
        for iseq, ilabel in a_trainset:
            # digitize label and convert it to a vector
            if ilabel is not None:
                if isinstance(ilabel, int):
                    dlabel = ilabel
                else:
                    if ilabel not in self.lbl2int:
                        try:
                            coeff = int(ilabel)
                        except (AssertionError, ValueError):
                            coeff = 1.
                        self.lbl2int[ilabel] = clabels
                        self.int2lbl[clabels] = ilabel
                        self.int2coeff[clabels] = coeff
                        clabels += 1
                    dint = self.lbl2int[ilabel]
                    dlabel = dint
                # dlabel = np.zeros(self.n_labels)
                # dlabel[dint] = 1 * self.int2coeff[dint]
                # for SVM
                # dlabel = np.ones(self.n_labels).astype("int32") * -1
                # dlabel[dint] = 1
            # convert features to indices and append new training
            # instance
            ret.append((self._feat2idcs(iseq, a_add=a_add), dlabel))
        return ret

    def _feat2idcs(self, a_seq, a_add=False, a_fill=True):
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
            max_len = max(ilen, self.conv3_width,
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

        Returns:
        --------
        (void)

        """
        # matrix of char vectors, corresponding to single word
        self.CHAR_INDICES = TT.ivector(name="CHAR_INDICES")

        ################
        # CONVOLUTIONS #
        ################
        self._init_convs()

        #######
        # CMO #
        #######
        self.CMO_W = theano.shared(value=
                                   HE_UNIFORM_RELU.sample(
                                       (self.n_cmo, self.n_cmo)),
                                   name="CMO_W")
        self.CMO_BIAS = theano.shared(
            HE_UNIFORM_RELU.sample((1, self.n_cmo)).flatten(),
            name="CMO_BIAS"
        )
        self._params += [self.CMO_W, self.CMO_BIAS]

        ########################
        # Intermediate Level 0 #
        ########################
        self.CMO2I0 = theano.shared(value=ORTHOGONAL.sample(
            (self.n_cmo, self.n_cmo)), name="CMO2I0")
        self.I0_BIAS = theano.shared(value=
                                     HE_UNIFORM.sample((1, self.n_cmo)),
                                     name="I0_BIAS")
        self._params += [self.CMO2I0, self.I0_BIAS]

        ########################
        # Intermediate Level 1 #
        ########################
        self.I02I1 = theano.shared(value=ORTHOGONAL.sample(
            (self.n_cmo, self.n_cmo)), name="I02I1")
        self.I1_BIAS = theano.shared(value=
                                     HE_UNIFORM.sample((1, self.n_cmo)),
                                     name="I1_BIAS")
        self._params += [self.I02I1, self.I1_BIAS]

        ###########
        # DROPOUT #
        ###########
        self.use_dropout = theano.shared(_floatX(1.))

    def _init_convs(self):
        """Initialize convolution layers.

        Returns:
        --------
        (void)

        """
        # convolutions of width 3
        self.n_conv3 = 4  # number of filters
        self.conv3_width = 3  # width of stride
        ## positive convolutions
        self.CONV3_P, self.CONV3_P_BIAS = self._init_conv_layer(
            self.n_conv3, self.conv3_width, self.vdim, "CONV3_P",
            "CONV3_P_BIAS"
        )
        ## negative convolutions
        self.CONV3_N, self.CONV3_N_BIAS = self._init_conv_layer(
            self.n_conv3, self.conv3_width, self.vdim, "CONV3_N",
            "CONV3_N_BIAS"
        )
        ## mirror convolutions
        self.CONV3_X, self.CONV3_X_BIAS = self._init_conv_layer(
            self.n_conv3, self.conv3_width, self.vdim, "CONV3_X",
            "CONV3_X_BIAS"
        )

        # convolutions of width 4
        self.n_conv4 = 8        # number of filters
        self.conv4_width = 4    # width of stride
        ## positive convolutions
        self.CONV4_P, self.CONV4_P_BIAS = self._init_conv_layer(
            self.n_conv4, self.conv4_width, self.vdim, "CONV4_P",
            "CONV4_P_BIAS"
        )
        ## negative convolutions
        self.CONV4_N, self.CONV4_N_BIAS = self._init_conv_layer(
            self.n_conv4, self.conv4_width, self.vdim, "CONV4_N",
            "CONV4_N_BIAS"
        )
        ## mirror convolutions
        self.CONV4_X, self.CONV4_X_BIAS = self._init_conv_layer(
            self.n_conv4, self.conv4_width, self.vdim, "CONV4_X",
            "CONV4_X_BIAS"
        )

        # convolutions of width 5
        self.n_conv5 = 12       # number of filters
        self.conv5_width = MIN_T_LEN    # width of stride (5)
        ## positive convolutions
        self.CONV5_P, self.CONV5_P_BIAS = self._init_conv_layer(
            self.n_conv5, self.conv5_width, self.vdim, "CONV5_P",
            "CONV5_P_BIAS"
        )
        ## negative convolutions
        self.CONV5_N, self.CONV5_N_BIAS = self._init_conv_layer(
            self.n_conv5, self.conv5_width, self.vdim, "CONV5_N",
            "CONV5_N_BIAS"
        )
        ## mirror convolutions
        self.CONV5_X, self.CONV5_X_BIAS = self._init_conv_layer(
            self.n_conv5, self.conv5_width, self.vdim, "CONV5_X",
            "CONV5_X_BIAS"
        )
        # size of output layer
        self.n_cmo = self.n_conv3 + self.n_conv4 + self.n_conv5

    def _init_conv_layer(self, a_n_filters, a_width, a_in_dim,
                         a_layer_name, a_bias_name):
        """Initialize single convolution layer.

        Args:
        -----
        a_n_filters: int
          number of filters
        a_width: int
          width of a single filter
        a_in_dim: int
          dimensionality of input vectors
        a_layer_name: str
          name for the new convolutional tensor
        a_bias_name: str
          name for the bias vector associated with the given layer

        Returns:
        --------
        tuple(theano.shared, theano.shared):
          2-tuple with convolutional tensor (collection of filters) and bias
          term

        """
        layer = theano.shared(value=HE_UNIFORM.sample((a_n_filters, 1,
                                                       a_width,
                                                       a_in_dim)),
                              name=a_layer_name)
        bias = theano.shared(value=HE_UNIFORM.sample((1, a_n_filters)),
                             name=a_bias_name)
        self._params += [layer, bias]
        self._convs += [layer, bias]
        return (layer, bias)

    def _init_emb(self, a_ts=None):
        """Initialize embeddings.

        Args:
        a_ts: list(2-tuple) or None
          training set

        Returns:
          (void)

        """
        # create embeddings (normal initialization)
        if a_ts is None:
            emb = HE_UNIFORM.sample((self.V, self.vdim))
            # set EMPTY and UNK units to 0
            emb[self.feat2idx[EMP], :] = \
                emb[self.feat2idx[UNK], :] = LEAKY_ZERO
            # set whitespace embedding to 0
            if ' ' in self.feat2idx:
                emb[self.feat2idx[' '], :] = LEAKY_ZERO
        # controlled initialization
        else:
            # initialize embeddings to very small vectors
            emb = _floatX(np.ones((self.V, self.vdim)) * LEAKY_ZERO)
            # compute length of vector regions that are responsible for one
            # particular class
            n_clsdim = self.vdim / self.n_labels
            # compute characters' covariance and correlation
            cls_cnt, char_cnt = compute_cnt((lbl, txt) for txt, lbl in a_ts)
            cls_stat, char_stat, char_cls_stat = \
                compute_stat(cls_cnt, char_cnt)
            cov_cor = compute_cov_cor(cls_stat, char_stat, char_cls_stat)
            # initialize characters' vectors to their correlation values
            istart = iend = 0
            char_cls_mu = char_sigma = char_corr = char_var = 0.
            for ichar, iclstat in cov_cor.iteritems():
                # obtain variance of that char sequence
                char_sigma = np.sqrt(char_stat[ichar][VAR_N_IDX])
                for i, icls in enumerate(cls_stat):
                    # obtain start and end of the vector region specific to
                    # that class
                    istart = i * n_clsdim
                    iend = istart + n_clsdim
                    # obtain correlation coefficient of that char sequence
                    char_cls_mu = iclstat[icls][COR_NMSG_IDX]
                    # initialize appropriate part of the embedding vector from
                    # normal distribution whose mean is the correlation
                    # coefficient and variance is that of the char sequence
                    ichar_cls_stat = iclstat[icls]
                    emb[self.feat2idx[ichar], istart:iend] = \
                        np.random.normal(char_cls_mu, char_sigma)
        self.EMB = theano.shared(value=emb, name="EMB")
        # add embeddings to the parameters to be trained
        self._params.append(self.EMB)

    def _emb2conv(self, a_x, a_balance, **a_kwargs):
        """Compute convolutions from indices

        Args:
        -----
        a_x: theano.variable
          indices of embeddings
        a_balance: lambda
          function for resampling the corpus
        a_kwargs: dict
          additional keyword arguments to be passed to pre-training of
          convolutions

        Returns:
        --------
        (list(tuple(str, int)), list(tuple(nd.array, int))):
          digitized and original training sets generated from lexicons

        """
        # length of character input
        self.IN_LEN = a_x.shape[0]
        # input to convolutional layer
        self.CONV_IN = self.EMB[a_x].reshape((1, 1, self.IN_LEN, self.vdim))
        # width-3 convolutions
        self.CONV3_P_OUT, self.CONV3_P_MAX_OUT = \
            self._get_conv_max_out(self.CONV3_P, self.CONV3_P_BIAS,
                                   self.n_conv3, self.conv3_width)
        self.CONV3_N_OUT, self.CONV3_N_MAX_OUT = \
            self._get_conv_max_out(self.CONV3_N, self.CONV3_N_BIAS,
                                   self.n_conv3, self.conv3_width)
        self.CONV3_X_OUT, self.CONV3_X_MAX_OUT = \
            self._get_conv_max_out(self.CONV3_X, self.CONV3_X_BIAS,
                                   self.n_conv3, self.conv3_width)
        # width-4 convolutions
        self.CONV4_P_OUT, self.CONV4_P_MAX_OUT = \
            self._get_conv_max_out(self.CONV4_P, self.CONV4_P_BIAS,
                                   self.n_conv4, self.conv4_width)
        self.CONV4_N_OUT, self.CONV4_N_MAX_OUT = \
            self._get_conv_max_out(self.CONV4_N, self.CONV4_N_BIAS,
                                   self.n_conv4, self.conv4_width)
        self.CONV4_X_OUT, self.CONV4_X_MAX_OUT = \
            self._get_conv_max_out(self.CONV4_X, self.CONV4_X_BIAS,
                                   self.n_conv4, self.conv4_width)
        # width-5 convolutions
        self.CONV5_P_OUT, self.CONV5_P_MAX_OUT = \
            self._get_conv_max_out(self.CONV5_P, self.CONV5_P_BIAS,
                                   self.n_conv5, self.conv5_width)
        self.CONV5_N_OUT, self.CONV5_N_MAX_OUT = \
            self._get_conv_max_out(self.CONV5_N, self.CONV5_N_BIAS,
                                   self.n_conv5, self.conv5_width)
        self.CONV5_X_OUT, self.CONV5_X_MAX_OUT = \
            self._get_conv_max_out(self.CONV5_X, self.CONV5_X_BIAS,
                                   self.n_conv5, self.conv5_width)
        # resulting polarity convolutions
        self.CONV_P_OUT = TT.concatenate([self.CONV3_P_MAX_OUT,
                                          self.CONV4_P_MAX_OUT,
                                          self.CONV5_P_MAX_OUT], axis=1)[0, :]
        self.CONV_N_OUT = TT.concatenate([self.CONV3_N_MAX_OUT,
                                          self.CONV4_N_MAX_OUT,
                                          self.CONV5_N_MAX_OUT], axis=1)[0, :]
        self.CONV_X_OUT = TT.concatenate([self.CONV3_X_MAX_OUT,
                                          self.CONV4_X_MAX_OUT,
                                          self.CONV5_X_MAX_OUT], axis=1)[0, :]

        self.CONV_MAX_OUT = TT.nnet.sigmoid(self.CONV_P_OUT -
                                            self.CONV_N_OUT) * \
            -TT.tanh(self.CONV_X_OUT)
        _params = [self.EMB, self.I12Y, self.Y_BIAS] + self._convs
        self._pretrain(self.CONV_MAX_OUT, a_balance, _params, "CONV_MAX_OUT")
        # initialize and pre-train embeddings/convolutions
        self.CMO = TT.nnet.relu(TT.dot(self.CONV_MAX_OUT, self.CMO_W) +
                                self.CMO_BIAS, alpha=RELU_ALPHA)
        _params += [self.CMO_W, self.CMO_BIAS]
        self._pretrain(self.CMO, a_balance, _params, "CMO")
        # # pre-train embeddings/convolutions
        # return self._pretrain_emb_convs(**a_kwargs)

    def _get_conv_max_out(self, a_conv_layer, a_conv_bias, a_n_filters,
                          a_width):
        """Obtain single conv max out layer.

        Args:
        -----
        a_conv_layer: theano.variable
          layer of convolution filters
        a_conv_bias: theano.variable
          bias term for the output convolutions
        a_n_filters: int
          number of convolution filters
        a_width: int
          filter width

        Returns:
        --------
        tuple(theano.shared, theano.shared):
          conv-out and conv-max-out layers

        """
        conv_out = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN,
                                                  a_conv_layer),
                              (a_n_filters, self.IN_LEN - a_width + 1)).T
        conv_max_out = conv_out.max(axis=0) + a_conv_bias
        return (conv_out, conv_max_out)

    def _conv2i1(self, a_balance):
        """Compute intermediate layers from character convolutions

        Args:
        -----
        a_balance: lambda
          function for resampling the corpus

        Returns:
        --------
        (void)

        """
        self.I0 = TT.tanh(TT.dot(self.CMO, self.CMO2I0) + self.I0_BIAS)
        _params = [self.CMO_W, self.CMO_BIAS, self.CMO2I0, self.I0_BIAS,
                   self.I12Y, self.Y_BIAS, self.EMB] + self._convs
        self._pretrain(self.I0, a_balance, _params, "I0")

        # initialize dropout layer
        # self.I1 = TT.nnet.sigmoid(TT.dot(self.I0[0, :], self.I02I1) +
        #                           self.I1_BIAS)
        # _params += [self.I02I1, self.I1_BIAS]
        # self._pretrain(self.I1, a_balance, _params, "I1")
        I1 = TT.nnet.sigmoid(TT.dot(self.I0[0, :], self.I02I1) +
                             self.I1_BIAS)
        _params += [self.I02I1, self.I1_BIAS]
        self.I1 = self._init_dropout(I1)
        self._pretrain(self.I1, a_balance, _params, "I1")

    def _i12y(self):
        """Compute output from intermediate layer

        Args:
        -----
        (void)

        Returns:
        --------
        (void)

        """
        self.Y = TT.nnet.sigmoid(TT.sum(TT.dot(self.I1, self.I12Y) +
                                        self.Y_BIAS))

    def _init_dropout(self, a_input):
        """Create a dropout layer.

        Args:
          a_input (theano.vector): input layer

        Returns:
          theano.vector: dropout layer

        """
        # generator of random numbers
        trng = RandomStreams()
        # the dropout layer itself
        output = TT.switch(self.use_dropout,
                           (a_input * trng.binomial(a_input.shape,
                                                    p=0.5, n=1,
                                                    dtype=a_input.dtype)),
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
            self._predict = theano.function([self.CHAR_INDICES],
                                            [self.Y >= 0.5,
                                             self.Y],
                                            name="predict")

    def _get_balance(self, a_ts, a_min, a_class2idcs,
                     a_ts_orig, a_generate_ds, a_binom,
                     a_pos_re, a_pos, a_neg_re, a_neg):
        """Create custom function for corpus sub-sampling.

        Args:
        -----
        a_ts: list((np.array, int))
          original (unbalanced) training set with digitized features
        a_min: int
          minimum number of instances pertaining to one class
        a_class2idcs: set of indices
          mapping from classes to the indices of their instances
        a_ts_orig: set((str, int))
          original trainset (not converted to features, used for binomi
          subsampling only)
        a_generate_ds: bool
          return a re-sampled development set as well
        a_binom: bool
          use Bernoulli subsampling of training instances
        a_pos_re: re
          regexp matching positive terms
        a_pos: set(str)
          set of positive terms
        a_pos_re: re
          regexp matching negative terms
        a_neg: set(str)
          set of negative terms

        Returns:
        --------
        lambda:
          custom function for corpus sub-sampling

        """
        # define private function
        def _balance(i=1):
            ts, ds = _balance_ts(a_ts, a_min, a_class2idcs, i,
                                 a_ts_orig, a_generate_ds, a_binom,
                                 self, a_pos_re, a_pos, a_neg_re, a_neg)
            return (ts, ds)
        # return defined function
        return _balance

    def _pretrain_emb_convs(self, a_trainset, a_pos_re, a_pos,
                            a_neg_re, a_neg):
        """Pre-train embeddings and convolutions using a shortened NN.

        Args:
        -----
        a_trainset: set
          trainig set as a list of 2-tuples with training instances as strings
          and their classes
        a_pos_re: re
          regexp matching positive terms
        a_pos: set(str)
          set of positive terms
        a_pos_re: re
          regexp matching negative terms
        a_neg: set(str)
          set of negative terms

        Returns:
        --------
        (list(tuple(str, int)), list(tuple(nd.array, int))):
          digitized and original training sets generated from lexicons

        """
        # obtain neutral and exchanger terms
        words = set()
        neutrals = set()
        xchangers = Counter()
        polar_terms = a_pos | a_neg
        for t, y in a_trainset:
            words.update(tokenize(t))
            if y == "positive":
                if words & a_neg:
                    xchangers.update(words - polar_terms)
                elif words & a_pos:
                    neutrals |= words - polar_terms
            else:
                if words & a_pos:
                    xchangers.update(words - polar_terms)
                elif words & a_neg:
                    neutrals |= words - polar_terms
            words.clear()
        # generate xchangers from frequent terms that appear in mixed polarity
        # items
        xchangers = set(k for k, v in xchangers.iteritems() if v >
                        XCHANGERS_MIN)

        # create training sets of positive, negative, and exchanger terms
        pos_ts = self._dict2ts(a_pos, a_neg | neutrals | xchangers)
        neg_ts = self._dict2ts(a_neg, a_pos | neutrals | xchangers)
        xchng_ts = self._dict2ts(xchangers, a_pos | a_neg | neutrals,
                                 a_pos_tag=1, a_neg_tag=-1)
        # generate enriched compositionality corpus
        xchangers = list(xchangers)
        enriched_ts_orig = [(x, 1) for x in a_pos] + \
                           [(x, 0) for x in a_neg] + \
                           [(np.random.choice(xchangers, 1)[0] + ' ' + x, 0)
                            for x in a_pos]
        enriched_ts = self._digitize_feats(enriched_ts_orig, a_add=True)

        # obtain base parameters which are shared by all functions
        base_params = [self.EMB]
        # obtain parameters specific to each particular classifier
        pos_params = base_params + [self.CONV3_P, self.CONV3_P_BIAS,
                                    self.CONV4_P, self.CONV4_P_BIAS,
                                    self.CONV5_P, self.CONV5_P_BIAS]
        neg_params = base_params + [self.CONV3_N, self.CONV3_N_BIAS,
                                    self.CONV4_N, self.CONV4_N_BIAS,
                                    self.CONV5_N, self.CONV5_N_BIAS]
        xchng_params = base_params + [self.CONV3_X, self.CONV3_X_BIAS,
                                      self.CONV4_X, self.CONV4_X_BIAS,
                                      self.CONV5_X, self.CONV5_X_BIAS]
        # create custom predicition and training functions and parameter sets
        pos_cost, pos_update, pos_shared = self._get_cost_update_shared(
            self.CONV_P_OUT, pos_params)
        neg_cost, neg_update, neg_shared = self._get_cost_update_shared(
            self.CONV_N_OUT, neg_params)
        xchng_cost, xchng_update, xchng_shared = self._get_cost_update_shared(
            self.CONV_X_OUT, xchng_params, a_dec_func=TT.tanh,
            a_cost=lambda y, pred: (y - pred)**2)
        # update convolutions and mebeddings
        time_delta = 0.
        start_time = end_time = None
        cost_pos = cost_neg = cost_xchng = 0.
        for i in xrange(RESAMPLE_AFTER):
            # reset the counters
            cost_pos = cost_neg = cost_xchng = 0.
            # reshuffle training sets
            np.random.shuffle(pos_ts)
            np.random.shuffle(neg_ts)
            np.random.shuffle(xchng_ts)
            # perform actual training
            start_time = datetime.utcnow()
            for ipos, ineg, ixchng in izip_longest(pos_ts, neg_ts, xchng_ts):
                if ipos is not None:
                    cost_pos += pos_cost(*ipos)
                    pos_update()
                if ineg is not None:
                    cost_neg += neg_cost(*ineg)
                    neg_update()
                if ixchng is not None:
                    cost_xchng += xchng_cost(*ixchng)
                    xchng_update()
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print("Pre-training iteration (CONV) #{:d}: "
                  "pos_train_cost = {:.10f}, neg_train_cost = {:.10f}, "
                  "xchng_train_cost = {:.10f} ({:.2f} sec);".format(
                      i, cost_pos, cost_neg, cost_xchng, time_delta),
                  file=sys.stderr)
        # clean up memory occupied by shared variables
        self._cleanup(pos_shared)
        self._cleanup(neg_shared)
        self._cleanup(xchng_shared)
        # free some memory
        del xchng_ts[:]
        del neg_ts[:]
        # pre-train compositional function
        cmo_params = [self.EMB, self.CMO_W, self.CMO_BIAS] + self._convs
        cmo_cost, cmo_update, cmo_shared = self._get_cost_update_shared(
            self.CMO, cmo_params)
        cost = 0.
        for i in xrange(RESAMPLE_AFTER):
            cost = 0.
            for x, y in enriched_ts:
                cost += cmo_cost(x, y)
                cmo_update()
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print("Pre-training iteration (CMO) #{:d}: "
                  "cost = {:.10f} ({:.2f} sec);".format(i, cost, time_delta),
                  file=sys.stderr)
        self._cleanup(cmo_shared)
        # return generated training sets
        return enriched_ts, enriched_ts_orig

    def _pretrain(self, a_x, a_balance, a_params, a_stage="", a_idim=None):
        """Pre-train embeddings and convolutions using a shortened NN.

        Args:
        -----
        a_x: theano.variable
          last activated layer (should have size n_cmo)
        a_balance: lambda(int)
          custom function for balancing training set
        a_params: list(theano.variable)
          list of parameters to be pre-trained
        a_stage: str
          symbolic name for the pre-training stage
        a_idim: int
          input dimensionality

        Returns:
        --------
        (void)

        """
        if a_idim is None:
            a_idim = self.n_cmo
        Y = TT.nnet.softmax(TT.dot(a_x, self.I12Y) + self.Y_BIAS)
        # prediction
        y = TT.scalar('y' + a_stage, dtype="int32")

        # cost
        cost = -TT.log(Y[0, y])  # + \
            # L2 * TT.sum([TT.sum(p**2) for p in a_params])
        # updates
        gradients = TT.grad(cost, wrt=a_params)
        f_grad_shared, f_update, shared_vars = rmsprop(a_params, gradients,
                                                       self.CHAR_INDICES,
                                                       y, cost)

        time_delta = 0.
        start_time = end_time = None
        icost = prev_cost = min_train_cost = INF
        for i in xrange(MAX_PRE_ITERS):
            icost = 0.
            if (i % RESAMPLE_AFTER) == 0:
                print("Resampled", file=sys.stderr)
                ts, _ = a_balance(i)
            start_time = datetime.utcnow()
            # iterate over the training instances
            np.random.shuffle(ts)
            for x_i, y_i in ts:
                try:
                    icost += f_grad_shared(x_i, y_i)
                    f_update()
                except:
                    print("self.feat2idx =", repr(self.feat2idx),
                          file=sys.stderr)
                    print("x_i =", repr(x_i), file=sys.stderr)
                    print("y_i =", repr(y_i), file=sys.stderr)
                    raise
            if icost < min_train_cost:
                min_train_cost = icost
            end_time = datetime.utcnow()
            time_delta = (end_time - start_time).seconds
            print("Pre-training iteration ({:s}) #{:d}: train_cost = {:.10f}"
                  " ({:.2f} sec);".format(a_stage, i, icost, time_delta),
                  file=sys.stderr)
            if prev_cost != INF and 0. <= (prev_cost - icost) < EPSILON:
                break
            prev_cost = icost
        # clean up memory occupied by shared variables
        self._cleanup(shared_vars)

    def _dict2ts(self, a_pos, a_neg, a_pos_tag=1, a_neg_tag=0):
        """Generate corpus of positive and negative instances.

        Args:
        -----
        a_pos: set(str)
          set of positive instances
        a_neg: set(str)
          set of negative instances
        a_pos_tag: int
          tag to be assigned to positive instances
        a_neg_tag: int
          tag to be assigned to positive instances

        Returns:
        --------
        list(tuple(np.array, np.array))
          newly constructed training set

        """
        res = []
        x = y = None
        # add positive instances
        for w in a_pos:
            x, y = self._digitize_feats([(w, a_pos_tag)])[0]
            res.append((np.asarray(x, dtype="int32"),
                        np.asarray(y, dtype="int32")))
        # add negative instances
        for w in a_neg:
            x, y = self._digitize_feats([(w, a_neg_tag)])[0]
            res.append((np.asarray(x, dtype="int32"),
                        np.asarray(y, dtype="int32")))
        return res

    def _get_cost_update_shared(self, a_x, a_params,
                                a_dec_func=TT.nnet.sigmoid,
                                a_cost=lambda y, pred:
                                y * (1 - pred) + (1 - y) * pred):
        """Obtain cost and update functions and shared variables.

        Args:
        -----
        a_x: theano.shared
          input layer used for prediction
        a_params: list(theano.shared)
          list of input parameters affecting prediction
        a_dec_func: list(theano.shared)
          list of input parameters affecting prediction

        Returns:
        --------
        3-tuple: cost function, update function, shared variables

        """
        # gold label
        y = TT.scalar('y', dtype="int32")
        # classifier's prediction
        pred = a_dec_func(TT.sum(a_x))
        # prediction cost
        cost = a_cost(y, pred)
        gradients = TT.grad(cost, wrt=a_params)
        # generate cost and update and set of shared vars
        return rmsprop(a_params, gradients, self.CHAR_INDICES, y, cost)

    def _cleanup(self, a_shared_vars):
        """Clean-up memory occupied by shared variables.

        Args:
        -----
        a_shared_vars: list(theano.shared)
          list of shared variables whose memory should be freed

        Returns:
        --------
        (void)

        """
        dim = 0
        for var_list in a_shared_vars:
            for v in var_list:
                dim = len(v.shape.eval())
                v.set_value(np.zeros([0] * dim).astype(config.floatX))

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
