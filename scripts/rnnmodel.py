#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module providing wrapper class around an RNN classifier.

Constants:

Classes:
RNNModel - wrapper class around an RNN classifier

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

import numpy as np
import sys
import theano

from theano import tensor as TT
from collections import OrderedDict
from itertools import chain

##################################################################
# Variables and Constants
INF = float("inf")

# default training parameters
ALPHA = 1e-3
EPSILON = 1e-5
MAX_ITERS = 1e2

# initial parameters for uniform distribution
UMIN = -1.5
UMAX = 1.5

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

# dimension of input vectors
VEC_DIM = 2 # 20
# context window
CW = 2

# custom function for generating random vectors
np.random.seed()
RND_VEC = lambda a_dim = VEC_DIM: np.random.uniform(UMIN, UMAX, a_dim)

# symbolic codes for auxiliary vectors
UNK = "___%UNK%___"
BEG = "___%BEG%___"
END = "___%END%___"
AUX_VEC_KEYS = [UNK, BEG, END]

##################################################################
# Methods
def _iwindow(seq, n=2):
    """Iterate over data with a sliding window (of width n)

    @param seq - sequence to iterate over
    @param n - width of the window

    @return window-sized iterator over sequence

    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

##################################################################
# Class
class RNNModel(object):
    """Wrapper class around an RNN classifier.

    Instance variables:
    vdim - default dimensionality of embedding vectors
    V - size of fature vocabulary
    nlabels - total number of distinct target labels
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
        self.nlabels = 0
        self.vdim = a_vdim
        # mapping from symbolic representations to indices
        self.lbl2int = dict()
        self.int2lbl = dict()
        self.int2coeff = dict()
        self.feat2idx = dict()
        self.alpha = ALPHA
        # declare symbolic Theano variables
        self.EMB = None
        # word compositionality tensor (2d x 2d x d) (from embeddings to hidden)
        self.E2H = theano.shared(value = RND_VEC((self.vdim, CW * self.vdim, CW * self.vdim)))
        # recurrence matrix for the hidden layer (d x d)
        self.H2H = theano.shared(value = RND_VEC((self.vdim, self.vdim)))
        # bias vector for the hidden layer (1 x d)
        self.HBV = theano.shared(value = RND_VEC((1, self.vdim)))
        # recurrent layer
        self.H0  = theano.shared(value = np.zeros([1, self.vdim], dtype=theano.config.floatX))
        # predictor matrix (hidden --> out) (nlabels x d) (it's empty
        # until we get to know the number of labels)
        self.H2Y = None
        # bias vector for the output layer (1 x nlabels)
        self.YBV = None
        self.recurrence = None
        # private prediction function (will be initialized after training)
        self._predict = None
        # auxiliary variable for keeping track of parameters
        self._params = [self.E2H, self.H2H, self.HBV]

    def fit(self, a_trainset):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes

        @return \c void

        """
        # estimate the number of distinct features
        self.V = len(set([f for t in a_trainset for f in t[0]])) + len(AUX_VEC_KEYS)
        # initialize embedding matrix for features
        self.EMB = theano.shared(value = RND_VEC((self.V, self.vdim)))
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
        # prepend embeddings matrix to the list of parameters to be trained
        self._params[0:0] = [self.EMB]
        self.nlabels = len(set([t[1] for t in a_trainset]))
        # initialize prediction matrix
        self.H2Y = theano.shared(value = RND_VEC((self.vdim, self.nlabels)))
        self.YBV = theano.shared(value = RND_VEC((1, self.nlabels)))
        self._params.append(self.H2Y); self._params.append(self.YBV)

        # define custom recurrence function
        def _recurrence(x_t, h_tm1):
            # embedding layer
            emb_t = self.EMB[x_t].reshape([1, CW * self.vdim], ndim = 2)
            # embedding layer propagated via tensor
            in_t = TT.dot(TT.tensordot(emb_t, self.E2H, [[1], [1]]), emb_t.T)
            in_t = in_t.reshape([1, self.vdim], ndim = 2)
            # print("in_t.shape", repr(in_t.shape), file = sys.stderr)
            # hidden layer gets embeddings and its own state from the last run
            h_t = TT.nnet.sigmoid(in_t + TT.dot(h_tm1, self.H2H) + self.HBV)
            # print("h_t.shape", repr(h_t.shape), file = sys.stderr)
            s_t = TT.nnet.softmax(TT.dot(h_t, self.H2Y) + self.YBV)
            return [h_t, s_t]

        # auxiliary variables used for training
        x = TT.lmatrix('x')
        y = TT.vector('y')
        [h, s], _ = theano.scan(fn = _recurrence, sequences = x, \
                                outputs_info = [self.H0, None], \
                                n_steps = x.shape[0])

        p_y_given_x_lastword = s[-1,0,:]
        y_pred = TT.argmax(p_y_given_x_lastword, axis = 0)
        nll = TT.sqrt(TT.sum(TT.pow(y - TT.log(p_y_given_x_lastword), 2)))
        gradients = TT.grad(nll, self._params)
        updates = OrderedDict((p, p - self.alpha * g) for p, g in \
                                  zip(self._params , gradients))
        # compile training function
        train = theano.function(inputs = [x, y], outputs = nll, updates = updates)
        # convert symbolic features to embedding indices
        a_trainset = self._digitize_feats(a_trainset)
        # perform training
        prev_s = s = INF
        for _ in xrange(MAX_ITERS):
            s = 0.
            for x_i, y_i in a_trainset:
                # print("x =", repr(x), file = sys.stderr)
                # print("y =", repr(y), file = sys.stderr)
                s += train(x_i, y_i)
            if prev_s != s and (prev_s - s) < EPSILON:
                break
            prev_s = s
            print("s =", repr(s), file = sys.stderr)
        # set prediction function
        self._predict = theano.function(inputs = [x], outputs = y_pred)

    def predict(self, a_seq):
        """Prediction function

        @param a_seq - input sequence whose class should be predicted

        @return predicted label

        """
        y = int(self._predict(self._feat2idcs(a_seq)))
        return self.int2lbl[y]

    def _digitize_feats(self, a_trainset):
        """Convert features and target classes to vectors and ints.

        @param a_trainset - training set as a list of 2-tuples with
                            training instances and classes

        @return new list of digitized features and classes

        """
        assert self.nlabels > 0, "Invalid number of labels."
        assert self.V > 0, "Invalid size of feature vocabulary."
        ret = []
        clabels = 0
        dlabel = None; dint = coeff = 1.
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
                # dlabel = dint
                dlabel = np.zeros(self.nlabels)
                dlabel[dint] = 1 * self.int2coeff[dint]
            # convert features to indices and append new training
            # instance
            ret.append((self._feat2idcs(iseq, a_add = True), dlabel))
        return ret

    def _feat2idcs(self, a_seq, a_add = False):
        """Convert features to their indices.

        @param a_seq - sequence of features to be converted
        @param a_add - boolean flag indicating whether features should
                       be added to an internal dictionary

        @return list of lists of feature indices within a given context

        """
        # append auxiliary inout items
        a_seq[:0] = [BEG]; a_seq.append(END)
        # initialize matrix of feature indices
        ditems = np.empty((len(a_seq) + 1 - CW, CW), dtype = int)
        # convert features to vector indices for the first
        feat_idcs = []
        cfeats = len(self.feat2idx)
        for iitem in a_seq[:CW]:
            if a_add and iitem not in self.feat2idx:
                self.feat2idx[iitem] = cfeats
                cfeats += 1
            feat_idcs.append(self.feat2idx.get(iitem, UNK))
        while len(feat_idcs) < CW:
            feat_idcs.append(self.feat2idx[UNK])
        ditems[0,:] = feat_idcs
        i = 1
        for iitem in a_seq[CW:]:
            if a_add and iitem not in self.feat2idx:
                self.feat2idx[iitem] = cfeats
                cfeats += 1
            # unshift first element
            feat_idcs[:1] = []
            # append new element
            feat_idcs.append(self.feat2idx.get(iitem, UNK))
            ditems[i,:] = feat_idcs
            i += 1
        return ditems

    def _reset(self):
        """Reset instance variables.

        @return \c void

        """
        self._predict = None
