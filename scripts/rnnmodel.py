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

from theano import printing, tensor as TT
from collections import OrderedDict
from itertools import chain

##################################################################
# Variables and Constants
INF = float("inf")

# default training parameters
ALPHA = 5e-3
EPSILON = 1e-5
MAX_ITERS = 50

# default dimension of input vectors
VEC_DIM = 4                     # 32
# default context window
# CW = 1

# initial parameters for uniform distribution
UMIN = -1.5
UMAX = 1.5

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

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
        self.max_len = 0        # maximum length of an input item
        # maximum width of a convolution stride
        self.max_conv_len = theano.shared(value = 5, name = "MAX_CONV_LEN")
        # mapping from symbolic representations to indices
        self.lbl2int = dict()
        self.int2lbl = dict()
        self.int2coeff = dict()
        self.feat2idx = dict()
        self.alpha = ALPHA
        # auxiliary zero matrix used for padding the input
        self._subzero = TT.zeros((self.max_conv_len, self.vdim))
        # matrix of items' embeddings (either words or characters)
        self.EMB = None
        self.INDICES = TT.ivector(name = "INDICES")
        # number of embeddings per training item
        self.M_EMB = TT.shape(self.INDICES)[0]
        # number of padding rows
        self.M_PAD = TT.max([self.max_conv_len - self.M_EMB, 0])
        # length of input
        self.IN_LEN = self.M_EMB + self.M_PAD
        # three convolutional filters for strides of width 2
        self.n_conv2 = 3 # number of filters
        self.conv2_width = 2 # width of stride
        self.CONV2 = theano.shared(value = RND_VEC((self.n_conv2, 1, self.conv2_width, self.vdim)), \
                                       name = "CONV2")
        # four convolutional filters for strides of width 3
        self.n_conv3 = 4 # number of filters
        self.conv3_width = 3 # width of stride
        self.CONV3 = theano.shared(value = RND_VEC((self.n_conv3, 1, self.conv3_width, self.vdim)), \
                                       name = "CONV3")
        # five convolutional filters for strides of width 4
        self.n_conv4 = 5 # number of filters
        self.conv4_width = 4 # width of stride
        self.CONV4 = theano.shared(value = RND_VEC((self.n_conv4, 1, self.conv4_width, self.vdim)), \
                                       name = "CONV4")
        # map from hidden layer to output
        self.H2Y = None
        # bias vector for the output layer (1 x nlabels)
        self.YBV = None
        # private prediction function (will be initialized after training)
        self._predict = None
        # auxiliary variable for keeping track of parameters
        self._params = [self.CONV2, self.CONV3, self.CONV4]

    def fit(self, a_trainset):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes

        @return \c void

        """
        # estimate the number of distinct features and the longest sequence
        featset = set()
        self.max_len = 0
        for wlist, _ in a_trainset:
            for w in wlist:
                # append auxiliary items to training instances
                w[:0] = [BEG]; w.append(END)
                self.max_len = max(self.max_len, len(w))
                featset.update(w)
        self.V = len(AUX_VEC_KEYS) + len(featset)
        del featset

        # initialize embedding matrix for features
        self.EMB = theano.shared(value = RND_VEC((self.V, self.vdim)))
        # embeddings obtained for specific indices
        self.EMB_I = self.EMB[self.INDICES]
        # input to convolutional layer
        self.CONV_IN = TT.concatenate([self.EMB_I, self._subzero[:self.M_PAD,:]], \
                                          0).reshape((1, 1, self.IN_LEN, self.vdim))
        self.CONV2_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV2), \
                                        (self.n_conv2, self.IN_LEN - self.conv2_width + 1)).T
        self.CONV2_MAX_OUT = self.CONV2_OUT[TT.argmax(TT.sum(self.CONV2_OUT, axis = 1)),:]
        self.f_conv2 = theano.function([self.INDICES], self.CONV2_MAX_OUT)

        self.CONV3_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV3), \
                                        (self.n_conv3, self.IN_LEN - self.conv3_width + 1)).T
        self.CONV3_MAX_OUT = self.CONV3_OUT[TT.argmax(TT.sum(self.CONV3_OUT, axis = 1)),:]
        self.f_conv3 = theano.function([self.INDICES], self.CONV3_MAX_OUT)

        self.CONV4_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV4), \
                                        (self.n_conv4, self.IN_LEN - self.conv4_width + 1)).T
        self.CONV4_MAX_OUT = self.CONV4_OUT[TT.argmax(TT.sum(self.CONV4_OUT, axis = 1)),:]
        self.f_conv4 = theano.function([self.INDICES], self.CONV4_MAX_OUT)

        # prepend embeddings matrix to the list of parameters to be trained
        self._params[0:0] = [self.EMB]
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
        self.nlabels = len(set([t[1] for t in a_trainset]))

        # auxiliary debug methods
        # CONV2_OUT = TT.nnet.conv.conv2d(self.CONV_IN, self.CONV2)
        # CONV2_OUT_RESHAPE = TT.reshape(CONV2_OUT, (self.n_conv2, self.IN_LEN - self.conv2_width + 1)).T
        # CONV2_MAX_OUT = CONV2_OUT_RESHAPE[TT.argmax(TT.sum(self.CONV2_OUT, axis = 1)),:]

        # get_ilen = theano.function([self.INDICES], self.IN_LEN)
        get_emb = theano.function([self.INDICES], self.EMB_I)
        # get_conv_in = theano.function([self.INDICES], self.CONV_IN)
        # get_conv_out = theano.function([self.INDICES], CONV2_OUT)
        # get_conv_out_reshape = theano.function([self.INDICES], CONV2_OUT_RESHAPE)
        # get_conv_out_max = theano.function([self.INDICES], CONV2_MAX_OUT)

        a_trainset = self._digitize_feats(a_trainset)
        # print("a_trainset =", repr(a_trainset), file = sys.stderr)
        # print("self.CONV2 =", self.CONV2.eval(), file = sys.stderr)
        for x_i, y_i in a_trainset:
            print("x_i =", repr(x_i), file = sys.stderr)
            # print("y =", repr(y), file = sys.stderr)
            for iword in x_i:
                print("emb_i = ", repr(get_emb(iword)), file = sys.stderr)
                # print("conv_in = ", repr(get_conv_in(iword)), file = sys.stderr)
                # print("ILEN = ", repr(get_ilen(iword)), file = sys.stderr)
                # print("conv2 =", repr(get_conv_out(iword)), file = sys.stderr)
                # print("conv2.reshape =", repr(get_conv_out_reshape(iword)), file = sys.stderr)
                # print("conv2.max =", repr(get_conv_out_max(iword)), file = sys.stderr)
                # print("conv2_max =", repr(get_conv_out_max(iword)), file = sys.stderr)
                print("self.conv2_max =", repr(self.f_conv2(iword)), file = sys.stderr)
                print("self.conv3_max =", repr(self.f_conv3(iword)), file = sys.stderr)
                print("self.conv4_max =", repr(self.f_conv4(iword)), file = sys.stderr)
                # print("f_conv3 =", repr(self.f_conv3(iword)), file = sys.stderr)
                # print("f_conv4 =", repr(self.f_conv4(iword)), file = sys.stderr)
                break
            break
        sys.exit(66)
        # initialize prediction matrix
        # self.H2Y = theano.shared(value = RND_VEC((self.vdim, self.nlabels)))
        # self.YBV = theano.shared(value = RND_VEC((1, self.nlabels)))
        # self._params.append(self.H2Y); self._params.append(self.YBV)

        # # define custom recursive function
        # def _recurrence(x_t):
        #     # embedding layer
        #     emb_t = self.EMB[x_t].reshape([1, CW * self.vdim], ndim = 2)
        #     # embedding layer propagated via tensor
        #     # in_t = TT.dot(TT.tensordot(emb_t, self.E2H, [[1], [1]]), emb_t.T)
        #     # in_t = in_t.reshape([1, self.vdim], ndim = 2)
        #     # print("in_t.shape", repr(in_t.shape), file = sys.stderr)
        #     # 0-th hidden layer
        #     h0_t = TT.nnet.relu(TT.dot(emb_t, self.E2H) + self.H0BV, 0.5)
        #     h1_t = TT.nnet.sigmoid(TT.dot(h0_t, self.H02H1) + self.H1BV)
        #     h2_t = TT.nnet.relu(TT.dot(h1_t, self.H12H2) + self.H2BV)
        #     # print("h_t.shape", repr(h_t.shape), file = sys.stderr)
        #     s_t = TT.nnet.softmax(TT.dot(h2_t, self.H2Y) + self.YBV)
        #     return [s_t]

        # # auxiliary variables used for training
        # X = TT.lmatrix('X')
        # # y = TT.vector('y')
        # y = TT.iscalar('y')
        # s, _ = theano.scan(fn = _recurrence, sequences = X, \
        #                    n_steps = X.shape[0])

        # p_y_given_x_lastword = s[-1,0,:]
        # y_pred = TT.argmax(p_y_given_x_lastword, axis = 0)
        # score = p_y_given_x_lastword[y_pred]
        # nll = -TT.log(p_y_given_x_lastword)[y]
        # # nll =  TT.sum(TT.pow(y - TT.log(p_y_given_x_lastword), 2))
        # gradients = TT.grad(nll, self._params)
        # updates = OrderedDict((p, p - self.alpha * g) for p, g in \
        #                           zip(self._params , gradients))
        # # compile training function
        # train = theano.function(inputs = [x, y], outputs = nll, updates = updates)
        # convert symbolic features to embedding indices
        a_trainset = self._digitize_feats(a_trainset)
        sys.exit(66)
        # perform training
        prev_s = s = INF
        # set prediction function
        self._predict = theano.function(inputs = [x], outputs = [y_pred, score])
        for _ in xrange(MAX_ITERS):
            s = 0.
            for x_i, y_i in a_trainset:
                # print("x =", repr(x), file = sys.stderr)
                # print("y =", repr(y), file = sys.stderr)
                # s += train(x_i, y_i)
                pass
            if prev_s != s and (prev_s - s) < EPSILON:
                break
            prev_s = s
            print("s =", repr(s), file = sys.stderr)

    def predict(self, a_seq):
        """Prediction function

        @param a_seq - input sequence whose class should be predicted

        @return 2-tuple with predicted label and its assigned score

        """
        y, score = self._predict(self._feat2idcs(a_seq))
        return (self.int2lbl[int(y)], score)

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
                # dlabel = np.zeros(self.nlabels)
                # dlabel[dint] = 1 * self.int2coeff[dint]
                dlabel = dint
            # convert features to indices and append new training
            # instance
            ret.append((self._feat2idcs(iseq, a_add = True), dlabel))
            break
        return ret

    def _feat2idcs(self, a_seq, a_add = False):
        """Convert features to their indices.

        @param a_seq - sequence of features to be converted
        @param a_add - boolean flag indicating whether features should
                       be added to an internal dictionary

        @return list of lists of feature indices within a given context

        """
        print("self.max_len = ", repr(self.max_len), file = sys.stderr)
        print("a_seq = ", repr(a_seq), file = sys.stderr)
        # initialize matrix of feature indices
        ditems = []
        # convert features to vector indices
        feat_idcs = None
        cfeats = len(self.feat2idx)
        for iword in a_seq:
            feat_idcs = []
            # append auxiliary items
            for ichar in iword:
                if a_add and ichar not in self.feat2idx:
                    self.feat2idx[ichar] = cfeats
                    cfeats += 1
                feat_idcs.append(self.feat2idx.get(ichar, UNK))
            # pad with zeros
            # emb = TT.concatenate([self.EMB[feat_idcs], \
            #                           self._subzero[:max(self.max_conv_len - len(iword), 0),:]], 0)
            # print("EMB: ", repr(emb.eval()))
            # ditems.append(emb.reshape((1, 1, max(self.max_conv_len, len(iword)), self.vdim)))
            ditems.append(feat_idcs)
        # print("ditems = ", repr([i.shape.eval() for i in ditems]), file = sys.stderr)
        return ditems

    def _reset(self):
        """Reset instance variables.

        @return \c void

        """
        self._predict = None
