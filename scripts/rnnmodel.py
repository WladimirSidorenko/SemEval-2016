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
ADADELTA = 0
SGD = 1

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
def _rnd_orth_mtx(a_dim):
    """Return orthogonal matrix with random weights.

    @param a_dim - dimensionality of square matrix

    @return orthogonal Theano matrix with random weights

    """
    W = np.random.randn(a_dim, a_dim)
    u, _, _ = numpy.linalg.svd(W)
    return u.astype(config.floatX)

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

    def __init__(self, a_vdim = VEC_DIM, a_use_dropout = True):
        """Class constructor.

        @param a_vdim - default dimensionality of embedding vectors
        @param a_use_dropout - boolean flag indicating whether to use dropout

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
        self.alpha = ALPHA
        self.use_dropout = a_use_dropout
        # NN parameters to be learned
        self._params = []

        # the parameters below will be initialized during training
        # private prediction function
        self._predict = None
        # matrix of items' embeddings (either words or characters) that serve as input to RNN
        self.EMB = self.EMB_I = self.CONV_IN = None
        # output of convolutional layers
        self.CONV2_OUT = self.CONV3_OUT = self.CONV4_OUT = None
        # max-sample output of convolutional layers
        self.CONV2_MAX_OUT = self.CONV3_MAX_OUT = self.CONV4_MAX_OUT = self.CONV_MAX_OUT = None
        # custom recurrence function (hiding LSTM)
        self._recurrence = None
        # map from hidden layer to output and bias for the output layer
        self.H = self.H2Y = self.Y_BIAS = None

        # the remianing parameters will be initialized immediately
        self._init_params()

    def fit(self, a_trainset, a_batch_size = 16, a_optimizer = ADADELTA):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
        @param a_batch_size - size of single training batch
        @param a_optimizer - optimizer to use (ADADELTA or SGD)

        @return \c void

        """
        # estimate the number of distinct features and the longest sequence
        featset = set()
        self.max_len = 0
        for wlist, _ in a_trainset:
            for w in wlist:
                featset.update(w)
                # append auxiliary items to training instances
                w[:0] = [BEG]; w.append(END)
                self.max_len = max(self.max_len, len(w))
        self.V = len(AUX_VEC_KEYS) + len(featset)
        del featset
        self.n_labels = len(set([t[1] for t in a_trainset]))

        # initialize embedding matrix for features
        self._init_emb()

        # initialize convolutional layers
        self._init_conv()

        # initialize LSTM layer
        self._init_lstm()

        # hidden layer
        self.H = TT.nnet.relu(TT.dot(self.CONV_MAX_OUT, self.CONV2H) + self.H_BIAS)
        # mapping from hidden layer to output
        self.H2Y = theano.shared(value = RND_VEC((self.n_hidden, self.n_labels)),
                                 name = "H2Y")
        # output bias
        self.Y_BIAS = theano.shared(value = RND_VEC((1, self.n_labels)),
                                    name = "Y_BIAS")
        # add newly initialized weights to the parameters to be trained
        self._params += [self.EMB, self.H2Y, self.Y_BIAS]

        # output layer
        self.Y = TT.nnet.softmax(TT.dot(self.H, self.H2Y) + self.Y_BIAS)
        # predicted label
        y = TT.iscalar('y')
        y_pred = TT.argmax(self.Y, axis = 1)

        # cost gradients and updates
        alpha = TT.scalar("alpha")
        cost = -TT.log(self.Y[0,y])
        gradients = TT.grad(cost, self._params)
        updates = OrderedDict((p, p - alpha * g) for p, g in zip(self._params , gradients))

        # define training function and let training begin
        train = theano.function(inputs  = [self.INDICES, y, alpha], \
                                outputs = cost, updates = updates)

        icost = None
        a_trainset = self._digitize_feats(a_trainset)
        for x_i, y_i in a_trainset:
            print("x_i =", repr(x_i), file = sys.stderr)
            print("y_i =", repr(y_i), file = sys.stderr)
            for iword in x_i:
                icost = train(iword, y_i, self.alpha)
                print("icost =", repr(icost), file = sys.stderr)
                # print("f_conv3 =", repr(self.f_conv3(iword)), file = sys.stderr)
                # print("f_conv4 =", repr(self.f_conv4(iword)), file = sys.stderr)
                break
            break
        sys.exit(66)

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

    def _init_params(self):
        """Initialize parameters which are independent of the training data.

        @return \c void

        """
        self.n_hidden = self.n_lstm = self.vdim
        # auxiliary zero matrix used for padding the input
        self._subzero = TT.zeros((self.max_conv_len, self.vdim))
        self.INDICES = TT.ivector(name = "INDICES")
        # number of embeddings per training item
        self.M_EMB = TT.shape(self.INDICES)[0]
        # number of padding rows
        self.M_PAD = TT.max([self.max_conv_len - self.M_EMB, 0])
        # length of input
        self.IN_LEN = self.M_EMB + self.M_PAD

        ################
        # CONVOLUTIONS #
        ################
        # three convolutional filters for strides of width 2
        self.n_conv2 = 3 # number of filters
        self.conv2_width = 2 # width of stride
        self.CONV2 = theano.shared(value = RND_VEC((self.n_conv2, 1, self.conv2_width, self.vdim)), \
                                       name = "CONV2")
        self.CONV2_BIAS = theano.shared(value = RND_VEC((1, self.n_conv2)), name = "CONV2_BIAS")
        # four convolutional filters for strides of width 3
        self.n_conv3 = 4 # number of filters
        self.conv3_width = 3 # width of stride
        self.CONV3 = theano.shared(value = RND_VEC((self.n_conv3, 1, self.conv3_width, self.vdim)), \
                                       name = "CONV3")
        self.CONV3_BIAS = theano.shared(value = RND_VEC((1, self.n_conv3)), name = "CONV3_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv4 = 5 # number of filters
        self.conv4_width = 4 # width of stride
        self.CONV4 = theano.shared(value = RND_VEC((self.n_conv4, 1, self.conv4_width, self.vdim)), \
                                       name = "CONV4")
        self.CONV4_BIAS = theano.shared(value = RND_VEC((1, self.n_conv4)), name = "CONV4_BIAS")
        # remember parameters to be learned
        self._params += [self.CONV2, self.CONV3, self.CONV4, \
                         self.CONV2_BIAS, self.CONV3_BIAS, self.CONV4_BIAS]
        ########
        # LSTM #
        ########
        self.LSTM_W = theano.shared(value = np.concatenate([_rnd_orth_mtx(self.n_lstm) \
                                                                for _ in xrange(4)], axis = 1), \
                                        name = "LSTM_W")
        self.LSTM_U = theano.shared(value = np.concatenate([_rnd_orth_mtx(self.n_lstm) \
                                                                for _ in xrange(4)], axis = 1), \
                                        name = "LSTM_U")
        self.LSTM_BIAS = theano.shared(RND_VEC((1, self.n_lstm * 4)), name = "LSTM_BIAS")
        self._params += [self.LSTM_W, self.LSTM_U, self.LSTM_BIAS]


    def _init_emb(self):
        """Initialize embeddings.

        @return \c void

        """
        self.EMB = theano.shared(value = RND_VEC((self.V, self.vdim)))
        # obtain indices for special embeddings (BEGINNING, END, UNKNOWN)
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
        # add embeddings to the parameters to be trained
        self._params.append(self.EMB)

    def _init_conv(self):
        """Initialize parameters of convolutional layer.

        @return \c void

        """
        # embeddings obtained for specific indices
        self.EMB_I = self.EMB[self.INDICES]
        # input to convolutional layer
        self.CONV_IN = TT.concatenate([self.EMB_I, self._subzero[:self.M_PAD,:]], \
                                          0).reshape((1, 1, self.IN_LEN, self.vdim))
        # width-2 convolutions
        self.CONV2_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV2), \
                                        (self.n_conv2, self.IN_LEN - self.conv2_width + 1)).T
        self.CONV2_MAX_OUT = self.CONV2_OUT[TT.argmax(TT.sum(self.CONV2_OUT, axis = 1)),:] + \
                             self.CONV2_BIAS
        # width-3 convolutions
        self.CONV3_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV3), \
                                        (self.n_conv3, self.IN_LEN - self.conv3_width + 1)).T
        self.CONV3_MAX_OUT = self.CONV3_OUT[TT.argmax(TT.sum(self.CONV3_OUT, axis = 1)),:] + \
                             self.CONV3_BIAS
        # width-4 convolutions
        self.CONV4_OUT = TT.reshape(TT.nnet.conv.conv2d(self.CONV_IN, self.CONV4), \
                                        (self.n_conv4, self.IN_LEN - self.conv4_width + 1)).T
        self.CONV4_MAX_OUT = self.CONV4_OUT[TT.argmax(TT.sum(self.CONV4_OUT, axis = 1)),:] + \
                             self.CONV4_BIAS
        # output convolutions
        self.CONV_MAX_OUT = TT.nnet.relu(TT.concatenate([self.CONV2_MAX_OUT, self.CONV3_MAX_OUT, \
                                                             self.CONV4_MAX_OUT], axis = 1))

    def _init_lstm(self):
        """Initialize parameters of LSTM layer.

        @return \c void

        """
        pass

    def _reset(self):
        """Reset instance variables.

        @return \c void

        """
        self._predict = None

##################################################################
# Auxiliary Debug Methods and Variables

# # auxiliary debug variables and methods
# CONV2_OUT = TT.nnet.conv.conv2d(self.CONV_IN, self.CONV2)
# CONV2_OUT_RESHAPE = TT.reshape(CONV2_OUT, (self.n_conv2, self.IN_LEN - self.conv2_width + 1)).T
# CONV2_MAX_OUT = CONV2_OUT_RESHAPE[TT.argmax(TT.sum(self.CONV2_OUT, axis = 1)),:]
# CONV2_MAX_OUT_BIAS = CONV2_MAX_OUT + self.CONV2_BIAS

# # get_ilen = theano.function([self.INDICES], self.IN_LEN)
# f_conv2 = theano.function([self.INDICES], self.CONV2_MAX_OUT)
# f_conv3 = theano.function([self.INDICES], self.CONV3_MAX_OUT)
# f_conv4 = theano.function([self.INDICES], self.CONV4_MAX_OUT)
# get_emb = theano.function([self.INDICES], self.EMB_I)
# get_conv_in = theano.function([self.INDICES], self.CONV_IN)
# get_conv_out = theano.function([self.INDICES], CONV2_OUT)
# get_conv_out_reshape = theano.function([self.INDICES], CONV2_OUT_RESHAPE)
# get_conv_out_max = theano.function([self.INDICES], CONV2_MAX_OUT)
# get_conv_out_max_bias = theano.function([self.INDICES], CONV2_MAX_OUT_BIAS)
# get_conv_out_total = theano.function([self.INDICES], self.CONV_MAX_OUT)
# get_H = theano.function([self.INDICES], self.H)
# get_Y = theano.function([self.INDICES], self.Y)
# get_y_pred = theano.function([self.INDICES], y_pred)
# get_cost = theano.function([self.INDICES, y], cost)

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
# a_trainset = self._digitize_feats(a_trainset)
# sys.exit(66)
# perform training
# prev_s = s = INF
# set prediction function
