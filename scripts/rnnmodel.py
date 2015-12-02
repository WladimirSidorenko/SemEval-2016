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

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config, printing, tensor as TT
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
EMP = "___%EMP%___"
UNK = "___%UNK%___"
BEG = "___%BEG%___"
END = "___%END%___"
AUX_VEC_KEYS = [EMP, UNK, BEG, END]

##################################################################
# Methods
def _floatX(data):
    """Return numpy array populated with the given data.

    @param data - input tensor

    @return numpy array populated with the given data

    """
    return np.asarray(data, dtype = config.floatX)

def _slice(_x, n, dim):
    """Return slice of input tensor from the last two dimensions.

    @param _x - input tensor
    @param n - input tensor
    @param dim - input tensor

    @return tensor of size `n * dim x (n + 1) * dim`

    """
    if _x.ndim == 3:
        return _x[:, :, n * dim:(n + 1) * dim]
    return _x[:, n * dim:(n + 1) * dim]

def _rnd_orth_mtx(a_dim):
    """Return orthogonal matrix with random weights.

    @param a_dim - dimensionality of square matrix

    @return orthogonal Theano matrix with random weights

    """
    W = np.random.randn(a_dim, a_dim)
    u, _, _ = np.linalg.svd(W)
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
        self.alpha = ALPHA
        self.optimizer = ADADELTA
        self.use_dropout = False
        # NN parameters to be learned
        self._params = []

        # custom function for computing convolutions from indices
        self._emb2conv = None
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
        self.LSTM2Y = self.Y_BIAS = self.Y = None

        # the remianing parameters will be initialized immediately
        self._init_params()

    def fit(self, a_trainset, a_use_dropout = True, a_batch_size = 16, a_optimizer = ADADELTA):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
        @param a_use_dropout - boolean flag indicating whether to use dropout
        @param a_batch_size - size of single training batch
        @param a_optimizer - optimizer to use (ADADELTA or SGD)

        @return \c void

        """
        self.use_dropout = a_use_dropout
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

        # initialize embedding 2 convolutions function
        self._init_conv()

        # initialize LSTM layer
        lstm_out = self._init_lstm()
        lstm_debug = theano.function([self.W_INDICES], lstm_out, name = "lstm_debug")

        print("res =", lstm_out[-1], type(lstm_out[-1]))
        print("res.ndim =", lstm_out[-1][-1].ndim)
        # mapping from the LSTM layer to output
        self.LSTM2Y = theano.shared(value = RND_VEC((self.n_lstm, self.n_labels)),
                                    name = "LSTM2Y")
        # output bias
        self.Y_BIAS = theano.shared(value = RND_VEC((1, self.n_labels)),
                                    name = "Y_BIAS")
        # # output layer
        # self.Y = TT.nnet.softmax(TT.dot(lstm_out[0], self.LSTM2Y) + self.Y_BIAS)

        # add newly initialized weights to the parameters to be trained
        self._params += [self.LSTM2Y, self.Y_BIAS]

        # correct label
        # y = TT.iscalar('y')
        # # predicted label
        # y_pred = TT.argmax(self.Y, axis = 1)

        # # cost gradients and updates
        # alpha = TT.scalar("alpha")
        # cost = -TT.log(self.Y[0,y])
        # gradients = TT.grad(cost, self._params)
        # updates = OrderedDict((p, p - alpha * g) for p, g in zip(self._params , gradients))

        # # define training function and let training begin
        # train = theano.function(inputs  = [self.W_INDICES, y, alpha], \
        #                         outputs = [cost], updates = updates)

        icost = None
        a_trainset = self._digitize_feats(a_trainset)
        for x_i, y_i in a_trainset:
            print("x_i =", repr(x_i), file = sys.stderr)
            print("y_i =", repr(y_i), file = sys.stderr)
            print("lstm_debug =", repr(lstm_debug(x_i)), file = sys.stderr)
            # for iword in x_i:
            #     icost = train(iword, y_i, self.alpha)
            #     print("icost =", repr(icost), file = sys.stderr)
            #     # print("f_conv3 =", repr(self.f_conv3(iword)), file = sys.stderr)
            #     # print("f_conv4 =", repr(self.f_conv4(iword)), file = sys.stderr)
            #     break
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
            ret.append((np.asarray(self._feat2idcs(iseq, a_add = True), \
                                   dtype = "int32"), dlabel))
            break
        return ret

    def _feat2idcs(self, a_seq, a_add = False):
        """Convert features to their indices.

        @param a_seq - sequence of features to be converted
        @param a_add - boolean flag indicating whether features should
                       be added to an internal dictionary

        @return list of lists of feature indices within a given context

        """
        # initialize matrix of feature indices
        ditems = []
        # convert features to vector indices
        feat_idcs = None
        cfeats = len(self.feat2idx)
        # determine maximum word length in sequence
        max_len = max(max([len(w) for w in a_seq]), self.conv2_width, \
                      self.conv3_width, self.conv4_width)
        ilen = 0
        for iword in a_seq:
            feat_idcs = []
            print("iword = ", repr(iword))
            # append auxiliary items
            ilen = len(iword)
            for ichar in iword:
                if a_add and ichar not in self.feat2idx:
                    self.feat2idx[ichar] = cfeats
                    cfeats += 1
                feat_idcs.append(self.feat2idx.get(ichar, UNK))
            # pad indices with embeddings for empty character
            if ilen < max_len:
                feat_idcs += [self.feat2idx[EMP]] * (max_len - ilen)
            ditems.append(feat_idcs)
        # print("ditems = ", repr([i.shape.eval() for i in ditems]), file = sys.stderr)
        return ditems

    def _init_params(self):
        """Initialize parameters which are independent of the training data.

        @return \c void

        """
        # auxiliary zero matrix used for padding the input
        self._subzero = TT.zeros((self.max_conv_len, self.vdim))
        # matrix of char vectors, corresponding to single word
        self.W_INDICES = TT.imatrix(name = "W_INDICES")
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
        self.n_lstm = self.n_conv2 + self.n_conv3 + self.n_conv4
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
        """Initialize function for computing convolutions from indices

        @return \c void

        """
        def _emb2conv(a_x):
            """Private function for computing convolutions from indices

            @param a_x - indices of embeddings

            @return max convolutions computed from these indices

            """
            # length of character input
            in_len = a_x.shape[0]
            # input to convolutional layer
            conv_in = self.EMB[a_x].reshape((1, 1, in_len, self.vdim))
            # width-2 convolutions
            conv2_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV2), \
                                   (self.n_conv2, in_len - self.conv2_width + 1)).T
            conv2_max_out = conv2_out[TT.argmax(TT.sum(conv2_out, axis = 1)),:] + \
                            self.CONV2_BIAS
            # width-3 convolutions
            conv3_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV3), \
                                   (self.n_conv3, in_len - self.conv3_width + 1)).T
            conv3_max_out = conv3_out[TT.argmax(TT.sum(conv3_out, axis = 1)),:] + \
                            self.CONV3_BIAS
            # width-4 convolutions
            conv4_out = TT.reshape(TT.nnet.conv.conv2d(conv_in, self.CONV4), \
                                   (self.n_conv4, in_len - self.conv4_width + 1)).T
            conv4_max_out = conv4_out[TT.argmax(TT.sum(conv4_out, axis = 1)),:] + \
                            self.CONV4_BIAS
            # output convolutions
            conv_max_out = TT.nnet.relu(TT.concatenate([conv2_max_out, conv3_max_out, \
                                                        conv4_max_out], axis = 1))
            return conv_max_out
        self._emb2conv = _emb2conv

    def _init_lstm(self):
        """Initialize parameters of LSTM layer.

        @return 2-tuple with result and updates of LSTM scan

        """
        # single LSTM recurrence step function
        def _lstm_step(x_, o_, m_):
            """Single LSTM recurrence step.

            @param x_ - indices of input characters
            @param o_ - previous output
            @param m_ - previous state of memory cell

            @return 2-tuple (with the output and memory cells)

            """
            # print("_lstm_step: x_", x_.eval())
            # print("_lstm_step: o_", o_.eval())
            # print("_lstm_step: m_", m_.eval())
            # obtain character convolutions for input indices
            iconv = self._emb2conv(x_)
            # print("_lstm_step: iconv", iconv.eval())
            # common term for all LSTM components
            proxy = TT.dot(iconv, self.LSTM_W) + TT.dot(o_, self.LSTM_U) + \
                self.LSTM_BIAS
            # input
            i = TT.nnet.sigmoid(_slice(proxy, 0, self.n_lstm))
            # forget
            f = TT.nnet.sigmoid(_slice(proxy, 1, self.n_lstm))
            # output
            o = TT.nnet.sigmoid(_slice(proxy, 2, self.n_lstm))
            # new state of memory cell (input * current + forget * previous)
            m = i * TT.tanh(_slice(proxy, 3, self.n_lstm)) + f * m_
            # new state of hidden recurrence
            o = o * TT.tanh(m)
            # return new state of memory cell and state of hidden recurrence
            return o, m
        # `scan' function
        res, _ = theano.scan(_lstm_step,
                             sequences = [self.W_INDICES],
                             outputs_info = [TT.alloc(_floatX(0.),
                                                      self.W_INDICES.shape[0],
                                                      self.n_lstm),
                                             TT.alloc(_floatX(0.),
                                                      self.W_INDICES.shape[0],
                                                      self.n_lstm)],
                                   name = "_lstm_layers")
        return res

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
