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

# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config, tensor as TT
from collections import OrderedDict
from itertools import chain

##################################################################
# Variables and Constants
INF = float("inf")

# default training parameters
ALPHA = 5e-3
EPSILON = 1e-5
MAX_ITERS = 2 # 500
ADADELTA = 0
SGD = 1

# default dimension of input vectors
VEC_DIM = 32
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
RND_VEC = lambda a_dim = VEC_DIM: np.random.uniform(UMIN, UMAX, a_dim).astype(config.floatX)

# symbolic codes for auxiliary vectors
EMP = "___%EMP%___"
UNK = "___%UNK%___"
BEG = "___%BEG%___"
END = "___%END%___"
AUX_VEC_KEYS = [EMP, UNK, BEG, END]

# theano options
config.allow_gc = True
config.scan.allow_gc = True

# theano profiling
# specify on the command line: THEANO_FLAGS=device=cpu,optimizer_excluding=fusion:inplace
# config.profile = True
# config.profile_memory = True
# config.profile_optimizer = True

##################################################################
# Methods
def _floatX(data):
    """Return numpy array populated with the given data.

    @param data - input tensor

    @return numpy array populated with the given data

    """
    return np.asarray(data, dtype = config.floatX)

def _rnd_orth_mtx(a_dim):
    """Return orthogonal matrix with random weights.

    @param a_dim - dimensionality of square matrix

    @return orthogonal Theano matrix with random weights

    """
    W = np.random.randn(a_dim, a_dim)
    u, _, _ = np.linalg.svd(W)
    return u.astype(config.floatX)


def adadelta(tparams, grads, x, y, cost):
    """An adaptive learning rate optimizer

    @param tpramas - model parameters
    @param grads - gradients of cost w.r.t to parameres
    @param x - model inputs
    @param y - targets
    @param cost - objective function to minimize

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
    # rg2up = [(rg2, rg2 + g)
    #          for rg2, g in zip(running_grads2, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost, updates = zgup + rg2up,
                                    name = "adadelta_f_grad_shared")

    # updir = [-TT.sqrt(ru2) / TT.sqrt(rg2) * zg
    #          for zg, ru2, rg2 in zip(zipped_grads,
    #                                  running_up2,
    #                                  running_grads2)]
    updir = [-TT.sqrt(ru2 + 1e-6) / TT.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    # ru2up = [(ru2, ru2 + ud)
    #          for ru2, ud in zip(running_up2, updir)]
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
        # self.alpha = ALPHA
        # self.optimizer = ADADELTA
        self.use_dropout = False
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
        # custom recurrence function (hiding LSTM)
        # self._recurrence = None
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

        # initialize LSTM layer
        lstm_out = self._init_lstm()
        # # LSTM debug function
        # lstm_debug = theano.function([self.W_INDICES], lstm_out, name = "lstm_debug")

        # mapping from the LSTM layer to output
        self.LSTM2Y = theano.shared(value = RND_VEC((self.n_lstm, self.n_labels)),
                                    name = "LSTM2Y")
        # output bias
        self.Y_BIAS = theano.shared(value = RND_VEC((self.n_labels)), name = "Y_BIAS")
        # output layer
        self.Y = TT.nnet.softmax(TT.dot(lstm_out[-1], self.LSTM2Y) + self.Y_BIAS)

        # add newly initialized weights to the parameters to be trained
        self._params += [self.LSTM2Y, self.Y_BIAS]

        # correct label
        y = TT.scalar('y', dtype = "int32")

        # cost gradients and updates
        cost = -TT.log(self.Y[0, y])
        gradients = TT.grad(cost, wrt = self._params)
        f_grad = theano.function([self.W_INDICES, y], gradients, name = "f_grad")

        # define training function and let the training begin
        f_grad_shared, f_update = adadelta(self._params, gradients, \
                                           self.W_INDICES, y, cost)

        # print("self._params =", self._params, file = sys.stderr)
        # alpha = TT.scalar("alpha")
        # updates = OrderedDict((p, p - alpha * g) for p, g in zip(self._params , gradients))
        # train = theano.function(inputs  = [self.W_INDICES, y, alpha], \
        #                             outputs = cost, updates = updates)

        prev_cost = icost = best_cost = INF
        a_trainset = [self._digitize_feats(a_trainset)[0]]

        # debug functions
        in_len = self.CHAR_INDICES.shape[0]
        # input to convolutional layer
        conv_in = self.EMB[self.CHAR_INDICES].reshape((1, 1, in_len, self.vdim))
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
        conv_max_out = TT.concatenate([conv2_max_out, conv3_max_out, \
                                           conv4_max_out], axis = 1)
        emb2conv = theano.function([self.CHAR_INDICES], conv_in, name = "emb2conv")
        emb2conv2_out = theano.function([self.CHAR_INDICES], conv2_out, name = "emb2conv2_out")
        emb2conv2_max_out = theano.function([self.CHAR_INDICES], conv2_max_out, name = "emb2conv2_max_out")
        emb2conv3_out = theano.function([self.CHAR_INDICES], conv3_out, name = "emb2conv3_out")
        emb2conv3_max_out = theano.function([self.CHAR_INDICES], conv3_max_out, name = "emb2conv3_max_out")
        emb2conv4_out = theano.function([self.CHAR_INDICES], conv4_out, name = "emb2conv4_out")
        emb2conv4_max_out = theano.function([self.CHAR_INDICES], conv4_max_out, name = "emb2conv4_max_out")
        emb2conv_max_out = theano.function([self.CHAR_INDICES], conv_max_out, name = "emb2conv4_max_out")

        print("self.CONV2 = ", repr(self.CONV2.get_value()))
        print("self.CONV2_BIAS = ", repr(self.CONV2_BIAS.get_value()))
        print("self.CONV3 = ", repr(self.CONV3.get_value()))
        print("self.CONV2_BIAS = ", repr(self.CONV3_BIAS.get_value()))
        print("self.CONV4 = ", repr(self.CONV4.get_value()))
        print("self.CONV2_BIAS = ", repr(self.CONV4_BIAS.get_value()))
        for i in xrange(MAX_ITERS):
            icost = 0.
            for x_i, y_i in a_trainset:
                x_i = x_i[:3]
                print("x_i = ", repr(x_i))
                print("y_i = ", repr(y_i))
                sys.exit(66)
                # icost += train(x_i, y_i, ALPHA)
                icost += f_grad_shared(x_i, y_i)
                f_update()
            print("Iteration #{:d}: cost = {:.10f}".format(i, icost), file = sys.stderr)
            if prev_cost != INF and 0 < (prev_cost - icost) < EPSILON:
                break
            prev_cost = icost
        return icost

    def predict(self, a_seq):
        """Prediction function

        @param a_seq - input sequence whose class should be predicted

        @return 2-tuple with predicted label and its assigned score

        """
        if self._predict is None:
            y_pred = TT.argmax(self.Y, axis = 1)
            # prediction function
            self._predict = theano.function([self.W_INDICES], \
                                            [y_pred, self.Y[0, y_pred]], \
                                            name = "predict")
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
            # print("iword = ", repr(iword))
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
        self.n_conv2 = 12 # number of filters
        self.conv2_width = 2 # width of stride
        self.CONV2 = theano.shared(value = RND_VEC((self.n_conv2, 1, self.conv2_width, self.vdim)), \
                                       name = "CONV2")
        self.CONV2_BIAS = theano.shared(value = RND_VEC((1, self.n_conv2)), name = "CONV2_BIAS")
        # four convolutional filters for strides of width 3
        self.n_conv3 = 16 # number of filters
        self.conv3_width = 3 # width of stride
        self.CONV3 = theano.shared(value = RND_VEC((self.n_conv3, 1, self.conv3_width, self.vdim)), \
                                       name = "CONV3")
        self.CONV3_BIAS = theano.shared(value = RND_VEC((1, self.n_conv3)), name = "CONV3_BIAS")
        # five convolutional filters for strides of width 4
        self.n_conv4 = 20 # number of filters
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

        self.LSTM_BIAS = theano.shared(RND_VEC((self.n_lstm * 4)), name = "LSTM_BIAS")
        self._params += [self.LSTM_W, self.LSTM_U, self.LSTM_BIAS]

    def _init_emb(self):
        """Initialize embeddings.

        @return \c void

        """
        self.EMB = theano.shared(value = RND_VEC((self.V, self.vdim)), name = "EMB")
        # obtain indices for special embeddings (BEGINNING, END, UNKNOWN)
        cnt = 0
        for ikey in AUX_VEC_KEYS:
            self.feat2idx[ikey] = cnt
            cnt += 1
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
        conv_max_out = TT.concatenate([conv2_max_out, conv3_max_out, \
                                           conv4_max_out], axis = 1)
        return TT.dot(conv_max_out[0,:], self.LSTM_W  + self.LSTM_BIAS)

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
            # obtain character convolutions for input indices
            iconv = self._emb2conv(x_)
            # compute common term for all LSTM components
            proxy = TT.nnet.sigmoid(iconv + TT.dot(o_, self.LSTM_U))
            # input
            i = proxy[:self.n_lstm]
            # forget
            f = proxy[self.n_lstm:2*self.n_lstm]
            # output
            o = proxy[2*self.n_lstm:3*self.n_lstm]
            # new state of memory cell (input * current + forget * previous)
            m = i * TT.tanh(proxy[3*self.n_lstm:]) + f * m_
            # new outout state
            o = o * TT.tanh(m)
            # return new output and memory state
            return o, m
        # `scan' function
        res, _ = theano.scan(_lstm_step,
                             sequences = [self.W_INDICES],
                             outputs_info = [np.zeros(self.n_lstm).astype(config.floatX),
                                             np.zeros(self.n_lstm).astype(config.floatX)],
                                   name = "_lstm_layers")
        return res[0]

    def _reset(self):
        """Reset instance variables.

        @return \c void

        """
        self._predict = None
