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
import theano.tensor as TT
import theano

##################################################################
# Variables and Constants

# initial parameters for uniform distribution
UMIN = -1.5
UMAX = 1.5

# initial parameters for normal distribution
MU = 0.
SIGMA = 1.5

# dimension of input vectors
VEC_DIM = (1, 20)

# custom function for generating random vectors
np.random.seed()
RND_VEC = lambda a_dim = VEC_DIM: np.random.uniform(UMIN, UMAX, a_dim)

# symbolic code for missing vectors
UNK = "___%UNK%___"

##################################################################
# Class
class RNNModel(object):
    """Wrapper class around an RNN classifier.

    Instance variables:
    lbl2int - mapping from string classes to integers
    int2lbl - reverse mapping from integers to string classes
    int2coeff - mapping from label integer to its coefficient
    feat2vec - mapping from string features to their learned
                representations

    Methods:
    train - extract features and adjust parameters of the model
    _reset - clear instance variables

    """

    def __init__(self):
        """Class constructor.

        """
        self.nlabels = 0
        self.lbl2int = dict()
        self.int2lbl = dict()
        self.int2coeff = dict()
        self.feat2vec = dict()
        # declare Theano symbolic variables
        # word compositionality tensor (2d x 2d x d)
        self.W = theano.shared(np.random.randn(2*VEC_DIM[-1], 2*VEC_DIM[-1], VEC_DIM[-1]))
        # bias tensor (2d x d)
        self.V = theano.shared(np.random.randn(2*VEC_DIM[-1], VEC_DIM[-1]))

    def fit(self, a_trainset):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes

        @return \c void

        """
        print("original trainset =", repr(a_trainset[0]), file = sys.stderr)
        a_trainset = self._digitize_feats(a_trainset)
        print("digitized trainset =", repr(a_trainset[0]), file = sys.stderr)
        # compile training function
        # train = theano.function(inputs=[x,y], outputs=[prediction, xent], \
        #                         updates=[(w, w-0.01*gw), (b, b-0.01*gb)], name = "train")
        pass

    def _digitize_feats(self, a_trainset):
        """Convert features and target classes to vectors and ints.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
        @param a_feat_size - dimension of feature representation

        @return new list of digitized features and classes

        """
        ret = []
        ditems = None; coeff = 1.
        # create a vector for unknown words
        self.feat2vec[UNK] = RND_VEC()
        for iseq, ilabel in a_trainset:
            ditems = []
            # digitize label
            if ilabel is not None:
                if ilabel not in self.lbl2int:
                    try:
                        coeff = int(ilabel)
                    except (AssertionError, ValueError):
                        pass
                    self.lbl2int[ilabel] = self.nlabels
                    self.int2lbl[self.nlabels] = ilabel
                    self.int2coeff[self.nlabels] = abs(coeff) + 1e-5
                    self.nlabels += 1
                dlabel = self.lbl2int[ilabel]
            # convert features to vectors
            for iitem in iseq:
                if iitem not in self.feat2vec:
                    self.feat2vec[iitem] = RND_VEC()
                ditems.append(self.feat2vec[iitem])
            ret.append((ditems, dlabel))
        # convert labels to vectors
        lvec = None
        for i, (feats, lbl) in enumerate(ret):
            lvec = np.zeros(self.nlabels); lvec[lbl] = 1.
            ret[i] = (feats, lvec)
        return ret
