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
import theano.Tensor as T

##################################################################
# Variables and Constants

# initial values for normal distribution
MU = 0.
SIGMA = 0.5

# dimension of input vectors
VEC_DIM = (1, 20)

##################################################################
# Class
class RNNModel(object):
    """Wrapper class around an RNN classifier.

    Instance variables:
    class2int - mapping from string classes to integers
    class2int - reverse mapping from integers to string classes
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
        self.feat2vec = dict()

    def fit(self, a_trainset):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes

        @return \c void

        """
        print("original trainset =", repr(a_trainset[0]), file = sys.stderr)
        a_trainset = self._digitize_feats(a_trainset)
        print("digitized trainset =", repr(a_trainset[0]), file = sys.stderr)
        pass

    def _digitize_feats(self, a_trainset):
        """Convert features and target classes to vectors and ints.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes
        @param a_feat_size - dimension of feature representation

        @return new list of digitized features and classes

        """
        ret = []
        ditems = None
        for iseq, ilabel in a_trainset:
            ditems = []
            # digitize label
            if ilabel is not None:
                if ilabel not in self.lbl2int:
                    try:
                        assert int(ilabel) not in self.int2lbl, \
                            "Integral label {:s} already exists in the label map".format(ilabel)
                        self.lbl2int[ilabel] = int(ilabel)
                    except (AssertionError, ValueError):
                        self.lbl2int[ilabel] = self.nlabels
                    self.nlabels += 1
                dlabel = self.lbl2int[ilabel]
                self.int2lbl[dlabel] = ilabel
            for iitem in iseq:
                if iitem not in self.feat2vec:
                    self.feat2vec[iitem] = np.random.normal(MU, SIGMA, VEC_DIM)
                ditems.append(self.feat2vec[iitem])
            ret.append((ditems, dlabel))
        return ret
