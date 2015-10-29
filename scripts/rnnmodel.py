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

##################################################################
# Variables and Constants

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
        self.class2int = dict()
        self.int2class = dict()
        self.feat2vec = dict()

    def fit(self, a_trainset):
        """Train RNN model on the training set.

        @param a_trainset - trainig set as a list of 2-tuples with
                            training instances and classes

        @return \c void

        """
        pass
