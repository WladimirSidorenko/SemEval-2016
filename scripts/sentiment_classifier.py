#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module defining classifier class for predicting sentiment.

Constants:
DFLT_MODEL_PATH - default path to pre-trained model

Classes:
SentimentClassifier - classifier class for predicting sentiments

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from rnnmodel import RNNModel
# cPickle can't serialize function objects, e.g. rnn._predict()
# from cloud.serialization.cloudpickle import dump, load
from cPickle import dump, load

import os
import sys

##################################################################
# Variables and Constants
DFLT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models",
                               "sentiment.model")


##################################################################
# Class
class SentimentClassifier(object):
    """Class for sentiment classification of tweets.

    Instance variables:
    model - (pre-trained) model

    Instance methods:
    train - train model on training set

    """

    def __init__(self, a_path=DFLT_MODEL_PATH):
        """Class constructor.

        @param a_path - path to pre-trained model (or None if no model exists)
        """
        if a_path is None:
            self.model = None
            self.predict = self._invalid_func
            self.debug = self._invalid_func
        else:
            if not os.path.isfile(a_path) or not os.access(a_path, os.R_OK):
                raise RuntimeError(
                    "Can't create model from file {:s}".format(a_path))
            with open(a_path, "rb") as ifile:
                self.model = load(ifile)
            self.predict = self._predict
            self.debug = self._debug

    def train(self, a_train_set, a_path=DFLT_MODEL_PATH,
              a_dev_set=None, **a_kwargs):
        """Train model and store it at specified path.

        @param a_train_set - training set with examples and classes
        @param a_path - training set with examples and classes
        @param a_dev_set - development set with examples and classes
        @param a_kwargs - additional keyword arguments

        @return \c void

        """
        if not a_path:
            a_path = DFLT_MODEL_PATH

        a_path = os.path.abspath(a_path)
        if (os.path.exists(a_path) and not os.access(a_path, os.W_OK)) or \
                (not os.path.exists(a_path) and
                 not os.access(os.path.dirname(a_path), os.W_OK)):
            raise RuntimeError(
                "Can't create model at specified path: '{:s}'".format(a_path))
        # create and train an RNN model
        imodel = RNNModel()
        imodel.fit(a_train_set, a_path, a_dev_set, **a_kwargs)
        # remember model as instance attribute
        self.model = imodel
        # self.predict = self._predict

    def _predict(self, a_inst):
        """Predict label of a new instance.

        @param a_inst - instance whose label should be predicted

        @return 2-tuple with predicted symbolic label and its score

        """
        return self.model.predict(a_inst)

    def _debug(self, a_inst):
        """Predict label of a new instance.

        @param a_inst - instance whose label should be predicted

        @return 2-tuple with predicted symbolic label and its score

        """
        return self.model.debug(a_inst)

    def _invalid_func(self, a_inst):
        """Invalid prediction function

        @param a_inst - input sequence whose class should be predicted

        @raise RuntimeError

        """
        raise RuntimeError("Model is not trained.")

