#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Module for predicting polarity classes of tweets.

The input file is assumed to be in TSV format where the second field
(starting with 0) is assumed to be the gold label and the last field
contains the text of the messages.

USAGE:
classify.py [OPTIONS] [input_file]

OPTIONS:
-h|--help  print this screen and exit
-v|--verbose  output errors along with evaluation

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from classifier import SentimentClassifier
from evaluate import evaluate, get_fields, GLD_IDX, TXT_IDX

import argparse
import sys

##################################################################
# Variables and Constants
TRAIN = "train"
TEST = "test"
EVALUATE = "evaluate"

##################################################################
# Methods
def _iterlines(a_file):
    """Iterate over input lines of a file

    @param a_file - input file to iterate over

    @return iterator over resulting TSV fields

    """
    ifields = None
    for iline in a_file:
        ifields = get_fields(iline)
        # compute prediction and append it to the list of fields
        ifields.append(a_classifier.predict(ifields[TXT_IDX]))
        yield ifields

def main():
    """Classify tweets according to their sentiment polarity

    @return \c 0 on success, non-0 other
    """
    # process CLI arguments
    argparser = argparse.ArgumentParser(description = """Script for classifying
tweets according to their sentiment polarity""")

    subparsers = argparser.add_subparsers(help="type of operation to perform", dest = "mode")
    # training options
    tr_parser = subparsers.add_parser(TRAIN, help = "train the model")
    # testing options
    tr_parser = subparsers.add_parser(TEST, help = "test the model")
    # evaluation options (train and test at the same time)
    ev_parser = subparsers.add_parser(EVALUATE, help = "evaluate trained model")
    ev_parser.add_argument("-v", "--verbose", help = "output errors along with evaluation",
                           action = "store_true")
    # common options
    argparser.add_argument("-m", "--model", help = "path to the (stored or to be stored) model", \
                               type = str)
    argparser.add_argument("files", help = "input files in TSV format", \
                           type = argparse.FileType('r'), nargs = '*', default = [sys.stdin])
    args = argparser.parse_args()
    # perform the requied action
    if args.mode == EVALUATE:
        raise NotImplementedError
        # classifier = pickle.load(args.model)
        # macro_MAE = micro_MAE = 0.
        # for ifile in a_files:
        #     macro_MAE, micro_MAE = evaluate(classify(classifier, ifile), args.verbose, lambda x: x)
        #     print("{:20s}{:.7}".format("Macro-averaged MAE:", macro_MAE), file = sys.stderr)
        #     print("{:20s}{:.7}".format("Micro-averaged MAE:", micro_MAE), file = sys.stderr)
    elif args.mode == TRAIN:
        classifier = SentimentClassifier(a_path = None)
        classifier.train([(ifields[TXT_IDX], ifields[GLD_IDX]) for ifile in args.files \
                              for ifields in _iterlines(ifile)], a_path = args.model)
    return 0

##################################################################
# Main
if __name__ == "__main__":
    main()
