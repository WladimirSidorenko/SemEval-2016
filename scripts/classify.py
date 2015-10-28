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
-e|--evaluate  only evaluate the model
-v|--verbose  output errors along with evaluation

"""

##################################################################
# Imports
from evaluate import evaluate, get_fields, TXT_IDX

##################################################################
# Variables and Constants
TRAIN = "train"
EVALUATE = "evaluate"

##################################################################
# Methods
def classify(a_classifier, a_file):
    """Classify tweets into five polarity classes [-2..2]

    @param a_classifier - classifier to use
    @param a_file - input file to process

    @return iterator over resulting fields

    """
    ifields = None
    for iline in ifile:
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
    tr_parser = argparser.add_subparser(TRAIN, help = "train model")

    ev_parser = argparser.add_subparser(EVALUATE, help = "evaluate trained model")
    ev_parser.add_argument("-v", "--verbose", help = "output errors along with evaluation",
                           action = "store_true")
    argparser.add_argument("-m", help = "path to (stored or to be stored) model", type = str)
    argparser.add_argument("files", help = """input files in TSV format""", \
                           type = argparse.FileType('r'), nargs = '*', default = [sys.stdin])
    args = argparser.parse_args()
    # perform the requied action
    if getattr(args, EVALUATE):
        classifier = pickle.load(args.model)
        macro_MAE = micro_MAE = 0.
        for ifile in a_files:
            macro_MAE, micro_MAE = evaluate(classify(classifier, ifile), args.verbose, lambda x: x)
            print("{:20s}{:.7}".format("Macro-averaged MAE:", macro_MAE), file = sys.stderr)
            print("{:20s}{:.7}".format("Micro-averaged MAE:", micro_MAE), file = sys.stderr)
    else:
        classifier = Classifier(args.model)
        train
    return 0

##################################################################
# Main
if __name__ == "__main__":
    main()
