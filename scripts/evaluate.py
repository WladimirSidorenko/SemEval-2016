#!/usr/bin/env python

"""Evaluation script for estimating mean absolute error (MAE).

The input file is assumed to be in TSV format where the second field
(starting with 0) is assumed to be the gold label and the last field
contains the predicted label.

See http://alt.qcri.org/semeval2016/task4/data/uploads/eval.pdf for
further details regarding computation of macro- and micro-averaged
MAEs.

USAGE:
evaluate.py [OPTIONS] [input_file]

OPTIONS:
-h|--help  print this screen and exit
-v|--verbose  output errors

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from collections import defaultdict
import argparse
import re
import sys

##################################################################
# Variables and Constants
MIN_FIELDS = 5                  # minimum number of fields in input

CLASSES = set(xrange(-2, 3))
ENCODING = "utf-8"

COMMENT_RE = re.compile(r"\A\s*#.*\Z")
TAB_RE = re.compile(r"\t+")

TOTAL_IDX = 0
DIFF_IDX = 1

TXT_IDX = -2

##################################################################
# Methods
def evaluate(a_ifile, a_verbose = False):
    """Estimating mean absolute error on single input file

    @param a_ifile - iterable over lines
    @param a_verbose - boolean flag indiciating whether errors should
                     be printed as well

    @return 2-tuple with macro- and micro-averaged MAE

    """
    macro_MAE = micro_MAE = 0.

    ifields = None
    gold = pred = -1
    cstat = defaultdict(lambda: [0, 0])
    for iline in a_ifile:
        iline = COMMENT_RE.sub("", iline).strip()
        if not iline:
            continue
        ifields = TAB_RE.split(iline)
        if len(ifields) < MIN_FIELDS:
            print("WARNING: Unrecognized line format: '{:s}'".format(iline), file = sys.stderr)
            continue
        # obtain labels
        gold, pred = int(ifields[2]), int(ifields[-1])
        assert gold in CLASSES, "Unrecognized gold label: {:d}".format(gold)
        assert pred in CLASSES, "Unrecognized predicted label: {:d}".format(pred)
        # output error
        if a_verbose and gold != pred:
            print("{:d} confused with {:d} in message '{:s}'".format(gold, pred, ifields[TXT_IDX]))
        # update statistics
        cstat[gold][TOTAL_IDX] += 1
        cstat[gold][DIFF_IDX] += abs(gold - pred)
    # estimate MAE scores
    macro_MAE = sum([float(dff) / (float(ttl) or 1.) for ttl, dff in cstat.itervalues()])
    macro_MAE /= float(len(CLASSES)) or 1.
    micro_MAE = float(sum([dff for _, dff in cstat.itervalues()]))
    micro_MAE /= float(sum([ttl for ttl, _ in cstat.itervalues()])) or 1.
    return (macro_MAE, micro_MAE)

def main():
    """Main method for estimating mean absolute error on input files.

    @return 0 on success, non-0 otherwise

    """
    # process CLI arguments
    argparser = argparse.ArgumentParser(description = """Evaluation script
 for estimating mean absolute error of predictions.""")
    argparser.add_argument("-v", "--verbose", help = "output errors",
                           action = "store_true")
    argparser.add_argument("files", help = """input files with predictions
in TSV format""", type = argparse.FileType('r'), nargs = '*', default = [sys.stdin])
    args = argparser.parse_args()

    # estimate MAE on each file
    macro_MAE = micro_MAE = 0.
    for ifile in args.files:
        macro_MAE, micro_MAE = evaluate(ifile, args.verbose)
        print("{:20s}{:.7}".format("Macro-averaged MAE:", macro_MAE), file = sys.stderr)
        print("{:20s}{:.7}".format("Micro-averaged MAE:", micro_MAE), file = sys.stderr)

##################################################################
# Main
if __name__ == "__main__":
    main()
