#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

"""Script for predicting polarity classes of tweets.

The input file is assumed to be in TSV format where the second field
(starting with 0) is assumed to be the gold label and the last field
contains the text of the messages.

USAGE:
twitter_sentiment [MODE] [OPTIONS] [input_file]

@author = Wladimir Sidorenko (Uladzimir Sidarenka)
@mail = <sidarenk at uni dash potsdam dot de>
@version = 0.0.1

"""

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from sentiment_classifier import SentimentClassifier
from evaluate import evaluate, get_fields, GLD_IDX, TXT_IDX, TOPIC_IDX
from twokenize import tokenize

import argparse
import codecs
import os
import re
import sys

##################################################################
# Variables and Constants
TRAIN = "train"
TEST = "test"
EVALUATE = "evaluate"
ENCODING = "utf-8"

TAB = "\t"
TAB_RE = re.compile(TAB)
NONMATCH_RE = re.compile("(?!)")
POS_CHAR = ''
NEG_CHAR = ''
TOPIC2RE = {}
POS_RE = re.compile(r"[\b\A](amazing|better|best|cool||fun|fant|good|great|like|love|luv|wow|wonder\b|:\))")

# text normalization
STOP_WORDS_RE = re.compile(r"(?:\b|\A)(i|the(?:re|ir|[my])?|(?:any|every)body|here|from|an?|i[stn]|[a']re|am|was|we(?:re)?|ha[ds]|have|to|with|will|all|each|wh(?:en|ich|ere|o|at)(?:ever)?|makes?|making|may|might|i(?:'?m)?|that|we\'?re|\'?s|through|by|(?:get|put|say|set)(?:ting|s)?|be(?:en|ing)?|this|as|going|do(?:es)?|did|tomorrow|week|today|monday|tuesday|thursday|wednesday|friday|saturday|yesterday|tonight|sunday|gives?|about|[bmw]e|y?ou(?:rs?)?|u|some(?:thing|one)?|and|o[fr]|for|\'?(?:ve|d)|us|s?he|on|(?:some)?where|so|[ia]t|it\'?s|gonna|cc|\'?ll|may|take[ns]?|took|how|let|her?|she|it|hi[ms]|my|(?:an)?other|g[eo]ts?|while|can|either|[oi]nto|via|if|hi|one|october|january|february|march|april|may|june|july|august|september|october|november|december|nov|aug|feb|sept|two)(?:[.%$\"'`])?(?=\s|[?!]|\Z)")
PUNCT_RE = re.compile("(\s|\A|\w)(?:-+|,+|:+|[.%$\"'`]+)(\s|\Z)")
NEG_RE = re.compile("(?:\s|\A)(?:has|have|is|was|do|does|need|ca|wo|would)(n'?t)(?:\s|\Z)")
AMP_RE = re.compile(r"&amp;")
GT_RE = re.compile(r"&gt;")
LT_RE = re.compile(r"&lt;")
DIGIT_RE = re.compile(r"\b[\d:]+(?:[ap]\.?m\.?|th|rd|nd|m|b)?\b")
HTTP_RE = re.compile(r"(?:https?://\S*|\b(?:(?:[\w]{3,5}://?|(?:www|bit)[.]|(?:\w[-\w]+[.])+(?:a(?:ero|sia|[c-gil-oq-uwxz])|b(?:iz|[abd-jmnorstvwyz])|c(?:at|o(?:m|op)|[acdf-ik-orsuvxyz])|d[dejkmoz]|e(?:du|[ceghtu])|f[ijkmor]|g(?:ov|[abd-ilmnp-uwy])|h[kmnrtu]|i(?:n(?:fo|t)|[del-oq-t])|j(?:obs|[emop])|k[eghimnprwyz]|l[abcikr-vy]|m(?:il|obi|useum|[acdeghk-z])|n(?:ame|et|[acefgilopruz])|o(?:m|rg)|p(?:ro|[ae-hk-nrstwy])|qa|r[eosuw]|s[a-eg-or-vxyz]|t(?:(?:rav)?el|[cdfghj-pr])|xxx)\b)(?:[^\s,.:;]|\.\w)*))")
AT_RE = re.compile(r"(RT\s+)?[.]?@\S+")
SPACE_RE = re.compile(r"\s\s+")

# HSAN dictionary
HSAN = "HashtagSentimentAffLexNegLex"
HSAN_W_IDX = 0                # index of a polar term
HSAN_SCORE_IDX = 1              # index of the score
HSAN_THRSHLD = 4.               # minimum required absolute score
HSAN_NEG_RE = re.compile(r"(.*)(_neg(?:first)?)")

# Sentiment 140
S140 = "Sentiment140-Lexicon-v0.1"
S140_W_IDX = 0                # index of a polar term
S140_SCORE_IDX = 1              # index of the score
S140_THRSHLD = 3.               # minimum required absolute score
S140_NEG_RE = HSAN_NEG_RE

# NRC Hashtag
NRC_HSHTAG = "NRC-Hashtag-Sentiment-Lexicon-v0.1"
NRC_HSHTAG_W_IDX = 0            # index of a polar term
NRC_HSHTAG_SCORE_IDX = 1        # index of the score
NRC_POSITIVE = "positive"
NRC_NEGATIVE = "negative"

##################################################################
# Methods
def _add_cmn_options(a_parser):
    """Add common options to option subparser

    @param a_parser - option subparser

    @return \c void

    """
    a_parser.add_argument("-m", "--model", help = "path to the (stored or to be stored) model", \
                              type = str)
    a_parser.add_argument("files", help = "input files in TSV format", \
                              type = argparse.FileType('r'), nargs = '*', default = [sys.stdin])

def _safe_sub(a_re, a_sub, a_line):
    """Perform substitution on line and return it unchanged if line gets empty

    Args:
    -----
      a_re - regular expression to substitue
      a_sub - substitution
      a_line - line where the substitution should be done

    Returns:
    --------
      (str): line with substitutions or unchanged line if it got empty

    """
    return a_re.sub(a_sub, a_line).strip() or a_line

def _cleanse(a_line, a_topic = ""):
    """Remove spurious elements from line.

    @param a_line - line to be cleansed
    @param (optional) a_topic - topic which the given line pertains to

    @return cleansed line

    """
    global TOPIC2RE
    line = _safe_sub(HTTP_RE, "", a_line.lower())
    line = _safe_sub(AT_RE, "", line)
    line = _safe_sub(DIGIT_RE, "", line)
    line = _safe_sub(AMP_RE, "", line)
    line = _safe_sub(AMP_RE, "", line)
    line = _safe_sub(STOP_WORDS_RE, " ", line)
    if a_topic:
        a_topic = a_topic.lower()
        if a_topic not in TOPIC2RE:
            TOPIC2RE[a_topic] = re.compile(r"\b" + SPACE_RE.sub(r"\s+", re.escape(a_topic)) + r"\b")
        line = _safe_sub(TOPIC2RE[a_topic], "", line)
    line = _safe_sub(PUNCT_RE, "\\1\\2", line)
    line = _safe_sub(SPACE_RE, ' ', line)
    # print("modified line =", repr(line), file = sys.stderr)
    # line = GT_RE.sub(">", LT_RE.sub("<", line))
    # line = NEG_RE.sub(" not ", line)
    # line = POS_RE.sub(POS_CHAR + "\\1", line)
    # line = NEG_RE.sub(NEG_CHAR + "\\1", line)
    return line

def _iterlines(a_file):
    """Iterate over input lines of a file

    @param a_file - input file to iterate over

    @return iterator over resulting TSV fields

    """
    ifields = []
    for iline in a_file:
        iline = iline.decode(ENCODING)
        # compute prediction and append it to the list of fields
        ifields = get_fields(iline)
        ifields[TXT_IDX] = _cleanse(ifields[TXT_IDX], ifields[TOPIC_IDX])
        yield ifields

def _merge_lexica(a_lexica):
    """Read sentiment terms from lexicon file.

    Args:
    -----
      a_dname (tuple of 2 sets):
        list of positive and negative polar term sets

    Returns:
    --------
      (4-tuple): merges sets of positive and negative terms with their regexps

    """
    pos = set(); neg = set()
    for ipos, ineg in a_lexica:
        pos |= ipos
        neg |= ineg
        # if not pos:
        #     pos |= ipos
        # else:
        #     pos &= ipos
        # if not neg:
        #     neg |= ineg
        # else:
        #     neg &= ineg

    if pos:
        pos_re = re.compile(r"\b(" + "|".join([re.escape(t) for t in pos]) + r")(?=\s|\Z)")
    else:
        pos_re = NONMATCH_RE

    if neg:
        neg_re = re.compile(r"\b(" + "|".join([re.escape(t) for t in neg]) + r")(?=\s|\Z)")
    else:
        neg_re = NONMATCH_RE
    return (pos, pos_re, neg, neg_re)


def _read_lexicon(a_dname):
    """Read sentiment terms from lexicon file.

    Args:
    -----
      a_dname (str): path to the directory with lexicon files

    Returns:
    --------
      (2-tuple): sets of positive and negative terms

    """
    if not a_dname:
        return
    elif a_dname[-1] == '/':
        a_dname = os.path.dirname(a_dname)
    basename = os.path.basename(a_dname)
    if basename == HSAN:
        return _read_hsan(a_dname)
    elif basename == S140:
        return _read_s140(a_dname)
    elif basename == NRC_HSHTAG:
        return _read_nrc_hshtag(a_dname)
    else:
        raise Exception("Unknown dictionary format: '{:s}'".format(basename))

def _read_hsan(a_dname):
    """Read HashtagSentimentAffLexNegLex sentiment lexicon.

    Args:
    -----
      a_dname (str): path to the directory with lexicon files

    Returns:
    --------
      (2-tuple): sets of positive and negative terms

    """
    score = 0.
    term = ""
    fields = None
    pos = set(); neg = set()
    print("Reading HashtagSentimentAffLexNegLex... ", end = "", file = sys.stderr)
    for fname in ["HS-AFFLEX-NEGLEX-unigrams.txt"]: # skip bigrams for the time being
        fname = os.path.join(a_dname, fname)
        with codecs.open(fname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline:
                    continue
                fields = TAB_RE.split(iline)
                score = float(fields[HSAN_SCORE_IDX])
                if abs(score) > HSAN_THRSHLD:
                    term = HSAN_NEG_RE.sub("not \1", _cleanse(fields[HSAN_W_IDX]))
                    if len(term) < 3:
                        continue
                    if score > 0.:
                        pos.add(term)
                    else:
                        neg.add(term)
    print("done", file = sys.stderr)
    return (pos, neg)

def _read_s140(a_dname):
    """Read Sentiment140-Lexicon-v0.1 sentiment lexicon.

    Args:
    -----
      a_dname (str): path to the directory with lexicon files

    Returns:
    --------
      (2-tuple): sets of positive and negative terms

    """
    score = 0.
    term = ""
    fields = None
    pos = set(); neg = set()
    print("Reading Sentiment140-Lexicon-v0.1... ", end = "", file = sys.stderr)
    for fname in ["unigrams-pmilexicon.txt"]: # skip bigrams for the time being
        fname = os.path.join(a_dname, fname)
        with codecs.open(fname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline:
                    continue
                fields = TAB_RE.split(iline)
                score = float(fields[S140_SCORE_IDX])
                if abs(score) > S140_THRSHLD:
                    term = S140_NEG_RE.sub("not \1", _cleanse(fields[S140_W_IDX]))
                    if len(term) < 3:
                        continue
                    if score > 0.:
                        pos.add(term)
                    else:
                        neg.add(term)
    print("done", file = sys.stderr)
    return (pos, neg)

def _read_nrc_hshtag(a_dname):
    """Read NRC-Hashtag-Sentiment-Lexicon-v0.1 sentiment lexicon.

    Args:
    -----
      a_dname (str): path to the directory with lexicon files

    Returns:
    --------
      (2-tuple): sets of positive and negative terms

    """
    fields = None
    term = tclass = ""
    pos = set(); neg = set()
    print("Reading NRC-Hashtag-Sentiment-Lexicon-v0.1... ", end = "", file = sys.stderr)
    for fname in ["sentimenthashtags.txt"]: # skip bigrams for the time being
        fname = os.path.join(a_dname, fname)
        with codecs.open(fname, 'r', ENCODING) as ifile:
            for iline in ifile:
                iline = iline.strip()
                if not iline:
                    continue
                term, tclass = TAB_RE.split(iline)
                term = _cleanse(term)
                if not term:
                    continue
                if tclass == NRC_POSITIVE:
                    pos.add(term)
                elif tclass == NRC_NEGATIVE:
                    neg.add(term)
    print("done", file = sys.stderr)
    return (pos, neg)

def _read_dataset(a_files):
    """Read data set into a list of two-tuples with input items and gold classes.

    @param a_files - input files containing training data

    @return list of 2-tuples with input items and their gold classes

    """
    return [(list(ifields[TXT_IDX]), ifields[GLD_IDX]) \
            for ifile in a_files for ifields in _iterlines(ifile)]

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
    tr_parser.add_argument("-d", "--dev-set", help = "development set",
                             type = argparse.FileType('r'))
    tr_parser.add_argument("-l", "--lexicon", help = "sentiment lexicon to use for sampling",
                           type = str, action = "append", default = [])
    _add_cmn_options(tr_parser)
    # testing options
    test_parser = subparsers.add_parser(TEST, help = "test the model")
    test_parser.add_argument("-d", "--debug", help = "output debug information", \
                             action = "store_true")
    test_parser.add_argument("-v", "--verbose", help = "output scores along with predicted labels",
                             action = "store_true")
    _add_cmn_options(test_parser)
    # evaluation options (train and test at the same time)
    ev_parser = subparsers.add_parser(EVALUATE, help = "evaluate trained model")
    _add_cmn_options(ev_parser)
    ev_parser.add_argument("-v", "--verbose", help = "output errors along with evaluation",
                           action = "store_true")
    args = argparser.parse_args()
    # perform the requied action
    if args.mode == TRAIN:
        classifier = SentimentClassifier(a_path = None)
        if args.dev_set is None:
            dev_set = None
        else:
            dev_set = _read_dataset([args.dev_set])
            lexica = [_read_lexicon(ilex) for ilex in args.lexicon]
            pos, pos_re, neg, neg_re = _merge_lexica(lexica)
        classifier.train(_read_dataset(args.files), a_path = args.model, \
                         a_dev_set = dev_set, a_pos = (pos_re, pos), \
                         a_neg = (neg_re, neg))
    elif args.mode == TEST:
        # load model from default location
        y = ""; score = 0.
        if args.model:
            classifier = SentimentClassifier(args.model)
        else:
            classifier = SentimentClassifier()
        for ifile in args.files:
            for ifields in _iterlines(ifile):
                if args.debug:
                    classifier.debug(list(ifields[TXT_IDX]))
                else:
                    y, score = classifier.predict(list(ifields[TXT_IDX]))
                    if args.verbose:
                        ifields.append(str(score))
                    ifields.append(y)
                    print(TAB.join(ifields))
    else:
        raise NotImplementedError
        # for ifile in a_files:
        #     macro_MAE, micro_MAE = evaluate(classify(classifier, ifile), args.verbose, lambda x: x)
        #     print("{:20s}{:.7}".format("Macro-averaged MAE:", macro_MAE), file = sys.stderr)
        #     print("{:20s}{:.7}".format("Micro-averaged MAE:", micro_MAE), file = sys.stderr)
    return 0

##################################################################
# Main
if __name__ == "__main__":
    main()
