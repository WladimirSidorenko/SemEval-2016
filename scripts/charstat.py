#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Imports
from __future__ import unicode_literals, print_function

from evaluate import GLD_IDX, TXT_IDX
from preprocessing import iterlines

from collections import defaultdict, Counter
from math import sqrt
import argparse
import sys

##################################################################
# Variables and Constants
DFLT_N = 1

# indices used in class_char_cnt
TTL_MSG_IDX = 0
TTL_CHAR_IDX = 1
STAT_IDX = 2

# keys and indices used in char_stat

# class of total occurrences
TOTAL = 1

# binomial probability and variance of classes in corpus
CLS_PROB_IDX = 0
CLS_VAR_IDX = 1

# indices of global character statistics
# binomial probability and variance of a char sequence in corpus
CNT_CHAR_IDX = 0
PROB_CHAR_IDX = 1
VAR_CHAR_IDX = 2
# binomial probability and variance of a char sequence in a message
CNT_MSG_IDX = 3
PROB_MSG_IDX = 4
VAR_MSG_IDX = 5
# expectation and variance of the number of occurrences of char sequences in
# message
MEAN_N_IDX = 6
VAR_N_IDX = 7
MAX_CHAR_STAT_IDX = 8

# indices of class-specific character statistics
EXY_CHAR_IDX = 0
EXY_MSG_IDX = 1
EXY_NMSG_IDX = 2
MAX_CHAR_CLS_IDX = 3

COV_CHAR_IDX = 0
COR_CHAR_IDX = 1
COV_MSG_IDX = 2
COR_MSG_IDX = 3
COV_NMSG_IDX = 4
COR_NMSG_IDX = 5
COV_COR_MAX_IDX = 6


##################################################################
# Methods
def _iterchars(a_str, a_n=DFLT_N):
    """Iterate over character n-grams in given string.

    Args:
    -----
      a_str: string
        string to iterate over
      a_n: length
        length of character n-grams

    Yields:
    -------
      str:
        character n-grams length a_n

    """
    for i in xrange(len(a_str) - a_n + 1):
        yield a_str[i:i+a_n]


def compute_cnt(a_it, a_n_char):
    """Compute counts of classes and characters.

    Args:
    -----
    a_it: iterator
      iterator over pairs of class labels and text
    a_n_char: int
      length of character n-grams

    Returns:
    --------
    (cls_cnt, char_cnt): (dict, dict)
      counts of classes and characters

    """
    cls_cnt = defaultdict(lambda: [0., 0.])
    char_cnt = defaultdict(lambda: defaultdict(Counter))
    iCounter = Counter()
    for icls, itext in a_it:
        # update class statistics
        cls_cnt[icls][TTL_MSG_IDX] += 1.
        cls_cnt[icls][TTL_CHAR_IDX] += len(itext) - a_n_char + 1.
        # update character statistics
        iCounter.update(_iterchars(itext, a_n_char))
        for ichar, icnt in iCounter.iteritems():
            char_cnt[ichar][icls][icnt] += 1.
        iCounter.clear()
    return (cls_cnt, char_cnt)


def compute_stat(a_cls_cnt, a_char_cnt):
    """Compute prob, mean, and variance of character sequences.

    Args:
    -----
    a_cls_cnt: defaultdict([int, int])
      statistics on classes
    a_char_cnt: defaultdict(defaultdict(Counter))
      statistics on character sequences

    Returns:
    --------
    (cls_stat, char_stat): (dict, dict)
      statistics on classes and characters

    """
    # compute total number of messages and char n-grams
    ttl_msgs = sum(item[TTL_MSG_IDX]
                   for item in a_cls_cnt.itervalues()) or 1.
    ttl_chars = sum(item[TTL_CHAR_IDX]
                    for item in a_cls_cnt.itervalues()) or 1.
    # remember all classes
    classes = a_cls_cnt.keys()
    # compute class statistics
    cls_stat = {icls: [float(istat[TTL_MSG_IDX])/ttl_msgs, 0.]
                for icls, istat in a_cls_cnt.iteritems()}
    for v in cls_stat.itervalues():
        v[CLS_VAR_IDX] = v[CLS_PROB_IDX] * (1. - v[CLS_PROB_IDX])
    # total statistics on char n-gram statistics
    char_stat = {ichar: [0.] * MAX_CHAR_STAT_IDX for ichar in a_char_cnt}
    # class-specific expectations of char n-grams
    char_cls_stat = {ichar: defaultdict(lambda: [0.] * MAX_CHAR_CLS_IDX)
                     for ichar in a_char_cnt}
    # calculate class-specific and total probability and statistics
    char_cls_s = None
    iCounter = Counter()
    for ichar, icharcnt in a_char_cnt.iteritems():
        char_cls_s = char_cls_stat[ichar]
        # gather total statistics on that character sequence and compute
        # class-specific statistics
        for icls, iclcnt in icharcnt.iteritems():
            iCounter.update(iclcnt)
            _compute_xy_stat(char_cls_s[icls], iclcnt,
                             a_cls_cnt[icls],
                             cls_stat[icls][CLS_PROB_IDX]
                             )
        # compute global character statistics
        _compute_char_stat(char_stat[ichar], iCounter, ttl_chars, ttl_msgs)
        iCounter.clear()
    return (cls_stat, char_stat, char_cls_stat)


def _compute_xy_stat(a_xy_stat, a_xy_cnt, a_y_cnt, a_p_y):
    """Compute probability, mean, and variance for single item.

    Args:
    -----
    a_xy_stat: list
      class-specific character statistics to be populated
    a_xy_cnt: list
      class-specific character counters
    a_y_cnt: list
      class-specific statistics
    a_p_y: float
      probability of the class

    Returns:
    --------
    (void)
      updates statistics in place

    """
    # estimate the total number of time that the character n-gram appeared with
    # that class, and compute expectations of the number of occurrences of that
    # char sequence in a message
    n = char_occ = e_msg_n = 0.
    ttl_msgs = float(a_y_cnt[TTL_MSG_IDX])
    for k, v in a_xy_cnt.iteritems():
        n = float(k * v)
        char_occ += n
        # E_{n_msg}[XY] = \sum_n(P_{msg}(x=n|y=1) * P(y=1))
        e_msg_n += n / ttl_msgs
    # E_{char}[XY] = P_{char}(x=1|y=1) * P(y=1)
    a_xy_stat[EXY_CHAR_IDX] = (char_occ / float(a_y_cnt[TTL_CHAR_IDX])) * a_p_y
    # total number of messages in which that character n-gram appeared with
    # that class
    msg_occ = float(sum(a_xy_cnt.itervalues()))
    # E_{msg}[XY] = P_{msg}(x=1|y=1) * P(y=1)
    a_xy_stat[EXY_MSG_IDX] = (msg_occ / ttl_msgs) * a_p_y
    # E_{n_msg}[XY] = \sum_n(P_{msg}(x=n|y=1) * P(y=1))
    a_xy_stat[EXY_NMSG_IDX] = e_msg_n * a_p_y


def _compute_char_stat(a_char_stat, a_cls_cnt, a_ttl_chars, a_ttl_msg):
    """Compute probability, mean, and variance for single item.

    Args:
    -----
    a_char_stat: list
      character statistics to be populated
    a_cls_cnt: list
      class counters on that char n-gram
    a_ttl_chars: float
      total number of characters
    a_ttl_msg: float
      total number of messages

    Returns:
    --------
    (void)
      updates statistics in place

    """
    # compute total counter and probability of that char n-gram for
    # that class
    a_char_stat[CNT_CHAR_IDX] = a_char_stat[PROB_CHAR_IDX] = \
        sum(k * v for k, v in a_cls_cnt.iteritems())
    # normalize character probability and compute its variance
    a_char_stat[PROB_CHAR_IDX] /= a_ttl_chars or 1.
    a_char_stat[VAR_CHAR_IDX] = a_char_stat[PROB_CHAR_IDX] * \
        (1. - a_char_stat[PROB_CHAR_IDX])
    # estimate the number of messages in which this char n-gram
    # occurred
    a_char_stat[CNT_MSG_IDX] = a_char_stat[PROB_MSG_IDX] = \
        sum(a_cls_cnt.itervalues())
    # normalize message probability
    a_char_stat[PROB_MSG_IDX] /= a_ttl_msg or 1.
    a_char_stat[VAR_MSG_IDX] = a_char_stat[PROB_MSG_IDX] * \
        (1. - a_char_stat[PROB_MSG_IDX])
    # estimate mean and variance of that char n-gram
    a_char_stat[MEAN_N_IDX], a_char_stat[VAR_N_IDX] = \
        _compute_mean_var(a_cls_cnt, a_ttl_msg)


def _compute_mean_var(a_stat, a_total):
    """Compute probability, mean, and variance of given variable.

    Args:
    -----
      a_stat: Counter
        statistics on values that the given variable might take on
      a_total: float
        total number of items in population

    Returns:
    --------
      (mean, var): tuple(float, float)

    """
    if a_total <= 0:
        a_total = 1
    mean = var = iprob = 0.
    stat = [(val, float(cnt) / a_total) for val, cnt in
            a_stat.iteritems()]
    for val, iprob in stat:
        mean += val * iprob
    # to prevent catastrophic cancellation
    for val, iprob in stat:
        var += iprob * ((val - mean)**2)
    return (mean, var)


def compute_cov_cor(a_cls_stat, a_char_stat, a_char_cls_stat):
    """Compute covariance and correlation coefficient of char n-grams.

    Args:
    -----
    a_cls_stat: dict(list)
      statistics on classes (dependant variables)

    a_char_stat: dict(list)
      statistics on character n-grams (covariates)

    a_char_cls_stat: dict(defaultdict(list))
      statistics on character n-grams (covariates)

    Returns:
    --------
    :dict(dict((float, float, float, float, float, float)))
      covariance and correlation coefficients for char n-grams

    """
    cov_cor = {char:
               {cls: [0.] * COV_COR_MAX_IDX for cls in a_cls_stat}
               for char in a_char_stat}

    mu_y = var_y = 0.
    char_stat = cls_stat = None
    char_cc = char_cls_cc = None
    for ichar, icharclsstat in a_char_cls_stat.iteritems():
        char_cc = cov_cor[ichar]
        char_stat = a_char_stat[ichar]
        for icls, (EXY_char, EXY_msg, EXY_nmsg) in icharclsstat.iteritems():
            char_cls_cc = char_cc[icls]
            mu_y, var_y = a_cls_stat[icls]
            # total covariance and correlation of that character sequence in
            # the whole corpus
            char_cls_cc[COV_CHAR_IDX], char_cls_cc[COR_CHAR_IDX] = \
                _cov_cor(EXY_char,
                         char_stat[PROB_CHAR_IDX], char_stat[VAR_CHAR_IDX],
                         mu_y, var_y
                         )
            # total covariance and correlation of the occurrence of that
            # character sequence in a message
            char_cls_cc[COV_MSG_IDX], char_cls_cc[COR_MSG_IDX] = \
                _cov_cor(EXY_msg,
                         char_stat[PROB_MSG_IDX], char_stat[VAR_MSG_IDX],
                         mu_y, var_y
                         )
            # total covariance and correlation of the number of occurrences of
            # that character sequence in a message
            char_cls_cc[COV_NMSG_IDX], char_cls_cc[COR_NMSG_IDX] = \
                _cov_cor(EXY_nmsg,
                         char_stat[MEAN_N_IDX], char_stat[VAR_N_IDX],
                         mu_y, var_y
                         )
    return cov_cor


def _cov_cor(a_EXY, a_mu_x, a_var_x, a_mu_y, a_var_y):
    """Compute char/class covariance/correlation.

    Args:
    -----
    a_EXY: float
      joint expectation
    a_mu_x: float
      mean of the covariate
    a_var_x: float
      variance of the covariate
    a_mu_y: float
      mean of the dependent variable
    a_var_y: float
      variance of the dependent variable

    Returns:
    --------
      2-tuple: covariances, correlation coefficient

    """
    cov = a_EXY - a_mu_x * a_mu_y
    cor = cov / sqrt(a_var_x * a_var_y)
    return (cov, cor)


##################################################################
# Main
def main():
    """Find character n-grams characteristic to specific sentiment classes.

    Returns:
    --------
    0 on success, non-0 other

    """
    argparser = argparse.ArgumentParser(description="""Script for finding
character n-grams characteristic to specific sentiment classes.""")
    argparser.add_argument("-n", "--n-char", help="length of character n-gram",
                           type=int, default=DFLT_N)
    argparser.add_argument("-v", "--verbose", help="output statistics",
                           action="store_true")
    argparser.add_argument("files", help="input files in TSV format",
                           type=argparse.FileType('r'), nargs='*',
                           default=[sys.stdin])
    args = argparser.parse_args()

    # iterate over lines of files and count character combinations
    cls_cnt, char_cnt = compute_cnt(((ifields[GLD_IDX], ifields[TXT_IDX])
                                     for ifile in args.files
                                     for ifields in iterlines(ifile)
                                     ), args.n_char)

    # compute statistics (mean and variance) of chararacter sequences
    cls_stat, char_stat, char_cls_stat = compute_stat(cls_cnt, char_cnt)

    # output class statistics
    print("{:13s}{:15s}{:10s}".format("Class", "Prob (Mean)", "Variance"))
    for icls, istat in cls_stat.iteritems():
        print("{:10s}{:-10.4f}{:-15.4f}".format(icls, *istat))
    print()
    # output char statistics
    print("{:8s}{:13s}{:13s}{:13s}{:10s}{:13s}{:11s}{:16s}{:13s}".format(
        "Char", "Char Cnt", "Char Prob",
        "Char Var", "Msg Cnt", "Msg Prob",
        "Msg Var", "Mean # in Msg", "Var # in Msg"))
    for ichar, icharstat in char_stat.iteritems():
        print("{:8s}{:8.0f}{:13f}{:13f}{:10.0f}{:13f}{:13f}"
              "{:13f}{:13f}".format(ichar, *icharstat))
    print()

    # compute covariance, correlations
    cov_cor = compute_cov_cor(cls_stat, char_stat, char_cls_stat)
    # compute and output class-character covariance statistics
    print("{:10s}{:8s}{:16s}{:16s}{:16s}{:16s}{:16s}{:16s}".format(
        "X", "Y", "COV_{char}[XY]", "COR_{char}[XY]",
        "COV_{msg}[XY]", "COR_{msg}[XY]",
        "COV_{n_msg}[XY]", "COR_{n_msg}[XY]"))
    for ichar, icharstat in cov_cor.iteritems():
        for icls, iclstat in icharstat.iteritems():
            print("{:7s}{:8s}{:16f}{:15f}{:16f}{:15f}{:16f}{:16f}".format(
                ichar, icls, *iclstat))
    # print("***char_cnt =", repr(char_cnt), file=sys.stderr)
    # print("*** cls_stat =", repr(cls_stat), file=sys.stderr)
    # print("*** char_stat =", repr(char_stat), file=sys.stderr)

if __name__ == "__main__":
    main()
