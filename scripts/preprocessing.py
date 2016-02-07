#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Imports
from __future__ import print_function, unicode_literals

from evaluate import get_fields, TXT_IDX, TOPIC_IDX

import re
import sys

##################################################################
# Variables and Constants
# text normalization
ENCODING = "utf-8"

STOP_WORDS_RE = re.compile(r"(?:\b|^|-)(i|the(?:re|ir|[myn])?|"
                           "(?:any|every)body|"
                           "here|from|an?|i[stn]|[a']re|am|was|we(?:re)?|"
                           "ha[ds]|hav(?:e|ing?)|to|with|will|all|each|"
                           "wh(?:en|ich|ere|o|at)(?:ever)?|makes?|making|"
                           "may|might|i(?:'?m)?|of course|that|we\'?re|"
                           "\'?s|through|by|(?:get|put|say|set)(?:ting|s)?|"
                           "be(?:en|ing)?|this|as|going|do(?:es|ing?)?|did|"
                           "able|tomorrow|week|"
                           "(?:mon|tues|thurs|wednes|fri|satur|sun|yester|to)"
                           "day|"
                           "tomorrow|tonight|gives?|about|[bmw]e|y?ou(?:rs?)?|"
                           "u|some(?:thing|one)?|and|o[fr]|for|\'?(?:ve|d)|us|"
                           "s?he|on|(?:some)?where|so|[ia]t|it\'?s|gonna|cc|"
                           "\'?ll|may|take[ns]?|took|how|let|her?|she|it|"
                           "hi[ms]|my|(?:an)?other|g[eo]ts?|while|can|either|"
                           "[oi]nto|via|if|actually|hi|one|october|january|"
                           "february|march|april|may|june|july|august|"
                           "september|october|november|december|nov|aug|"
                           "feb|sept|two)(?:[-:,.%$\"'`]+|'(?:s|ve))?"
                           "(?=\s|[,?!]|\Z)")
PUNCT_RE = re.compile("(\s|\A|\w)(?:-+|,+|:+|[.%$\"`]+)(?=\s|\w|\Z)")
NEG_RE = re.compile("(?:\s|\A)(?:has|have|is|was|do|does|need|ca|"
                    "wo|would)(n'?t)(?:\s|\Z)")
AMP_RE = re.compile(r"&amp;")
GT_RE = re.compile(r"&gt;")
LT_RE = re.compile(r"&lt;")
DIGIT_RE = re.compile(r"\b[\d:]+(?:[ap]\.?m\.?|th|rd|nd|m|b)?\b")
HTTP_RE = re.compile(r"(?:https?://\S*|\b(?:(?:[\w]{3,5}://?|(?:www|bit)"
                     "[.]|(?:\w[-\w]+[.])+(?:a(?:ero|sia|[c-gil-oq-uwxz])|"
                     "b(?:iz|[abd-jmnorstvwyz])|c(?:at|o(?:m|op)|"
                     "[acdf-ik-orsuvxyz])|d[dejkmoz]|e(?:du|[ceghtu])|"
                     "f[ijkmor]|g(?:ov|[abd-ilmnp-uwy])|h[kmnrtu]|"
                     "i(?:n(?:fo|t)|[del-oq-t])|j(?:obs|[emop])|"
                     "k[eghimnprwyz]|l[abcikr-vy]|m(?:il|obi|useum|"
                     "[acdeghk-z])|n(?:ame|et|[acefgilopruz])|o(?:m|rg)|"
                     "p(?:ro|[ae-hk-nrstwy])|qa|r[eosuw]|s[a-eg-or-vxyz]|"
                     "t(?:(?:rav)?el|[cdfghj-pr])|xxx)\b)"
                     "(?:[^\s,.:;]|\.\w)*))")
AT_RE = re.compile(r"(RT\s+)?[.]?@\S+")
SSPACE_RE = re.compile(r"\\\s+")
SPACE_RE = re.compile(r"\s\s+")

NONMATCH_RE = re.compile("(?!)")
POS_RE = re.compile(r"[\b\A](amazing|better|best|cool|"
                    "fun|fant|good|great|like|love|luv|wow|wonder\b|:\))")
POS_CHAR = ''
NEG_CHAR = ''
TOPIC2RE = {}


##################################################################
# Methods
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


def _cleanse(a_line, a_topic=""):
    """Remove spurious elements from line.

    @param a_line - line to be cleansed
    @param (optional) a_topic - topic which the given line pertains to

    @return cleansed line

    """
    global TOPIC2RE
    # print("original line =", repr(a_line), file = sys.stderr)
    line = _safe_sub(HTTP_RE, "", a_line.lower())
    line = _safe_sub(AT_RE, "", line)
    line = _safe_sub(DIGIT_RE, "", line)
    line = _safe_sub(AMP_RE, "", line)
    line = _safe_sub(AMP_RE, "", line)
    line = _safe_sub(STOP_WORDS_RE, " ", line)
    if a_topic:
        a_topic = a_topic.lower()
        if a_topic not in TOPIC2RE:
            TOPIC2RE[a_topic] = re.compile(r"\b" +
                                           SSPACE_RE.sub(r"\s+",
                                                         re.escape(a_topic)) +
                                           r"\b")
        line = _safe_sub(TOPIC2RE[a_topic], "", line)
    line = _safe_sub(PUNCT_RE, "\\1 ", line)
    line = _safe_sub(SPACE_RE, ' ', line)
    # print("modified line =", repr(line), file = sys.stderr)
    line = GT_RE.sub(">", LT_RE.sub("<", line))
    # line = NEG_RE.sub(" not ", line)
    # line = POS_RE.sub(POS_CHAR + "\\1", line)
    # line = NEG_RE.sub(NEG_CHAR + "\\1", line)
    # print("modified line =", repr(line), file = sys.stderr)
    return line


def iterlines(a_file, a_tfld_idx=TOPIC_IDX):
    """Iterate over input lines of a file

    @param a_file - input file to iterate over
    @param a_tfld_idx - index of the topic field

    @return iterator over resulting TSV fields

    """
    ifields = []
    for iline in a_file:
        iline = iline.decode(ENCODING)
        # compute prediction and append it to the list of fields
        ifields = get_fields(iline)
        ifields[TXT_IDX] = _cleanse(ifields[TXT_IDX], ifields[a_tfld_idx])
        yield ifields
