#!/usr/bin/env python
# -*- coding: utf-8; -*-

"""Tokenizer for tweets.

might be appropriate for other social media dialects too.  general
philosophy is to throw as little out as possible.  development
philosophy: every time you change a rule, do a diff of this program's
output on ~100k tweets.  if you iterate through many possible rules
and only accept the ones that seeem to result in good diffs, it's a
sort of statistical learning with in-the-loop human evaluation :)

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function
import re, sys

##################################################################
# Variables and Constants
__author__="brendan o'connor (anyall.org)"
assert '-' != '―'

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
regex_or = lambda *items: '(' + '|'.join(items) + ')'
pos_lookahead = lambda(r): '(?=' + r + ')'
neg_lookahead = lambda(r): '(?!' + r + ')'
optional = lambda(r): '(%s)?' % r
regexify_abbrev = lambda(ichars): "".join([r'%s\.' % x for c in ichars \
                                               for x in "[%s%s]" % (c,c.upper())])

NormalEyes = r'[:=]'
Wink = r'[;]'
NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...
HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pP]'
OtherMouths = r'[doO/\\]'  # remove forward slash if http://'s aren't cleaned
Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)
Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )
Emoticon = ("("+NormalEyes+"|"+Wink+")" + NoseArea + "(" + Tongue + "|" + \
                OtherMouths + "|" + SadMouths + "|" + HappyMouths + ")")
Emoticon_RE = mycompile(Emoticon)
PunctChars = r'''['“".?!,:;]'''
Punct = '%s+' % PunctChars
Entity = '&(amp|lt|gt|quot);'
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
Url = (r'\b' +
    regex_or(UrlStart1, UrlStart2) +
    UrlBody +
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))
Url_RE = re.compile("(%s)" % Url, re.U|re.I)
Timelike = r'\d+:\d+'
NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))
Abbrevs1 = ['am','pm','us','usa','ie','eg']
Abbrevs = [regexify_abbrev(a) for a in Abbrevs1]

BoundaryNotDot = regex_or(r'\s', '[“"?!,:;]', Entity)
aa1 = r'''([A-Za-z]\.){2,}''' + pos_lookahead(BoundaryNotDot)
aa2 = r'''([A-Za-z]\.){1,}[A-Za-z]''' + pos_lookahead(BoundaryNotDot)
ArbitraryAbbrev = regex_or(aa1,aa2)
Separators = regex_or('--+', '―')
Decorations = r' [  ♫   ]+ '.replace(' ','')
EmbeddedApostrophe = r"\S+'\S+"
ProtectThese = [Emoticon, Url, Entity, Timelike, NumNum, \
                    NumberWithCommas, Punct, ArbitraryAbbrev, Separators, \
                    Decorations, EmbeddedApostrophe,]
Protect_RE = mycompile(regex_or(*ProtectThese))
# fun: copy and paste outta http://en.wikipedia.org/wiki/Smart_quotes
EdgePunct      = r"""[  ' " “ ” ‘ ’ < > « » { } ( \) [ \]  ]""".replace(' ','')
#NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
NotEdgePunct = r"""[a-zA-Z0-9]"""
EdgePunctLeft  = r"""(\s|^)(%s+)(%s)""" % (EdgePunct, NotEdgePunct)
EdgePunctRight =   r"""(%s)(%s+)(\s|$)""" % (NotEdgePunct, EdgePunct)
EdgePunctLeft_RE = mycompile(EdgePunctLeft)
EdgePunctRight_RE= mycompile(EdgePunctRight)
AposS = mycompile(r"(\S+)('s)$")
WS_RE = mycompile(r'\s+')

##################################################################
# Exceptions
class AlignmentFailed(Exception): pass

##################################################################
# Methods
def align(toks, orig):
  s_i = 0
  alignments = [None]*len(toks)
  for tok_i in range(len(toks)):
    while True:
      L = len(toks[tok_i])
      if orig[s_i:(s_i+L)] == toks[tok_i]:
        alignments[tok_i] = s_i
        s_i += L
        break
      s_i += 1
      if s_i >= len(orig): raise AlignmentFailed((orig,toks,alignments))
      #if orig[s_i] != ' ': raise AlignmentFailed("nonspace advance: %s" % ((s_i,orig),))
  for a in alignments:
    if a is None:
      raise AlignmentFailed((orig,toks,alignments))
  return alignments

def unicodify(s, encoding='utf8', *args):
  if isinstance(s,unicode): return s
  if isinstance(s,str): return s.decode(encoding, *args)
  return unicode(s)

def tokenize(tweet):
  text = unicodify(tweet)
  text = squeeze_whitespace(text)
  t = Tokenization()
  t += simple_tokenize(text)
  t.text = text
  t.alignments = align(t, text)
  return split_contractions(t)

def simple_tokenize(text):
  s = text
  s = edge_punct_munge(s)

  # strict alternating ordering through the string.  first and last are goods.
  # good bad good bad good bad good
  goods = []
  bads = []
  i = 0
  if Protect_RE.search(s):
    for m in Protect_RE.finditer(s):
      goods.append( (i,m.start()) )
      bads.append(m.span())
      i = m.end()
    goods.append( (m.end(), len(s)) )
  else:
    goods = [ (0, len(s)) ]
  assert len(bads)+1 == len(goods)

  goods = [s[i:j] for i,j in goods]
  bads  = [s[i:j] for i,j in bads]
  #print goods
  #print bads
  goods = [unprotected_tokenize(x) for x in goods]
  res = []
  for i in range(len(bads)):
    res += goods[i]
    res.append(bads[i])
  res += goods[-1]

  res = post_process(res)
  return res

def post_process(pre_toks):
  # hacky: further splitting of certain tokens
  post_toks = []
  for tok in pre_toks:
    m = AposS.search(tok)
    if m:
      post_toks += m.groups()
    else:
      post_toks.append( tok )
  return post_toks

def squeeze_whitespace(s):
  new_string = WS_RE.sub(" ",s)
  return new_string.strip()

def edge_punct_munge(s):
  s = EdgePunctLeft_RE.sub( r"\1\2 \3", s)
  s = EdgePunctRight_RE.sub(r"\1 \2\3", s)
  return s

def unprotected_tokenize(s):
  return s.split()

def split_contractions(tokens):
    # Fix "n't", "I'm", "'re", "'s", "'ve", "'ll"  cases
    new_token_list = []
    for token in tokens:
        new_tk = None
        if token[-3:] == 'n\'t':
            new_tk = token[:-3]
            new_token_list.append('n\'t')
        elif token == 'I\'m' or token == 'i\'m':
            new_token_list.append('I')
            new_token_list.append('\'m')
        elif token[-3:] == '\'re':
            new_tk = token[:-3]
            new_token_list.append('\'re')
        elif token[-2:] == '\'s':
            new_tk = token[:-2]
            new_token_list.append('\'s')
        elif token[-3:] == '\'ve':
            new_tk = token[:-3]
            new_token_list.append('\'ve')
        elif token[-3:] == '\'ll':
            new_tok = token[:-3]
            new_token_list.append('\'ll')
        else:
            new_token_list.append(token)
        # Add new token if one exists
        if new_tk:
            #sys.stderr.write('Split following contraction: %s\n' % token)
            new_token_list.insert(-1, new_tk)
    return new_token_list

##################################################################
# Classes
class Tokenization(list):
  "List of tokens, plus extra info."

  def __init__(self):
    self.alignments = []
    self.text = ""

  def subset(self, tok_inds):
    new = Tokenization()
    new += [self[i] for i in tok_inds]
    new.alignments = [self.alignments[i] for i in tok_inds]
    new.text = self.text
    return new

  def assert_consistent(t):
    assert len(t) == len(t.alignments)
    assert [t.text[t.alignments[i] : (t.alignments[i]+len(t[i]))] for i \
                in xrange(len(t))] == list(t)

##################################################################
# Main
if __name__=='__main__':
  for line in sys.stdin:
    print(" ".join(tokenize(line[:-1])).encode('utf-8'))
