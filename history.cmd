This submission relies on a deep character-convolution neural network
approach that treats input messages as a plain sequence of characters
and successively converts those characters into their vector
representation (embeddings).  After this conversion is done, a set of
convolutional filters is applied to the input embeddings (currently,
we are using 4 filters of width three, 16 filters of width four, and
24 filters of width five).  We take max-over-time of these
convolutions and pass this output to four non-linear layers with
linear in-between transformations (currently, we are using one highway
layer (Srivastava, 2015), one hyper-tangent layer, one layer with a
sigmoid activation function, and a soft-max prediction layer as the
final step).  We use RMS-prop (Tieleman, 2012) as a training
function. The cost function that we are optimizing has been tailored
to the peculiarities of this task and is formulated as follows:

C = -log(pred[y]) + L3 * dist + L2 * sum([sum(p**2) for p in
params]),

where $pred[y]$ is the probability of predicting the gold label, $L3 *
dist$ is the squared Euclidean distance between the made prediction
and the true gold label, and the $L2 * sum([sum(p**2) for p in
params])$ term denotes the L2 regularization of system's parameters.

We do no preprocessing of the input messages except for removing stop
words and lowercasing the input strings.  Additionally, we have
enriched the training corpus by randomly sampling polar terms from two
sentiment lexica -- the Sentiment Clues Lexicon (Wilson et al., 2005)
and the NRC Hashtag Sentiment Lexicon (Mohammad, 2012) -- and
assigning them gold labels according to their specified prior
polarities.

References:
-----------
@inproceedings{Wilson:05,
  author    = {Theresa Wilson and Janyce Wiebe and Paul Hoffmann},
  title     = {Recognizing Contextual Polarity in Phrase-Level
                  Sentiment Analysis},
  booktitle = {{HLT/EMNLP} 2005, Human Language Technology Conference
                  and Conference on Empirical Methods in Natural
                  Language Processing, Proceedings of the Conference,
                  6-8 October 2005, Vancouver, British Columbia,
                  Canada},
  year      = {2005},
  crossref  = {DBLP:conf/emnlp/2005},
  url       = {http://acl.ldc.upenn.edu/H/H05/H05-1044.pdf},
  timestamp = {Thu, 21 Dec 2006 10:35:02 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/naacl/WilsonWH05},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}

@InProceedings{Mohammad:12,
  author    = {Mohammad, Saif},
  title     = {\#Emotional Tweets},
  booktitle = {{*SEM 2012}: The First Joint Conference on Lexical and Computational Semantics -- Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation {(SemEval 2012)}},
  month     = {7-8 June},
  year      = {2012},
  address   = {Montr\'eal, Canada},
  publisher = {Association for Computational Linguistics},
  pages     = {246--255},
  url       = {http://www.aclweb.org/anthology/S12-1033}
}

@misc{Tieleman2012,
  title={{Lecture 6.5---RmsProp: Divide the gradient by a running average of its recent magnitude}},
  author={Tieleman, T. and Hinton, G.},
  howpublished={COURSERA: Neural Networks for Machine Learning},
  year={2012}
}

@article{Srivastava:15,
  author    = {Rupesh Kumar Srivastava and
               Klaus Greff and
               J{\"{u}}rgen Schmidhuber},
  title     = {Highway Networks},
  journal   = {CoRR},
  volume    = {abs/1505.00387},
  year      = {2015},
  url       = {http://arxiv.org/abs/1505.00387},
  timestamp = {Mon, 01 Jun 2015 14:13:54 +0200},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/SrivastavaGS15},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}

------------------------------------------------------------------
Dedicated Development Set
./scripts/twitter_sentiment train -l scripts/data/HashtagSentimentAffLexNegLex/ -l scripts/data/SubjectivityClues/ -d data/dev/topic-five-point.CE.dev.txt data/train/topic-five-point.CE.train.txt

Minimum train cost = 42.7560875863, minimum dev error rate = 1.1855878855

>./scripts/twitter_sentiment test data/dev/.topic-five-point.CE.dev.txt | ./scripts/evaluate.py
Using gpu device 0: GeForce 840M (CNMeM is disabled)
rhostat {0: 'negative', 1: 'positive'} =  [0.3771043771043771, 0.3141592920353982, 0.23809523809523808, 0.0, 0.33587786259541985]
Macro-averaged MAE: 1.3123960
Micro-averaged MAE: 0.9594743
$\rho$^{PN}:        0.2530474
Accuracy:           0.3318729

------------------------------------------------------------------
Mixed-In Development Set
./scripts/twitter_sentiment train -l scripts/data/HashtagSentimentAffLexNegLex/ -l scripts/data/SubjectivityClues/ data/dev/topic-five-point.CE.dev.txt data/train/topic-five-point.CE.train.txt

>./scripts/twitter_sentiment test data/dev/.topic-five-point.CE.dev.txt | ./scripts/evaluate.py
Using gpu device 0: GeForce 840M (CNMeM is disabled)
rhostat {0: 'negative', 1: 'positive'} =  [0.494949494949495, 0.252212389380531, 0.2857142857142857, 0.0, 0.12213740458015267]
Macro-averaged MAE: 1.2318927
Micro-averaged MAE: 0.9496166
$\rho$^{PN}:        0.2310027
Accuracy:           0.3099671
