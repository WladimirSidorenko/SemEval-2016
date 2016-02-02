******************************************************
* SemEval-2016 Task 4: Sentiment Analysis on Twitter *
*                                                    *
*               SCORER                               *
*                                                    *
* http://alt.qcri.org/semeval2016/task4/             *
* semevaltweet@googlegroups.com                      *
*                                                    *
******************************************************


Version 2.0; January 12, 2016


Task organizers:

* Preslav Nakov, Qatar Computing Research Institute, HBKU
* Alan Ritter, The Ohio State University
* Sara Rosenthal, Columbia University
* Fabrizio Sebastiani, Qatar Computing Research Institute, HBKU
* Veselin Stoyanov, Facebook


To run the scorer (format checker) for task X (where X in [A..E]), run this:

> perl score-semeval2016-task4-subtask{X}.pl GOLD_STANDARD_FILE PREDICTION_FILE



DATA FORMAT for gold standard and for the prediction files (given input).


-----------------------SUBTASK A-----------------------------------------

--Test Data--
The format for the test input file is as follows:

  id<TAB>UNKNOWN<TAB>tweet_text

for example:

  1       UNKNOWN  amoure wins oscar
  2       UNKNOWN  who's a master brogramer now?

--System Output--
We expect the following format for the prediction file (as for GOLD):

  id<TAB>predicted_sentiment_4_tweet

where predicted_sentiment_4_tweet can be 'positive', 'neutral' or 'negative'.

For example:
1        positive
2        neutral


-----------------------SUBTASK B-----------------------------------------

--Test Data--
The format for the test input file is as follows:

  id<TAB>topic<TAB>UNKNOWN<TAB>tweet_text

for example:

  1      aaron rodgers       UNKNOWN       I just cut a 25 second audio clip of Aaron Rodgers talking about Jordy Nelson's grandma's pies. Happy Thursday.
  2      aaron rodgers       UNKNOWN       Tough loss for the Dolphins last Sunday in Miami against Aaron Rodgers &amp; the Green Bay Packers: 27-24.

--System Output--
We expect the following format for the prediction file (as for GOLD):

  topic<TAB>predicted_sentiment_4_topic

where predicted_sentiment_4_topic can be "positive" or "negative" (NOTE: no "neutral"!)

For example:
  1      aaron rodgers       positive
  2      aaron rodgers       negative


-----------------------SUBTASK C-----------------------------------------
--Test Data--
Same as for subtask B.

--System Output--
We expect the following format for the prediction file (as for GOLD):

  id<TAB>topic<TAB>predicted_sentiment_4_topic

where predicted_sentiment_4_topic can be -2, -1, 0, 1, or 2.

For example:
  1      aaron rodgers       1
  2      aaron rodgers       0


-----------------------SUBTASK D-----------------------------------------
--Test Data--
Same as for subtask B.

--System Output--
We expect the following format for the prediction file (as for GOLD):

  topic<TAB>part_positive<TAB>part_negative

where part_positive and part_negative are floating point numbers between 0.0 and 1.0, and part_positive + part_negative == 1.0

For example:
  aaron rodgers       0.7       0.3
  peter pan           0.9       0.1


-----------------------SUBTASK E-----------------------------------------
--Test Data--
Same as for subtask B.

--System Output--
We expect the following format for the prediction file (as for GOLD):

  topic<TAB>label-2<TAB>label-1<TAB>label0<TAB>label1<TAB>label2

where label-2 to label2 are floating point numbers between 0.0 and 1.0, and the five numbers sum to 1.0. label-2 corresponds to the fraction of tweets labeled as -2 in the data and so on.

For example:
  aaron rodgers       0.025 0.325   0.5    0.1 0.05
  peter pan           0.05  0.40    0.5    0.05 0.0

-------------------------------------------------------------------------


AGGREGATORS:

We also provide two aggregators that can turn tweet-level predictions (if any) into the aggregate predictions needed for tasks D and E. Here is how you can run them:

> perl aggregate-semeval2016-task4-subtask{X}.pl FILE

The aggregator will produce file FILE.aggregate with the aggregate predictions in it. Note that this is not meant in any way to suggest that the aggregate predictions should necessarily be obtained by first generating tweet-level predictions and then aggregating these; participants can use any method they want for obtaining the aggregate predictions, including methods that do not involve the generation of tweet-level predictions as an intermediate step.

LICENSE

The accompanying dataset is released under a Creative Commons Attribution 3.0 Unported License (http://creativecommons.org/licenses/by/3.0/).



CITATION

You can cite the following paper when referring to the dataset:


@InProceedings{SemEval:2016:task4,
  author    = {Preslav Nakov and Alan Ritter and Sara Rosenthal and Veselin Stoyanov and Fabrizio Sebastiani},
  title     = {{SemEval}-2016 Task 4: Sentiment Analysis in {T}witter},
  booktitle = {Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval 2016)},
  year      = {2016},
  publisher = {Association for Computational Linguistics}
}


USEFUL LINKS:

Google group: semevaltweet@googlegroups.com
SemEval-2016 Task 4 website: http://alt.qcri.org/semeval2016/task4/
SemEval-2016 website: http://alt.qcri.org/semeval2016/
