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
