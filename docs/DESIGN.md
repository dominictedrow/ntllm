# DESIGN.md: Name That LLM, design v2 (trap defense: all 5 traps held), locked 2026-11-18

Baseline Model Submission:
I think as a baseline we can just submit the kNN gzip classifier and see where that lands
It requires no training so it will get us on the leaderboard early and also a reasonable idea as to how many params I should commit to the project
Features:
For this model there isn’t anything that isn’t a feature the entire training set is compressed into the kNN so for this model feature engineering isn’t really a factor
For the other models like the SVM or naive bayes I will use TF-IDF to filter the text a little more but filtering out the topic words for every prompt is impossible and also not a good practice for generalization
In this context the features will be the vocab
For the ensemble model the features will basically be the same but if I use the meta model the model predictions will be features, and if I use the gzip classifier at the beginning for families then the models will need to intake an addition family feature
For the encoder just the text itself will be enough as a feature, maybe an addition gzip family input if we find out it helps
Models:
SVM
Naive Bayes
Gzip + kNN
Encoder
Data:
I will obviously use the test and training set that kaggle gives me
I have gpt pro and claude pro so I can get all the model outputs from those companies
Regardless of whether they match the exact config as long as i disable what I have added to these models I will most likely get a similar result to the models they are running
I can get all of the gemini models except for 3.7 flash since it has been retired
Kimi3 I can run on something since it is open source and same with the ollama models if they are open source
All other models I won’t have access to
All the models I don’t have access to should be trained on more heavily so whenever a gemini example comes through the weights should be punished more if the model is wrong
All synthetic data can only go into the training split
For prompt similarity I guess I could compare the words that appear using the TF-IDF tables but like just having claude read all of them is faster
First Three Experiments:
Comparing Balanced acc of all 4 models
Tuning the hyperparams of each for about 5 different training runs and then seeing how each of them performs on the k-fold val
Comparing Encoder with gzip family input and without
Difference of one feature
Comparing ensembling the SVM and gzip in comparison to just running the SVM
Validation Scheme:
I will use blending by separating out a val split and then have all the models train on the non val split
I will then run the models on the val split and use their prediction outputs to train the meta learner model
I will then score the meta model on the ensembles performance in the test set
I will also use k-folds to train the models individually and I will use 4 folds because I want the val fold to be decently large
All folds will be separated by prompt 
Training Setup:
There should be a different training script for each design that I have including the four models by themselves
When the training runs I want to see the runtime per epoch, the balanced acc, precision, and perplexity
There should be model check pointing written to a file so that claude can access results
I feel like putting the data into folds is something that should be efficiently done I guess we can redo it every time I do a training run but I feel there’s a better way
There should also be scripts dedicated to running the three experiments I already wrote so when I run that experiment script the comparison is shown
All the scripts should be easy for me to run with intuitive flags I can configure
I shouldn’t have to like write the most complicated folder path I’ve ever seen when it could just be python scripts/train.py or something like that
Once balanced acc is where I want it after running on test set then we can push to kaggle and see how we do
Inference and compute budget:
I can run 24/7 on this windows box
The specs are: 5070, i5 core, 16gb of RAM, 12 gb of VRAM
Predictions:
I think the encoder will do well on local data and then get like a 0.5 on the leader board
I think the gzip ensemble will get a 0.6 on leader board
Gzip by itself will get a 0.4 at most
SVM will probably get a 0.5
Naive bayes will probably get a 0.4 also
Fallback:
I will just look at what other people are doing on the leader board
Or I will just make a humungous transformer and turn the generalization hyper params really high (I wanna try it it prolly won’t do super good though)
For synthetic data I think that there will be datasets out there with prompt responses readily available for many models
Leakage Risks:
If I have the TF-IDF table read too much data it could skew predictions
All tables can only read from training data nothing that is in val or test
If i have the meta learner get scored on the val rows it will do far too well
If any of the folds aren’t completely separated by prompt then performance won’t work as intended
Leaderboard:
In all honesty, I will try to maximise my position on the leader board since that could mean people might want to join my team, I might get some recognition, and I can say that I was in the __% for a little bit
That being said the public leader board doesn’t really equate to how well you will do on the actual test data set since the test data set is 60% and different than public
This means that I have to generalize as much as possible with literally everything I submit
For my final 2 submissions I will choose the models that have the highest scores and have been generalized the most
I would love to set a number but I don’t know how I will feel, the numbers I will be working with, or even the models I will be using so I can’t set a specific threshold but I can say what I am looking at
Stacking:
If I try stacking which I likely will I will have k folds where k=num base models and then have each model have a different fold held out during training
I will then have the meta learner model train on how each base model predicts on these data points
During training the encoder can only have the training set, the TF-IDF model can only be from the training set, kNN can only be the training set

---
## Addenda (Dom, 2026-11-18, his words; these override the lines above where they conflict)

Task (from Oct teardown): I am making a model that can detect speech patterns within AI-generated text so I can match it with a frontier model. I get text and then I have to decide the LLM behind it.

Metric (worked example): 1,000 GPT right + 50 Claude wrong gives accuracy 0.95 but balanced accuracy 0.5.

Ensembling: I want to do blending. (The Stacking section above is superseded by blending.)

Validation: "Nah I meant like how you split a data set into train, val, an test i would run it on the test one." = three splits of train.csv, all separated by prompt. Base models train on train; the meta model trains on their predictions on val; the meta model is scored on the local test split. Kaggle's test.csv is only for submissions.

Final submissions: pick B-and-C style: not the one tuned on the public leaderboard; two different approaches as a hedge.

Prompt overlap: use gzip similarity to flag synthetic prompts near competition prompts and delete those within a threshold.

---
## Open checks from the defense (questions to answer during the build, not design changes)
1. Does train.csv include prompt text, or only an anonymized prompt_id? If no text, what do you compare synthetic prompts against?
2. How do you pick the gzip similarity threshold for short prompts, and check it catches real rewordings?
3. Prompts per split: with 150 prompts across train / val / local test, how many does each piece get? Will meta-model selection happen inside val (folds) so local test is touched once?
4. What is the noise bar (fold-to-fold spread) for calling a change real?
5. Per-label recall: do labels you generated synthetic data for gain while unreachable labels (GLM, maybe Qwen, gemini-3.7-flash) lose?
6. Do app outputs look like the competition's passages for that label? (e.g. does your classifier label them correctly before you generate thousands?)
7. Write the concrete rule for "generalized" used to pick the 2 finals.
