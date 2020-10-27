# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. I built and optimized an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset holds information from marketing campaigns initiated by a Portuguese marketing institution, detail coming in from phone calls. We seek to predict through classification whether or not the client would subscribe to a bank term deposit

LogisticRegression performed adequately with an accuracy of 91.1%, with AutoML's results (through a VotingEnsemble) being marginally better (91.7%). This is expected since the VotingEnsemble for classification (or 'VotingClassifier') sums up the class predictions from multiple classifiers to get a final 'vote tally' on which class is the most popularly predicted. That class is then produced as the VotingEnsemble's output. Like any other ensemble algorithm, this approach of getting many inputs surpasses any single model i.e. LogisticRegression.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
Data is fed in with feature engineering completed to encode non-numerical features. After splitting into testing and training sets, the LogisticRegression classifier is used to fit and train with hyperparameter tuning done by HyperdriveConfig.
See diagram of model architecture pipeline.

**What are the benefits of the parameter sampler you chose?**
The RandomParameter sampler is more flexible in permitting continuous rather than just 'choice' or discrete selections of hyperparameter values. In my evaluation, I ended up using 'choice' to iterate through ranges of C and max iterations any use of a 'continuous' or 'uniform' sampler is enabled by RandomParameter. RandomParameter is also less compute-intensive and expensive, thanks to a restricted range of values for sampling - an advantage not present in grid sampling.

**What are the benefits of the early stopping policy you chose?**
BanditPolicy is oriented more towards achieving absolute performance through use of a slack factor while Truncation Policies or Median Stopping policies focus on improving performance relative to other runs 

## AutoML
AutoML selected a VotingEnsemble classifier - as mentioned earlier, ensemble techniques improve machine learning performance through using multiple models. Voting is 'hard' meaning that it takes the most popularly predicted class as the final answer. This VotingClassifier has the following parameter values:
l1_ratio = 0.8367
learning_rate = constant
loss = 'modified huber'
max_iter = 1000
n_jobs = 1
penalty = 'l2'
power_t = 0.222
random_state = None
tol = 0.0001
weights = 0.111,0.333,0.222,0.111,0.111,0.111

## Pipeline comparison
LogisticRegression performed adequately with an accuracy of 91.3%, with AutoML's results (through a VotingEnsemble) being only marginally better (91.5%)

## Future work
Opportunities to improve modeling include: 
- ordinal encoding in place of one-hot encoding for variables like housing and education which may have a natural order when predicting policy subscriptions
- adding the solver_type to assess 'sag' in its performance for prediction
- Try BayesianParameter search
- Oversample the minority class to reduce existing imbalance between those who do and don't subscribe
