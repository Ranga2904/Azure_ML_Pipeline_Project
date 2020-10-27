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
AutoML selected a VotingEnsemble classifier - as mentioned earlier, ensemble techniques improve machine learning performance through using multiple models. Voting is 'soft' meaning that it does a weighted average of predicted probabilities to arrive at a final cass. This VotingClassifier has the following hyperparameter values, with accompanying explanations:

#### L1_ratio = 0.8367
This mixing parameter tells us that AutoML relied on a combination of L1 and L2 regularization to dampen overfitting during training.
#### Learning_rate = constant
This tells us that the step size at each iteration was held constant, as the model learnt new details. This isn't always advisable since we might prefer larger learning rates at the start that slow down as the model learns more.
#### Loss = modified huber
This is the loss function to be minimized when evaluating iterations. Unlike some other loss functions e.g. squared loss, Huber loss has a higher tolerance for outliers.
### max_iter = 1000
This gives us the maximum number of iterations allowable for training the model, irrespective of where model error was relative to tolerance (see below).
### n_jobs = 1
This setting means that I will only use 100% of 1 of the computing resource's cores - no concurrent operations. 
### penalty = 'l2'
This setting explains that the model chose l2 ridge regulatization to penalize errors i.e. it was willing to eliminate some features rather than simply shrinking coefficients
### power_t = 0.222
This is the t-test value used to decide whether a paricular outcome was statistically significant and therefore a legitimiate result i.e. were we outside the corresponding confidence interval to trust the final classification, or was this no better than simple chance?
### random_state = None
This enables the algorithm's inherent stochasticity and slightly different fits at different times mean that we might not get the same result at different trials.
### tol = 0.0001
This is the tolerance that AutoML searched for when iterating - once the error between predicted and actual were less than tolerance, the algorithm would stop iterating. 
### weights = 0.111,0.333,0.222,0.111,0.111,0.111
This is a sequence of weights applied to predicted class probabilities to decide the final model output

## Pipeline comparison
LogisticRegression performed adequately with an accuracy of 91.3%, with AutoML's results (through a VotingEnsemble) being only marginally better (91.5%)

## Future work
Opportunities to improve modeling include: 
- ordinal encoding in place of one-hot encoding for variables like housing and education which may have a natural order when predicting policy subscriptions
- adding the solver_type to assess 'sag' in its performance for prediction
- Try BayesianParameter search
- Oversample the minority class to reduce existing imbalance between those who do and don't subscribe
