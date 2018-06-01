# Kaggle: Mercedes-Benz Greener Manufacturing
This is a repo for the competition [Mercedes-Benz Greener Manufacturing](https://www.kaggle.com/c/mercedes-benz-greener-manufacturing). Note that the submissions here are late submission.

## Competition Description
### Goal
Predict the time (in seconds) that a car takes to pass testing for each variable.

### Data
* Training set size: 4209
* Testing set size: 4209
* Number of features: 387
...* Categorical: 9
...* Binary: 378

### Evaluation metric
Coefficient of determination (R-squared)

## Environments
* Language: python 3.6
* Editor: Sublime Text
* Libraries: numpy, pandas, xgboost, scikit-learn, matplotlib

## Used methods
All methods used are reported with R-squared and position on the private leaderboard. Since public leaderboard only uses 19% of test data, its score and position are unreliable. Most methods are ran in separate code files, which are stated in method descriptions.

Since the data consists of only categorical and binary variables, tree-based models seem like a natural fit. Therefore, [XGBoost](https://github.com/dmlc/xgboost/) is chosen as the standard model. To use XGBoost, categorical variables are converted using one-hot encoding.

### baseline
* R-squared: 0.52456
* position: 3225
* Code file: `/code/xgb_baseline.py`

Baseline is built by a XGBoost with default hyperparameters on default training set without feature engineering.

### Finetuned XGBoost
* R-squared: 0.54493
* position: 2234
* Code file: `/code/xgb_finetune.py`, `/code/random_search.py`

After baseline, a natural though to boost performance is to finetune the hyperparameters of XGBoost. Personally, I prefer random search over grid search, because given a great amount of hyperparameters for XGBoost, a thorough grid search is costly, while several research findings show that random search can provide fast convergence. 

The details of random search and sample distribution for each hyperparameter can be found in `/code/random_search.py`. Two-step searches are conducted, each for 4 hours. For the first search, hyperparameters are sampled from distributions stated empirically. For the second search, first search results are evaluated and used to refine the sample distributions. Search time can be set arbitrarily.

It achieves much higher performance than baseline.

### XGBoost with selected features
* R-squared: 0.54255 (top 50 features), 0.54388 (top 100 features)
* position: 2551, 2365
* Code file: `/code/xgb_topfeatures.py`

After finetuning the model hyperparameters, we start to look at the dataset itself. A few hundred of features sound like a lot so it might be a good idea to conduct feature reduction. Here the feature importance for XGBoost are used to select the top 50 and 100 features for training and submission. However, it didn't boost the performance. Maybe other ways are more feasible, like PCA, ICA, etc., but we didn't try those here.

### XGBoost: 2-layer ensemble
* R-squared: 0.53630
* position: 2969
* Code file: `/code/xgb_ensemble.py`

Ensembling is a very popular technique to achieve great performance in most competitions, so here I built a 2-layer ensemble with 4 XGBoost in the first layer and 1 XGBoost in the second for final prediction. Models are the top candidates in random search results. It didn't boost but drag down the performance. Maybe different models other than XGBoost should be used and layer size should be increased as well.

### 4 XGBoost averaging
* R-squared: 0.54959
* position: 1192
* Code file: `/code/xgb_average.py`

Ensembling doesn't work well, so I turned to averaging which is simpler. Models are also the top candidates in random search results. It works very well and is close to 0.55.

### 4 XGBoost averaging: add 'ID'
* R-squared: 0.55086
* position: 906
* Code file: `/code/xgb_average.py`

As I was looking for ways to further push the performance, I realized that the 'ID' variable isn't used in methods above. Normally it shouldn't be of much use and is regarded as an index, but I still tried to add it in dataset and it works like a charm. My guess is that 'ID' is related to testing process. Therefore, for the methods in the following, by default 'ID' is added.

### 4 XGBoost averaging: add 2 and 3-way interactions
* R-squared: 0.55168
* position: 485
* Code file: `/code/xgb_interactions.py`

A traditional feature engineering technique is to add interactions into dataset. Since there are 387 features, it's unrealistic to add all the interactions, so top 8 features from feature importance are chosen to produce 2 and 3-way interactions. It gives a significant boost to R-squared.

### 8 XGBoost averaging
* R-squared: 0.55335
* position: 38 (silver)
* Code file: `/code/xgb_interactions.py`

Finally, random search is conducted one more time to gather more models to form a 8-XGBoost averaging. By default, 'ID' variable is added, and 2/3-way interactions are added as well.