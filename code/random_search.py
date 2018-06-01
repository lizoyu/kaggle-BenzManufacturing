import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

# read training set
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
y = train['y']
train = train.drop(columns=['y'])

# convert categorical to one-hot encoding 
nrow = train.shape[0]
X = pd.concat([train,test], axis=0, ignore_index=True)
for colname in X.columns[1:9]:
	X[colname] = LabelEncoder().fit_transform(X[colname])
X = pd.concat([pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(X.iloc[:,1:9])), X.iloc[:,9:]], axis=1)
X_train = X.iloc[:nrow,:]
X_test = X.iloc[nrow:,:]

# fine-tuning: random search
best_score = 0
d = {}
while True:
	# sample hyperparameters
	'''
	max_depth = np.random.randint(2, 7) # sample from [2,6](int)
	learning_rate = np.random.uniform(0, 1) # sample from [0,1]
	n_estimators = np.random.randint(100, 1000) # sample from[100,1000](int)
	booster = np.random.choice(['gbtree','gblinear','dart']) # choose from three boosters
	gamma = np.random.uniform(0, 1) # sample from [0,1](float)
	min_child_weight = np.random.randint(1, 5) # sample from [1,4](int)
	subsample = np.random.uniform(0.5, 1) # sample from [0.5,1](float)
	colsample_bytree = np.random.uniform(0, 1) # sample from [0,1](float)
	colsample_bylevel = np.random.uniform(0, 1) # sample from [0,1](float)
	reg_alpha = np.random.uniform(0, 3) # sample from [0,3](float)
	reg_lambda = np.random.uniform(1, 3) # sample from [1,3](float)
	'''

	# refined search
	max_depth = np.random.randint(2, 5) # sample from [2,4](int)
	learning_rate = np.random.uniform(0, 0.1) # sample from [0,0.1](float)
	n_estimators = np.random.randint(100, 1000) # sample from[100,1000](int)
	booster = np.random.choice(['gbtree','dart']) # choose from two boosters
	gamma = np.random.uniform(0, 1) # sample from [0,1](float)
	min_child_weight = np.random.randint(1, 5) # sample from [1,4](int)
	subsample = np.random.uniform(0.5, 1) # sample from [0.5,1](float)
	colsample_bytree = np.random.uniform(0, 1) # sample from [0,1](float)
	colsample_bylevel = np.random.uniform(0, 0.5) # sample from [0,0.5](float)
	reg_alpha = np.random.uniform(1, 3) # sample from [1,3](float)
	reg_lambda = np.random.uniform(2, 4) # sample from [2,4](float)

	par = tuple([max_depth, learning_rate, n_estimators, booster, gamma, min_child_weight,
				subsample, colsample_bytree, colsample_bylevel, reg_alpha, reg_lambda])
	if par in d:
		continue
	else:
		d[par] = 1
	X_train, y = shuffle(X_train, y)
	clf = xgb.XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
						   booster=booster, gamma=gamma, min_child_weight=min_child_weight,
						   subsample=subsample, colsample_bytree=colsample_bytree, colsample_bylevel=colsample_bylevel,
						   reg_alpha=reg_alpha, reg_lambda=reg_lambda)
	scores = cross_val_score(clf, X_train, y, cv=30, n_jobs=-1, scoring='r2')
	#if np.mean(scores) > best_score:
	if np.mean(scores) > 0.58:
		print('best scores:', np.mean(scores), clf)
		best_score = np.mean(scores)
