import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# read data
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

# train
clf = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.3476398569954531,
		colsample_bytree=0.2731010282737144, gamma=0.4801938206633828,
		learning_rate=0.03993424498060205, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=217, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.1594018167318003, reg_lambda=2.1411176207654368,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9636596044220748)
clf.fit(X_train, y)

# predict and form submission
pred = clf.predict(X_test)
sub = pd.concat([test['ID'], pd.DataFrame(pred, columns=['y'])], axis=1)
sub.to_csv('../submission/sub_finetune.csv', index=False)