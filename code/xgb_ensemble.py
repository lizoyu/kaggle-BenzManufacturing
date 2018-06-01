import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

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

# 2-layer ensemble: 4 xgb -> 1 xgb -> prediction
xgb1 = xgb.XGBRegressor(base_score=0.5, booster='dart',
		colsample_bylevel=0.3759249886447098,
		colsample_bytree=0.5573748510206393, gamma=0.21417554627305047,
		learning_rate=0.0188450863417009, max_delta_step=0, max_depth=2,
		min_child_weight=1, missing=None, n_estimators=748, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.377191196440059, reg_lambda=2.042430749107093,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.668888078447673)
xgb2 = xgb.XGBRegressor(base_score=0.5, booster='dart',
		colsample_bylevel=0.27713794078765863,
		colsample_bytree=0.8743239284783383, gamma=0.9216043335260169,
		learning_rate=0.024556139699148827, max_delta_step=0, max_depth=3,
		min_child_weight=1, missing=None, n_estimators=514, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.5465636324593404, reg_lambda=2.3593765884892033,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.7704262629175511)
xgb3 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.26666344534762054,
		colsample_bytree=0.988461982162237, gamma=0.8672686544089706,
		learning_rate=0.01820894112110727, max_delta_step=0, max_depth=2,
		min_child_weight=2, missing=None, n_estimators=410, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.6996246467056564, reg_lambda=2.6614929583987603,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9905287003900232)
xgb4 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.3476398569954531,
		colsample_bytree=0.2731010282737144, gamma=0.4801938206633828,
		learning_rate=0.03993424498060205, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=217, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.1594018167318003, reg_lambda=2.1411176207654368,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9636596044220748)
xgb5 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.3476398569954531,
		colsample_bytree=0.2731010282737144, gamma=0.4801938206633828,
		learning_rate=0.03993424498060205, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=217, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.1594018167318003, reg_lambda=2.1411176207654368,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9636596044220748)

layer_1 = [xgb1, xgb2, xgb3, xgb4]
layer_2 = [xgb5]

output_1 = []
for clf in layer_1:
	clf.fit(X_train, y)
	output_1.append(clf.predict(X_train))

output_1 = pd.DataFrame(output_1).transpose()
xgb5.fit(output_1, y)

# predict and form submission
pred = xgb5.predict(pd.DataFrame([clf.predict(X_test) for clf in layer_1]).transpose())
sub = pd.concat([test['ID'], pd.DataFrame(pred, columns=['y'])], axis=1)
sub.to_csv('../submission/sub_ensemble.csv', index=False)