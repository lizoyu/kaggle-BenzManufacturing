import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# read data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
y = train['y']
train = train.drop(columns=['y'])

# convert categorical to one-hot encoding (with ID)
nrow = train.shape[0]
X = pd.concat([train,test], axis=0, ignore_index=True)
for colname in X.columns[1:9]:
	X[colname] = LabelEncoder().fit_transform(X[colname])
X = pd.concat([X['ID'], pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(X.iloc[:,1:9])), X.iloc[:,9:]], axis=1)
X_train = X.iloc[:nrow,:]
X_test = X.iloc[nrow:,:]

# 8-xgb averaging
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
xgb5 = xgb.XGBRegressor(base_score=0.5, booster='dart',
		colsample_bylevel=0.28660272296089867,
		colsample_bytree=0.7113678451377421, gamma=0.5297843639180689,
		learning_rate=0.03460405839200247, max_delta_step=0, max_depth=3,
		min_child_weight=2, missing=None, n_estimators=231, n_jobs=1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.764492665822777, reg_lambda=3.6684869335937735,
		scale_pos_weight=1, seed=None, silent=True,
       subsample=0.6234944286270437)
xgb6 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.4219320035464771,
		colsample_bytree=0.43257358387366185, gamma=0.16403390261927087,
		learning_rate=0.008810711401328164, max_delta_step=0, max_depth=4,
		min_child_weight=1, missing=None, n_estimators=955, n_jobs=1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.4657831442088178, reg_lambda=3.4255635649914753,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.7325220099645795)
xgb7 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.2476294228065425,
		colsample_bytree=0.6118503666095605, gamma=0.633521832569079,
		learning_rate=0.025154883191656742, max_delta_step=0, max_depth=3,
		min_child_weight=1, missing=None, n_estimators=844, n_jobs=1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.6863931877235654, reg_lambda=2.1281940079374695,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9475400621897251)
xgb8 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.39045885548320935,
		colsample_bytree=0.9819689238317887, gamma=0.6141781021705763,
		learning_rate=0.015565940001830925, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=338, n_jobs=1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.8521132423043207, reg_lambda=3.077937694995847,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.6974832289213175)

layer_1 = [xgb1, xgb2, xgb3, xgb4, xgb5, xgb6, xgb7, xgb8]

for clf in layer_1:
	clf.fit(X_train, y)

# predict and form submission
pred = np.mean(np.stack([clf.predict(X_test) for clf in layer_1]), axis=0)
sub = pd.concat([test['ID'], pd.DataFrame(pred, columns=['y'])], axis=1)
sub.to_csv('../submission/sub_average.csv', index=False)
