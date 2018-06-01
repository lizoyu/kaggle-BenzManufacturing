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

# create two-way and three-way interactions
feat = ['X127', 'X315', 'X314', 'X261', 'X47', 'X316', 'X267', 'X118']
interactions = pd.DataFrame()

# two-way interactions
for i in range(len(feat)-1):
	for j in range(i+1):
		interactions[feat[i]+'_'+feat[j]] = X[feat[i]].map(str) + X[feat[j]].map(str)
# three-way interactions
for i in range(len(feat)-2):
	for j in range(i+1, len(feat)-1):
		for k in range(j+1):
			interactions[feat[i]+'_'+feat[j]+'_'+feat[k]] = X[feat[i]].map(str) + X[feat[j]].map(str) + X[feat[k]].map(str)

# convert to one-hot encoding
for colname in interactions.columns:
	interactions[colname] = LabelEncoder().fit_transform(interactions[colname])
X = np.hstack((X.values, OneHotEncoder(sparse=False).fit_transform(interactions.values)))

X_train = X[:nrow,:]
X_test = X[nrow:,:]
# 8-xgb averaging
xgb1 = xgb.XGBRegressor(base_score=0.5, booster='dart',
		colsample_bylevel=0.28660272296089867,
		colsample_bytree=0.7113678451377421, gamma=0.5297843639180689,
		learning_rate=0.03460405839200247, max_delta_step=0, max_depth=3,
		min_child_weight=2, missing=None, n_estimators=231, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.764492665822777, reg_lambda=3.6684869335937735,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.6234944286270437)
xgb2 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.4219320035464771,
		colsample_bytree=0.43257358387366185, gamma=0.16403390261927087,
		learning_rate=0.008810711401328164, max_delta_step=0, max_depth=4,
		min_child_weight=1, missing=None, n_estimators=955, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.4657831442088178, reg_lambda=3.4255635649914753,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.7325220099645795)
xgb3 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.2476294228065425,
		colsample_bytree=0.6118503666095605, gamma=0.633521832569079,
		learning_rate=0.025154883191656742, max_delta_step=0, max_depth=3,
		min_child_weight=1, missing=None, n_estimators=844, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.6863931877235654, reg_lambda=2.1281940079374695,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9475400621897251)
xgb4 = xgb.XGBRegressor(base_score=0.5, booster='dart',
		colsample_bylevel=0.4383243374934337,
		colsample_bytree=0.4022432163731485, gamma=0.9883604178055854,
		learning_rate=0.03518038395117225, max_delta_step=0, max_depth=2,
		min_child_weight=1, missing=None, n_estimators=258, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.3529363350065147, reg_lambda=3.678809169347793,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.7396725400299815)
xgb5 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.39045885548320935,
		colsample_bytree=0.9819689238317887, gamma=0.6141781021705763,
		learning_rate=0.015565940001830925, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=338, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.8521132423043207, reg_lambda=3.077937694995847,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.6974832289213175)
xgb6 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.40347514908046694,
		colsample_bytree=0.8469637847708145, gamma=0.9154301772336731,
		learning_rate=0.013389488732065536, max_delta_step=0, max_depth=3,
		min_child_weight=1, missing=None, n_estimators=535, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.762940115821464, reg_lambda=2.7305364872328965,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9185287904453702)
xgb7 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.30995509037575025,
		colsample_bytree=0.5452227172266677, gamma=0.02036859092993426,
		learning_rate=0.010921429170096964, max_delta_step=0, max_depth=4,
		min_child_weight=4, missing=None, n_estimators=637, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=1.7933862412931567, reg_lambda=3.401577037334149,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.8730960899237689)
xgb8 = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
		colsample_bylevel=0.3476398569954531,
		colsample_bytree=0.2731010282737144, gamma=0.4801938206633828,
		learning_rate=0.03993424498060205, max_delta_step=0, max_depth=2,
		min_child_weight=3, missing=None, n_estimators=217, n_jobs=-1,
		nthread=None, objective='reg:linear', random_state=0,
		reg_alpha=2.1594018167318003, reg_lambda=2.1411176207654368,
		scale_pos_weight=1, seed=None, silent=True,
		subsample=0.9636596044220748)

layer_1 = [xgb1, xgb2, xgb3, xgb4, xgb5, xgb6, xgb7, xgb8]

for clf in layer_1:
	clf.fit(X_train, y)

# predict and form submission
pred = np.mean(np.stack([clf.predict(X_test) for clf in layer_1]), axis=0)
sub = pd.concat([test['ID'], pd.DataFrame(pred, columns=['y'])], axis=1)
sub.to_csv('../submission/sub_interactions.csv', index=False)
