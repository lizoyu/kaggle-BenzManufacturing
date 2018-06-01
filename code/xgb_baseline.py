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
clf = xgb.XGBRegressor()
clf.fit(X_train, y)

# predict and form submission
pred = clf.predict(X_test)
sub = pd.concat([test['ID'], pd.DataFrame(pred, columns=['y'])], axis=1)
sub.to_csv('../submission/sub_baseline.csv', index=False)