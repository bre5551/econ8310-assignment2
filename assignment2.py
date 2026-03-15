import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

X = train.drop(['meal', 'id', 'DateTime'], axis=1)
Y = train['meal']

Xt = test.drop(['meal', 'id', 'DateTime'], axis=1)
Yt = test['meal']

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=100, max_depth=75, learning_rate=0.5)
modelFit = model.fit(X, Y)
pred = modelFit.predict(Xt)

pred = [int(i) for i in pred]