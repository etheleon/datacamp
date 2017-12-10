# XGBoost originally written in C++
# Speed and performance (ther core algorithm is parallelizable both GPU & CPU)

# Modified version of Datacamp's version which uses churn data
# but in our case, we are going to use the iris dataset

import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#R datasets
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
#all rows, 1st column counting from the back
r['iris'].iloc[:,-1].head()
#all rows, until 1st column counting from the back.
r['iris'].iloc[:,:-1].head()

X_train, X_test, y_train, y_test = train_test_split(r['iris'].iloc[:,:-1], r['iris'].iloc[:,-1], test_size=0.2, random_state=123)

# xgb.XGBClassifier(objective="binary:logistic", n_estimators=10, seed=123)
xg_cl = xgb.XGBClassifier(
    objective="multi:softmax", #the type of learning objective (this case is multiclass)
    n_estimators=10, #number of boosted trees (the equivalent in R is the nrounds argument)
    seed=123) #seed

xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test)/ y_test.shape[0])

# seems too ez, lets try another


