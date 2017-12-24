
# XGBoost

# XGBoost originally written in C++
# Speed and performance (ther core algorithm is parallelizable both GPU & CPU)

# Modified version of Datacamp's version which uses churn data
# but in our case, we are going to use the iris dataset

```{python}
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
```

```{python}
#Example with iris
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
print(f'This is the accuracy {accuracy}')
# seems too ez, lets try another dataset ltr
```

At the base of XGBoost are individual Decision Trees

```{python}
# DecisionTree
from sklearn.tree import DecisionTreeClassifier
dt_clf_4 = DecisionTreeClassifier(max_depth=4)
dt_clf_4.fit(X_train, y_train) #only if the X_train are all numbers
y_pred_4 = dt_clf_4.predict(X_test)
```

XGBoost works by combining many smaller trees

# How is boosting accomplished

1. Iteratively learning a set of weak models on subsets of the data
2. Weighing each weak prediction according to each weak learner's performance
3. Combine the weighted predictions to obtain a single weighted prediction
4. ... that is much better than the individual predictions themselves!

The cross validation API requires the data be fit into xgb's special format

Note the xgb.DMatrix takes `data` and `label` as parameters

```{python}
# Create the DMatrix: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)
# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix,
    params=params,
    nfold=3,
    num_boost_round=5,
    metrics="error", #coudl be auc
    as_pandas=True,
    seed=123)
# Print cv_results
print(cv_results)
```

Gives you this version

|  test-error-mean  | test-error-std  | train-error-mean |  train-error-std |
| --- | --- | --- | --- |
|          0.28378    |   0.001932     |     0.28232    |    0.002366 |
|          0.27190    |   0.001932     |     0.26951    |    0.001855 |
|          0.25798    |   0.003963     |     0.25605    |    0.003213 |
|          0.25434    |   0.003827     |     0.25090    |    0.001845 |
|          0.24852    |   0.000934     |     0.24654    |    0.001981 |

# Print the accuracy
```{python}
print(((1-cv_results["test-error-mean"]).iloc[-1]))
```

Linear base learners with "booster:gblearner" and the object function as "objective:reg:linear"
NOTE: Tree base learners "booster:gbtree"

```{python}
xg_reg = xgb.XGBRegressor(objective="reg:linear", n_estimators = 10, seed=123)
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
```

```{python}
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))
```

Now with cross validated

```{python}
cv_results = xgb.cv(
    dtrain=housing_dmatrix,
    params=params,
    nfold=4,
    num_boost_round=5,
    metrics="rmse",
    as_pandas=True, #returns as df
    seed=123)
```

Regularization

Loss functions

How complex a model is, penalising complexity is regularization

gamma (for tree based learners)
    gamma - minimum loss reduction allowed for a split to occur
    i.e. the higher the gamma, the less we allow the tree nodes to split

alpha - l1 regularization on leaf weights, larger values mean more regularization (tend towards 0)
lambda - l2 regularization on leaf weights

```{python
reg_params = [1, 10, 100]
params = {"objective":"reg:linear","max_depth":3}
# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []
# Iterate over reg_params
for reg in reg_params:
    params["lambda"] = reg
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix,
        params=params,
        nfold=2,
        num_boost_round=5,
        metrics="rmse",
        as_pandas=True,
        seed=123
)

    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

```

## Plotting

```{python eval=FALSE}
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)
# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}
# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)
# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()
# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()
# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(X,y)
# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}
# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params=params, num_boost_round=10)
# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()
```

## Tunning

1. number of trees, `num_rounds`
```{python}
num_rounds = [5, 10, 15]
final_rmse_per_round = []

for curr_num_rounds in num_rounds:
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))
```

2. Early Stopping

```{python}
# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix,
    params=params,
    early_stopping_rounds=10,
    num_boost_round=50,
    seed=123,
    as_pandas=True
)
```

3. ETA learning rate `params["eta"] = value`
4. max depth `params["max_depth"] = value`
5. colsample_bytree `params["colsample_bytree"] = value`
    * simply specifies the fraction of features to choose from at every split in a given tree

### Paramter Search

1. GridSearch
2. RandomSearch

```{python}
# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'n_estimators': [50],
        'max_depth': [2, 5]
        # 'max_depth': range(2, 12) #larger more suitable for random
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring="neg_mean_squared_error", cv=4, verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

gbm = xgb.XGBRegressor(n_estimators=10)
# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm, scoring="neg_mean_squared_error", n_iter=4, cv=4, verbose=1, param_distributions=gbm_param_grid)
# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
```

### Categorical Encoding

```{python}
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# Fill missing values with 0
df.LotFrontage = df.LotFrontage.fillna(0)
# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)
# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()
# Print the head of the categorical columns
print(df[categorical_columns].head())
# Create LabelEncoder object: le
le = LabelEncoder()
# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))
# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())

# Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(categorical_features=categorical_mask, sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])
df
# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)
```

can be simplified into

```{python}
# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5,:])

# Print the vocabulary
print(dv.vocabulary_)

```

## Pipelines

```{python}
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
                  ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)
``


```{python}
# Import necessary modules
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Fill LotFrontage missing values with 0
X.LotFrontage = X.LotFrontage.fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
                  ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict("records"), y, scoring="neg_mean_squared_error", cv=10)

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))

```

```{python}
# Import necessary modules
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
                                                [([numeric_feature], Imputer(strategy="median")) for numeric_feature in non_categorical_columns],
                                                input_df=True,
                                                df_out=True

)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
                                                    [(category_feature, CategoricalImputer()) for category_feature in categorical_columns],
                                                    input_df=True,
                                                    df_out=True

)

# Import FeatureUnion
from sklearn.pipeline import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion([
                                              ("num_mapper", numeric_imputation_mapper),
                                              ("cat_mapper", categorical_imputation_mapper)

])
# Create full pipeline
pipeline = Pipeline([
                         ("featureunion", numeric_categorical_union),
                         ("dictifier", Dictifier()),
                         ("vectorizer", DictVectorizer(sort=False)),
                         ("clf", xgb.XGBClassifier(max_depth=3))

])

# Perform cross-validation
cross_val_scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))

```

put everything together

```
# Create the parameter grid
gbm_param_grid = {
        'clf__learning_rate': np.arange(0.05, 1, 0.05),
        'clf__max_depth': np.arange(3, 10, 1),
        'clf__n_estimators': np.arange(50, 200, 50)

}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(pipeline,
                                            param_distributions=gbm_param_grid,
                                            scoring="roc_auc",
                                            n_iter=2,
                                            cv=2,
                                            verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)
```

