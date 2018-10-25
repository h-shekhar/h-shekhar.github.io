---
layout: post
title: Pima Indians onset of Diabetes dataset
subtitle: Exploratory Data Analysis
tags: [machine learning, python]
---

## Context:
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

## Content:
The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Objective:
Build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?

### Select & Load Data from CSV  

```python
#Load dataset
import pandas as pd
import matplotlib.pyplot as plt

url = './pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
print(data.shape) #review the dimension of dataset
print(data.head()) #look at first few rows
```
### Understanding Data with Descriptive Statistics  

```python
print(data.describe()) # review the distribution of data
print(data.dtypes) # Look at the datatypes of each attributes
print(data.corr()) #Calculate pairwise correlation between each variables
```
### Understanding Data with Visualization  

```python
from pandas.plotting import scatter_matrix
scatter_matrix(data)
plt.show()
```
![Plot](/img/2018/09/diabetes.png){:class="img-responsive"}

### Preprocessing Data using Standardization  

```python
# Standardize the data(mean=0, std-dev=1)
from sklearn.preprocessing import StandardScaler 
import numpy
array = data.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
```
### Algorithm Evaluation With Resampling Methods  
Estimate the accuracy of the Logistic Regression algorithm on dataset using 10-fold cross validation.

```python
# Evaluate using Cross-Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold)
acc=("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)
print(acc)
```
Accuracy: 76.951% (4.841%)
{: .box-note}

### Algorithm Evaluation Metrics  

```python
# Calculating the LogLoss metric on diabetes dataset.
# Cross Validation Classification LogLoss
scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
logloss=("Logloss: %.3f (%.3f)") % (results.mean(), results.std())
print(logloss)
```
Logloss: -0.493 (0.047)
{: .box-note}

### Spot-Check Algorithms  

```python
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print(results.mean())
```
-0.19634244702665754
{: .box-note}

### Model Comparison and Selection  
Compares Logistic Regression and Linear Discriminant Analysis to each other on diabetes dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = KFold(n_splits=10, random_state=7)
	cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
```
LR: 0.769515 (0.048411)
LDA: 0.773462 (0.051592)
{: .box-note}

### Improve Accuracy with Algorithm Tuning  

```python
# Tune the parameter of the algorithm using a grid search for the Ridge Regression algorithm on dataset.
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)
print(grid.best_score_)
print(grid.best_estimator_.alpha)
```
0.2796175593129722  
1.0
{: .box-note}

### Improve Accuracy with Ensemble Predictions  

```python
# We use the Random Forest algorithm (a bagged ensemble of decision trees) on the dataset.
from sklearn.ensemble import RandomForestClassifier
num_trees = 100
max_features = 3
kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
```
0.770745044429255
{: .box-note}

### Finalize And Save Your Model  

```python
# Save model using Pickle
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
url = "./pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
```
0.7559055118110236
{: .box-note}


_In case if you found something useful to add to this article or you found a bug in the code or would like to improve some points mentioned, feel free to write it down in the comments. Hope you found something useful here._
{: .box-warning}
