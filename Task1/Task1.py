# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:58:47 2023

@author: madde
"""
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
col_name=['sepal lenght','sepal width','petal lenght','petal width','class']
dataset=pd.read_csv("E:/LetsGrowMore/Task1/iris.csv",names=col_name)
sns.violinplot(y='class',x='sepal lenght', data=dataset, inner='quartile')
plt.show()
sns.violinplot(y='class',x='sepal width',data=dataset,inner='quartile')
plt.show()
sns.violinplot(y='class',x='petal lenght',data=dataset,inner='quartile')
plt.show()
sns.violinplot(y='class',x='petal width',data=dataset,inner='quartile')
plt.show()
X=dataset.drop(['class'],axis=1)
y=dataset['class']
print(f'X shape: {X.shape} | y shape: {y.shape} ')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
models = []
models.append(('SVC', SVC(gamma='auto')))
results = []
model_names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  model_names.append(name)
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
  model = SVC(gamma='auto')
model.fit(X_train, y_train)
prediction = model.predict(X_test)
print(f'Test Accuracy: {accuracy_score(y_test, prediction)}')
print(f'Classification Report: \n {classification_report(y_test, prediction)}')
