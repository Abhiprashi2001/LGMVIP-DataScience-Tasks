# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:19:26 2023

@author: madde
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report,confusion_matrix
dataset=pd.read_csv("E:/LetsGrowMore/Task2/Iris.csv")
dataset.head(10)
dataset.sample(10)
dataset.shape
dataset.columns
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset['Species'].unique()
dataset['Species'].value_counts()
sns.pairplot(dataset,hue='Species')
fig, (ax1,ax2)=plt.subplots(ncols=2, figsize=(16, 5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=dataset,hue='Species',ax=ax1, s=300, marker='o')
sns.scatterplot(x='SepalWidthCm', y='PetalWidthCm',data=dataset, hue='Species',ax=ax2, s=300, marker='o')
sns.violinplot(y='Species',x='SepalLengthCm',data=dataset,inner='quartile')
plt.show()
sns.violinplot(y='Species',x='SepalWidthCm',data=dataset,inner='quartile')
plt.show()
sns.violinplot(y='Species',x='PetalLengthCm',data=dataset,inner='quartile')
plt.show()
sns.violinplot(y='Species',x='PetalWidthCm',data=dataset,inner='quartile')
plt.show()
colors=['#66b3ff','#ff9999','green']
dataset['Species'].value_counts().plot(kind='pie',autopct='%1.1f%%',shadow=True,colors=colors,explode=[0.08,0.08,0.08])
plt.figure(figsize=(7,5))
sns.heatmap(dataset.corr(),annot=True,cmap='CMRmap')
plt.show()
features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X=dataset.loc[:,features].values
y=dataset.Species
X_train,X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
decisiontree=DecisionTreeClassifier()
decisiontree.fit(X_train,y_train)
y_pred=decisiontree.predict(X_test)
y_pred
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)
def report(model):
    preds=model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model, X_test,y_test,cmap='nipy_spectral',colorbar=True)
    print('Decision Tree Classifier:')
report(decisiontree)
print(f'Accuracy: {round(score*100, 2)}%')
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
