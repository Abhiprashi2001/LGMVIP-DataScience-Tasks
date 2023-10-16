# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:05:44 2023

@author: madde
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"
df = pd.read_csv(url)
print(df.head())
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
plt.figure(figsize=(16,8))
plt.title('Stock Price Over Time')
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.show()
n = 30
df['Prediction'] = df['Close'].shift(-n)
X = np.array(df.drop(['Prediction'], 1))
X = X[:-n]
y = np.array(df['Prediction'])
y = y[:-n]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')
plt.figure(figsize=(16,8))
plt.title('Stock Price Prediction')
plt.plot(df.index[-len(y_test):], y_test, label='Actual Prices')
plt.plot(df.index[-len(y_test):], predictions, label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.show()
