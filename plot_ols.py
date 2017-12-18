#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
facebookData = pd.read_csv('energyEff.csv', sep = ';')
heldout = [0.95, 0.90, 0.75, 0.50, 0.01]

# Use only one feature
X = facebookData[facebookData.columns[1:2]].as_matrix()
y = facebookData['Y1']
rng = np.random.RandomState(42)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.75, random_state=rng)
#X_train = X_train.reshape(-1,1)
#X_test = X_test.reshape(-1,1)
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
print(regr.score(X_test,y_test))
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print(X_test.shape)
print(y_test.shape)
# Plot outputs
plt.scatter(X_test[:,0], y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()
