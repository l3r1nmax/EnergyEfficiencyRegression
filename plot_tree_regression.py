"""
===================================================================
Decision Tree Regression
===================================================================

A 1D regression with decision tree.

The :ref:`decision trees <tree>` is
used to fit a sine curve with addition noisy observation. As a result, it
learns local linear regressions approximating the sine curve.

We can see that if the maximum depth of the tree (controlled by the
`max_depth` parameter) is set too high, the decision trees learn too fine
details of the training data and learn from the noise, i.e. they overfit.
"""
print(__doc__)

# Import the necessary modules and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import timeit

# Read a dataset
facebookData = pd.read_csv('energyEff.csv', sep = ';')
X = facebookData[facebookData.columns[3:4]].as_matrix()
#X = X/100
y = facebookData['Y1']
rng = np.random.RandomState(42)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.75, random_state=rng)

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=1)
regr_2 = DecisionTreeRegressor(max_depth=2)
regr_3 = DecisionTreeRegressor(max_depth=3)
regr_4 = DecisionTreeRegressor(max_depth=4)
regr_5 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)
regr_3.fit(X_train, y_train)
regr_4.fit(X_train, y_train)
regr_5.fit(X_train, y_train)

# Predict
start1 = timeit.default_timer()
y_1 = regr_1.predict(X_test)
stop1 = timeit.default_timer()
start2 = timeit.default_timer()
y_2 = regr_2.predict(X_test)
stop2 = timeit.default_timer()
start3 = timeit.default_timer()
y_3 = regr_3.predict(X_test)
stop3 = timeit.default_timer()
start4 = timeit.default_timer()
y_4 = regr_4.predict(X_test)
stop4 = timeit.default_timer()
start5 = timeit.default_timer()
y_5 = regr_5.predict(X_test)
stop5 = timeit.default_timer()
time1 = stop1 - start1
time2 = stop2 - start2
time3 = stop3 - start3
time4 = stop4 - start4
time5 = stop5 - start5
print("DecisionTree regression 1 score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (regr_1.score(X_test,y_test), mean_squared_error(y_test, y_1), time1))
print("DecisionTree regression 2 score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (regr_2.score(X_test,y_test), mean_squared_error(y_test, y_2), time2))
print("DecisionTree regression 3 score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (regr_3.score(X_test,y_test), mean_squared_error(y_test, y_3), time3))
print("DecisionTree regression 4 score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (regr_4.score(X_test,y_test), mean_squared_error(y_test, y_4), time4))
print("DecisionTree regression 5 score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (regr_5.score(X_test,y_test), mean_squared_error(y_test, y_5), time5))
# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=1", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=2", linewidth=2)
plt.plot(X_test, y_3, color="yellow", label="max_depth=3", linewidth=2)
plt.plot(X_test, y_3, color="red", label="max_depth=4", linewidth=2)
plt.plot(X_test, y_3, color="blue", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
