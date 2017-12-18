"""
===================================================================
Support Vector Regression (SVR) using linear and non-linear kernels
===================================================================

Toy example of 1D regression using linear, polynomial and RBF kernels.

"""
print(__doc__)

import pandas as pd
import numpy as np
import timeit
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# #############################################################################
# Generate sample data
facebookData = pd.read_csv('energyEff.csv', sep = ';')
X = facebookData[facebookData.columns[0:1]].as_matrix()
#X = X/100
y = facebookData['Y2']
rng = np.random.RandomState(42)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.75, random_state=rng)

# #############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
start = timeit.default_timer()
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
ir = IsotonicRegression()
rid = Ridge(alpha=1.0)
logreg = LogisticRegression(C=1e5)
print("Processing rbf")
startRbf = timeit.default_timer()
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
stopRbf = timeit.default_timer()
print("Processing lin")
startLin = timeit.default_timer()
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
stopLin = timeit.default_timer()
print("Processing isotonic")
startIso = timeit.default_timer()
y_iso = ir.fit(X_train[:,0], y_train).predict(X_test[:,0])
stopIso = timeit.default_timer()
print("Processing ridge")
startRid = timeit.default_timer()
y_rid = rid.fit(X_train, y_train).predict(X_test)
stopRid = timeit.default_timer()
#print("Processing poly")
#startPoly = timeit.default_timer()
#y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
#stopPoly = timeit.default_timer()
#print("Processing logistic")
#y_log = logreg.fit(X_train, y_train).predict(X_test)
stop = timeit.default_timer()
time = stop - start
timeRbf = stopRbf - startRbf
timeLin = stopLin - startLin
#timePoly = stopPoly - startPoly
timeIso = stopIso - startIso
timeRid = stopRid - startRid
print('%.6f seconds' % time)
# #############################################################################
#print('Logistic regression score: %.3f' % logreg.score(X_test,y_test))
print("Linear regression score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (svr_lin.score(X_test,y_test), mean_squared_error(y_test, y_lin),timeLin))
print("Radial basis function score: %.3f\n Mean squared error: %.4f\n Time: %.6f" % (svr_rbf.score(X_test,y_test), mean_squared_error(y_test, y_rbf),timeRbf))
#print('Polynomial score: %.3f\n Mean squared error: %.4f\n Time: %.6f' % (svr_poly.score(X_test,y_test), mean_squared_error(y_test, y_poly), timePoly))
print('Isotonic score: %.3f\n Mean squared error: %.4f\n Time: %.6f' % (ir.score(X_test[:,0],y_test), mean_squared_error(y_test, y_iso), timeIso))
print('Ridge score: %.3f\n Mean squared error: %.4f\n Time: %.6f' % (rid.score(X_test,y_test), mean_squared_error(y_test, y_rid), timeRid))


# Look at the results
lw = 2
plt.scatter(X_test[:,0], y_test, color='darkorange', label='data')
plt.plot(X_test, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X_test, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X_test, y_iso, color='cornflowerblue', lw=lw, label='Isotonic model')
plt.plot(X_test, y_rid, color='yellow', lw=lw, label='Ridge model')
#plt.plot(X_test, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
#plt.plot(X_test, y_log, color='red', lw=lw, label='Logistic regression')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
