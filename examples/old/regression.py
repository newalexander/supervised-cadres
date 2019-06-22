## regression.py
## author: alexander new

'''In this example, we're training a regression cadre model on the boston housing dataset.
Information about the boston housing dataset can be found (e.g.)
here: https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
The regression task of this dataset is to predict the median value of occupied houses of a census tract in Boston, MA.
Our cadre model is chosen so that the cadre-assignment mechanism will be very
sparse. We plot the distribution of cadre assignment and target prediction weights,
along with the cadre centers.
We also compare the predictive ability of a supervised cadre model with a linear SVR
and a gaussian kernel SVR. The cadre model should outperform the linear SVR and be
competitive with the nonlinear SVR, depending on vagaries of the train-validation split.
'''
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats as ss
import pandas as pd
import cadreModels.regression as sc
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR, SVR
import matplotlib.pyplot as plt

#######################
## construct dataset ##
#######################

boston = load_boston()
D = pd.DataFrame(boston['data']).assign(target=boston['target']).sample(frac=1, random_state=662352).values

## train and test set sizes
Ntr = int(np.ceil(0.75*D.shape[0]))
Nva = D.shape[0] - Ntr
P = D.shape[1]-1

## train and test splits
Dtr, Dva = D[:Ntr,:], D[Ntr:,:]
Dva, Dtr = ss.zmap(Dva, Dtr), ss.zscore(Dtr)    
Xtr, Ytr = Dtr[:,:-1], np.expand_dims(Dtr[:,-1], 1)
Xva, Yva = Dva[:,:-1], np.expand_dims(Dva[:,-1], 1)

M = 3                # number of cadres
alpha = [0.95, 0.05] # d is more l1, W is more l2
lam = [1, 1]         # regularization strength

##################
## learn models ##
##################

cadreModel = sc.regressionCadreModel(lambda_d=lam[0], lambda_W=lam[1], M=M)
cadreModel.fit(Xtr, Ytr)

## learn SVRs
lsvr = LinearSVR(epsilon=0.1, C=1)
lsvr.fit(Xtr, np.squeeze(Ytr))
ksvr = SVR(kernel='rbf', C=2)
ksvr.fit(Xtr, np.squeeze(Ytr))

######################
## evaluate results ##
######################

## apply cadre model to testing set
fTr = cadreModel.predict(Xtr)
fVa, Gva, mVa = cadreModel.predictFull(Xva)

## plot true y values vs. predicted y values in the test set,
## colored by cadre membership
colors = ['blue','red','green']
for m in range(M):
    plt.scatter(Yva[np.where(mVa==m)], 
                fVa[np.where(mVa==m)],
                c=colors[m], label='cadre_'+str(m))
plt.legend(loc='upper left')
plt.xlabel('true Y')
plt.ylabel('predicted Y')
plt.show()

## do the same for svr models
plt.scatter(Yva, lsvr.predict(Xva), color='red', label='lsvr')
plt.scatter(Yva, ksvr.predict(Xva), color='blue', label='ksvr')
plt.legend(loc='upper left')
plt.xlabel('true Y')
plt.ylabel('predicted Y')
plt.show()

## plot cadre-membership weights for each cadre
for m in range(M):
    plt.scatter(np.arange(Nva), 
                Gva[:,m],
                c=colors[m], label='cadre_'+str(m))
plt.legend(loc='upper left')
plt.xlabel('observation index')
plt.ylabel('cadre-membership weight')
plt.show()

## plot distribution of cadre assignment weights
plt.plot(np.arange(P), np.abs(cadreModel.d))
plt.xlabel('feature index')
plt.ylabel('cadre assignment weight')
plt.show()

## find out what features are used for cadre assignment
threshold = 0.001
print('features used for cadre assignment')
print(boston.feature_names[np.where(np.abs(cadreModel.d) > threshold)])
print('only', np.sum(np.abs(cadreModel.d) > threshold), 'out of', P, 'features are used for',
      'cadre assignment at a threshold of', threshold)

## plot distribution of cadre centers
for m in range(M):
    plt.plot(np.arange(P), cadreModel.C[:,m], c=colors[m], label='cadre_'+str(m))
plt.axhline(y=0, xmin=0, xmax=P, color='black', label='reference')
plt.legend(loc='upper left')
plt.xlabel('feature index')
plt.ylabel('cadre center')
plt.show()

## plot distribution of target prediction weights (i.e. linear models),
## colored by cadre
## include linear svr for comparison
for m in range(M):
    plt.plot(np.arange(P), cadreModel.W[:,m], c=colors[m], label='cadre_'+str(m))
plt.plot(np.arange(P), lsvr.coef_, c='orange', label='lsvr')
plt.axhline(y=0, xmin=0, xmax=P, color='black', label='reference')
plt.legend(loc='upper left')
plt.xlabel('feature index')
plt.ylabel('linear model weight')
plt.show()

## calculate train set and test set MSE for each model
print('supervised cadre training error, generalization error')
print(np.mean((np.squeeze(Ytr) - fTr)**2),
      np.mean((np.squeeze(Yva) - fVa)**2))
print('linear SVR training error, generalization error')
print(np.mean((np.squeeze(Ytr) - lsvr.predict(Xtr))**2),
      np.mean((np.squeeze(Yva) - lsvr.predict(Xva))**2))
print('nonlinear SVR training error, generalization error')
print(np.mean((np.squeeze(Ytr) - ksvr.predict(Xtr))**2),
      np.mean((np.squeeze(Yva) - ksvr.predict(Xva))**2))
