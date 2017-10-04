## example.py
## author: alexander new

'''In this example, we're training a cadre model on the boston housing dataset.
Information about the boston housing dataset can be found (e.g.)
here: https://medium.com/@haydar_ai/learning-data-science-day-9-linear-regression-on-boston-housing-dataset-cd62a80775ef
The regression task of this dataset is to predict the median value of occupied houses of a census tract
in Boston, MA.'''

import numpy as np
import scipy.stats as ss
import pandas as pd
import supCadresRegression as sc
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

#######################
## construct dataset ##
#######################

boston = load_boston()
D = pd.DataFrame(boston['data']).assign(target=boston['target']).sample(frac=1, random_state=662352).values

## train and test set sizes
Ntr = int(np.ceil(0.75*D.shape[0]))
Nte = D.shape[0] - Ntr
P = D.shape[1]-1

## train and test splits
Dtr, Dva = D[:Ntr,:], D[Ntr:,:]
Dva, Dtr = ss.zmap(Dva, Dtr), ss.zscore(Dtr)    
Xtr, Ytr = Dtr[:,:-1], np.expand_dims(Dtr[:,-1], 1)
Xva, Yva = Dva[:,:-1], np.expand_dims(Dva[:,-1], 1)

M = 3
alpha = [0.95, 0.05] # d is more l1, W is more l2
lam = [0.01, 0.01]

cadreModel = sc.learnCadreModel(Xtr, Ytr, Xva, Yva, M, alpha, lam, 1)

## plot true y values vs. predicted y values in the test set,
## colored by cadre membership
colors = ['blue','red','green']
for m in range(M):
    plt.scatter(Yva[np.where(cadreModel['mVa']==m)], 
                cadreModel['fVa'][np.where(cadreModel['mVa']==m)], 
                c=colors[m], label=m)
plt.legend(loc='upper left')
plt.show()

## plot distribution of cadre assignment weights
plt.plot(np.arange(P), cadreModel['d'])
plt.show()

## plot distribution of target prediction weights (i.e. linear models),
## colored by cadre
for m in range(M):
	plt.plot(np.arange(P), cadreModel['W'][:,m], c=colors[m], label=m)
plt.legend(loc='upper left')
plt.show()