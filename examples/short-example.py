## short-example.py
## short example analysis, just to make sure dependencies are installed correctly

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, '../cadreModels')

from classificationBinary import binaryCadreModel
from sklearn.datasets import make_classification
from scipy.stats import zscore, zmap

from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, random_state=2125615)

data = pd.DataFrame(X).assign(target=y)
features = data.columns[data.columns != 'target']

D_tr, D_va = train_test_split(data, test_size=0.2, random_state=313616)

D_va[features] = zmap(D_va[features], D_tr[features])
D_tr[features] = zscore(D_tr[features])

scm = binaryCadreModel(Tmax=1001, record=50)
scm.fit(D_tr, 'target', features, features, D_va, progress=True)
