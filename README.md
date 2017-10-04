# supervised-cadres
Python code implementing the supervised cadre method for supervised learning. The model functions should work in either Python 2 or 3. The examples are written in Python 3 though.

This supervised cadre method is for regression problems in which the population under study may be softly partitioned into a set of cadres. The cadres create clusters of observations based on only a few predictive features. Within these cadres, the behavior of the target variable is more simply modeled than it is on the population as a whole. We introduce a discriminative model that, when trained on a set of observations, simultaneously learns cadre assignment and target prediction rules. Our formulation allows sparse priors to be put on the model parameters. These priors allow for independent feature selection processes to be performed during both the cadre assignment and target prediction processes, which results in simple and interpretable ensemble models. Models are learned using adaptive stepsize stochastic gradient descent.

The model-learning interface is straightforward but has a lot of arguments. Given:
- training observations `Xtr`
- training target values `Ytr`
- validation observations `Xva`
- validation target values `Yva`
- total number of cadres `M`
- elastic net mixing hyperparameters `alpha`
- regularization strengths `lam`
- a seed `seed`

You learn the model with\
`cadreModel = learnCadreModel(Xtr, Ytr, Xva, Yva, M, alpha, lam, seed)`

Then `cadreModel` is a `dict` object containing predicted target values for `Xtr` and  `Xva`, as well as model parameters, and some other useful information. There's also a prediction function `applytoObs()` for generating predictions from an existing cadre model.

The file `example.py` contains a more in-depth example.
