# supervised-cadres
Python code implementing the supervised cadre method for supervised learning.

This supervised cadre method (SCM) is for data analysis problems in which the population under study may be softly partitioned into a set of cadres. The cadres create clusters of observations based on only a few predictive features. Within these cadres, the behavior of the target variable is more simply modeled than it is on the population as a whole. We introduce a discriminative model that, when trained on a set of observations, simultaneously learns cadre assignment and target prediction rules. Our formulation allows sparse priors to be put on the model parameters. These priors allow for independent feature selection processes to be performed during both the cadre assignment and target prediction processes, which results in simple and interpretable ensemble models. Models are learned using adaptive stepsize stochastic gradient descent (Adam).

More about the SCM can be found at arXiv:1802.02500 and arXiv:1808.04880.

## Installation

This code was written assuming Python 3.6. To install it, I recommend creating a new virtual environment.

First clone the repo:

`git clone https://github.com/newalexander/supervised-cadres`

You can install all necessary dependencies with

`pip install -r requirements.txt`

To make sure everything is working, navigate to `examples` and run

`python short-example.py`

This generates a simple binary classification task and then trains an SCM to solve it.

## Different learning tasks

The SCM is a general learning paradigm for supervised learning and can be used to solve a variety of learning tasks. These include:

- Scalar regression: `regression.py`
- Multivariate regression: `regressionVec.py` (currently, only diagonal covariance matrices are supported)
- Binary classification: `classificationBinary.py` (either logistic loss (default) or hinge loss may be used)
- Multilabel classification: `classificationMulti.py` (only cross entropy loss is supported)
- Partial hazard analysis: `hazard.py`

## Interface basics

Each learning task has a different estimator object, with an interface based on those of `scikit-learn`. The major difference between a `supervised-cadres` estimator and a `scikit-learn` estimator is that, for `supervised-cadres` estimators, observations are supplied as `pandas.DataFrame` objects.

If `data` is a `pandas.DataFrame` object and `target` is the column-name of `data` giving the label, you can train a binary classification SCM with

    scm = binaryCadreModel()
    scm.fit(data, target, progress=True)

The files in the `examples` folder contain more in-depth examples. If you have questions, please email me at `newa` at `rpi` dot `edu`.

## Hints and Tricks

The SCM learning problem is nonconvex, and it can be ill-conditioned. Thus, training an SCM can be a more finicky and arduous task than, say, a support vector machine. Here are some helpful hints I've picked up.

Data Preparation:
- Continuous features should generally be standardized before training (with, e.g., `scipy.stats.zscore`)
- For scalar and multivariate regression, target columns should also be standardized
- Categorical features should be expanded into binary dummy variables (with, e.g., `pd.get_dummies`)
- If you only have binary features, you don't need to standardize them
- If you have a mixture of binary and continuous features, it is probably best to standardize all of them
- For binary classification, the `target` column should take values of either 0 or 1
- For multilabel classification, the `target` column should take values of `0, 1, ..., L-1`, where `L` is the number of classes

Hyperparameters:
- If the training process keeps returning `nan` values for loss, the most likely reasons are that either your features haven't been standardized, or the `gamma` cadre assignment sharpness hyperparameter is too large
- I have found that the default `gamma = 10` works well when the number of nonzero features an observation has is in the tens
- If an observation typically has hundreds or more nonzero features, you may need to decrease `gamma` to `gamma = 1` or `gamma = 0.1`
- The most important hyperparameter to tune is the number of cadres `M`
- It is best to supply a validation set `Dva` during training so you can monitor on-the-fly for overfitting or underfitting
- You can mitigate overfitting and underfitting by increasing and decreasing, respectively, the `lambda_d` and `lambda_W` hyperparameters
- The default `lambda_d` and `lambda_W` values seem to work best for fairly small and fairly noisy datasets. If your dataset is larger and less noisy, you may want to decrease them by a factor of ten.
