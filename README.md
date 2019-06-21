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
