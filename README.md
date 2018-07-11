# supervised-cadres
Python code implementing the supervised cadre method for supervised learning. The model functions should work in either Python 2 or 3.

This supervised cadre method (SCM) is for data analysis problems in which the population under study may be softly partitioned into a set of cadres. The cadres create clusters of observations based on only a few predictive features. Within these cadres, the behavior of the target variable is more simply modeled than it is on the population as a whole. We introduce a discriminative model that, when trained on a set of observations, simultaneously learns cadre assignment and target prediction rules. Our formulation allows sparse priors to be put on the model parameters. These priors allow for independent feature selection processes to be performed during both the cadre assignment and target prediction processes, which results in simple and interpretable ensemble models. Models are learned using adaptive stepsize stochastic gradient descent (Adam).

The mathematical formalism of the supervised cadre model is developed at arXiv:1802.02500. This also includes an in-depth case study of cadre model application and assessment.

To use these files, you'll need to have the SciPy ecosystem and TensorFlow installed.

In the `cadreModels` folder, there are four different types of cadre model, corresponding to different problem types:
- Scalar regression problems: `regression.py`
- Vector regression problems: `regressionVec.py`
- Binary classification problems: `riskModeling.py`
- Multilabel classification problems: `classification.py`

The regression SCM has the same interface as a `scikit-learn` estimator. This means it is compatible with `sklearn`'s hyperparameter tuning functions. 

Given:
- training observations `Xtr`
- training target values `Ytr`

You train a regression cadre model with:

    cadreModel = regressionCadreModel()
    cadreModel.fit(Xtr, Ytr)

The files in the `examples` folder contain more in-depth examples. If you have questions, please email me at `newa@rpi.edu`.
