# supervised-cadres
Python code implementing the supervised cadre method for supervised learning. The model functions should work in either Python 2 or 3.

This supervised cadre method (SCM) is for data analysis problems in which the population under study may be softly partitioned into a set of cadres. The cadres create clusters of observations based on only a few predictive features. Within these cadres, the behavior of the target variable is more simply modeled than it is on the population as a whole. We introduce a discriminative model that, when trained on a set of observations, simultaneously learns cadre assignment and target prediction rules. Our formulation allows sparse priors to be put on the model parameters. These priors allow for independent feature selection processes to be performed during both the cadre assignment and target prediction processes, which results in simple and interpretable ensemble models. Models are learned using adaptive stepsize stochastic gradient descent.

The mathematical formalism of the supervised cadre model is developed at arXiv:1802.02500.

There are three classes of problem that cadre models can be applied to:
- Regression problems, where the goal is to predict a continuous target
- Binary classification problems, where the goal is to predict a binary target
- Binary risk analysis problems. In this problem, the goal is to assess how use a feature or set of features is for the task of predicting a binary target.

The regression SCM has the same interface as a `scikit-learn` estimator. This means it is compatible with `sklearn`'s hyperparameter tuning functions. So given:
- training observations `X`
- training target values `Y`

You train a cadre model with:
`cadreModel = sc.regressionCadreModel()`
`cadreModel.fit(Xtr, Ytr)`

The files in the `examples` folder contain more in-depth examples. The classification and risk analysis cadre models use an out-of-date and more complicated training interface.
