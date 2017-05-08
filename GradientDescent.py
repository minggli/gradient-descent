"""
GradientDescent

This standard gradient descent implementation is to understand underlying
mathematics of Generalized Linear Model and Logistic Regression.

multi-class Logistic Regression is supported through OvR (One versus Rest)
"""

import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

__author__ = 'Ming Li'


class GradientDescent:

    def __init__(self,
                 alpha=1e-2,
                 max_epochs=1e5,
                 conv_thres=1e-6,
                 display=False):

        self._alpha = alpha
        self._max_epochs = max_epochs
        self._conv_thres = conv_thres
        self._display = display
        self._multiclass = False
        self._sigmoid = None
        self._linear = None

    def fit(self, model, X, y):

        self.X = np.array(X)
        self.m, self.n = self.X.shape
        self.y = np.array(y).reshape(self.m, -1)
        self.n_class = np.unique(self.y).shape[0]

        if isinstance(model, sm.OLS) or \
           isinstance(model, linear_model.LinearRegression):
            self._linear = True
            if hasattr(model, 'coef_'):
                self.params = model.coef_.reshape(-1, self.n)
            if hasattr(model, 'params'):
                self.params = np.array(model.params).reshape(-1, self.n)

        elif isinstance(model, linear_model.LogisticRegression):
            self._sigmoid = True
            if hasattr(model, 'coef_'):
                self.params = model.coef_.reshape(-1, self.n)

            if self.n_class < 2:
                raise Exception("Optimiser needs at least 2 classes"
                                " but only finds only one class.")

            self._multiclass = False if self.n_class == 2 else True
        else:
            raise Exception('model framework not supported: need either'
                            ' scikit-learn or statsmodels.')
        return self

    def _update_hypothesis(self, params):
        """recalculate h(x) during optimisation"""
        if self._linear:
            # GLM hypothesis in linear algebra representation, X includes
            # constant or intercept
            self.h = np.dot(self.X, params.T)

        if self._sigmoid:
            # logistic (sigmoid) hypothesis in linear algebra representation
            self.h = 1 / (1 + np.exp(-np.dot(self.X, params.T)))

    def _partial_derivative(self, params, y):
        """partial derivative terms for either linear or logistic regression
        albeit cosmetically the same, hypothesis is different for each.

        d is n-dimensioned vector.
        """
        self._update_hypothesis(params)
        d = np.dot((self.h - y).T, self.X) / self.m
        return d

    def _cost_function(self, params, y):
        """cost function or objective function to minimise"""
        # self._update_hypothesis(params)

        if self._linear:
            # GLM cost function in linear algebra representation: mean squared
            # error over 2
            # J = np.square(self.h - y) / self.m
            J = np.square(np.dot(self.X, params.T) - y) / self.m
            J /= 2

        if self._sigmoid:
            # original logistic (sigmoid) cost function:
            # J(θ) = -sum(y * log(hθ(x)) + (1 - y) * log(1 - hθ(x)))) / m
            # After plugging in sigmoid hypothesis and mathematical reasoning,
            # the equivalent is now simplified to:
            #          J(θ) = -sum(yθx - log(1 + exp(θx))) / m

            # original implementation:
            # J = - (y * np.log(self.h) + (1 - y) * np.log(1 - self.h)).mean()
            J = - (y * np.dot(self.X, params.T) -
                   np.log(1 + np.exp(np.dot(self.X, params.T)))).mean()

        return J

    def _process(self, params, y):
        """core operation to iteratively calculate partial derivative and
        update parameters and evaluate cost function"""
        # initial J of theta
        cost = self._cost_function(params, y)
        prev_cost = cost + 10
        costs = [cost]

        while (np.abs(prev_cost - cost) > self._conv_thres) and \
              (self.count <= self._max_epochs):
            prev_cost = cost

            params -= self._alpha * self._partial_derivative(params, y)

            cost = self._cost_function(params, y)
            costs.append(cost)

            if self._display:
                print('number iterations processed: {0:<10} '
                      'cost: {1:.6f}'.format(self.count, cost))
            self.count += 1

        return params, costs

    def optimise(self):
        """activate optmiser"""

        if not hasattr(self, 'X'):
            raise Exception('Fit the optimiser with model.')

        self.count = 0
        print('beginning gradient descend algorithm...')

        if not self._multiclass:

            new_thetas, costs = self._process(self.params, self.y)

            self.thetas = new_thetas
            self.costs = costs
            return self.thetas, self.costs

        elif self._multiclass:

            unique_classes = np.unique(self.y)
            master_params = list()
            master_costs = list()

            # one versus rest method handling multi-nominal optimisation
            for k, _class in enumerate(unique_classes):

                ovr_y = np.array(self.y == _class).astype(int)
                ovr_params = self.params[k].reshape(1, self.n)

                new_thetas, costs = self._process(ovr_params, ovr_y)

                master_costs.append(costs)
                master_params.append(new_thetas)

            self.thetas = np.array(master_params).reshape(self.n_class, self.n)
            self.costs = master_costs
            return self.thetas, self.costs
