"""
Gradient Descent

This is a standard Gradient Descent algorithm to numerically optimise
Generalized Linear Model (GLM) and Logistic Regression (incl. multi-nominal).

"""

import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

__author__ = 'Ming Li'


class GradientDescent(object):

    def __init__(self,
                 alpha=.1,
                 max_epochs=5000,
                 conv_thres=float(1e-4),
                 display=False):

        self._alpha = alpha    # learning rate
        self._max_epochs = max_epochs  # max number of iterations
        self._conv_thres = conv_thres    # convergence threshold
        self._display = display
        self._sigmoid = None
        self._linear = None
        self.thetas = None
        self.costs = None

    @property
    def _params(self):
        return self.params

    @_params.setter
    def _params(self, value):
        self.params = value
        if self._linear:
            # GLM hypothesis in algebratic representation
            self._h = np.dot(self.X, self.params.T)
        elif self._sigmoid:
            # sigmoid hypothesis algebratic representation
            self._h = 1 / (1 + np.exp(-np.dot(self.X, self.params.T)))

    def fit(self, model, X, y):

        self.X = np.array(X)
        self.y = np.array(y).reshape(len(y), 1)

        if isinstance(model, sm.OLS) or \
                isinstance(model, linear_model.LinearRegression):
            self._linear = True
            if hasattr(model, 'coef_'):
                self._params = np.array(np.matrix(model.coef_))
            if hasattr(model, 'params'):
                self._params = np.array(np.matrix(model.params))

        if isinstance(model, linear_model.LogisticRegression):
            self._sigmoid = True
            if hasattr(model, 'coef_'):
                self._params = np.array(np.matrix(model.coef_))

            unique_classes = np.unique(self.y)
            n = len(unique_classes)
            if n < 2:
                raise ValueError("Optimiser needs at least 2 classes"
                                 " in the data, but the data contains only one"
                                 " class: {0}".format(unique_classes[0]))
            if n == 2:
                self._multi_class = False
            else:
                self._multi_class = True

        return self

    def __partial_derivative__(self):
        """partial derivative terms in vector representation"""

        # partial derivative for cost function of either linear or logistic
        # regression, only difference is the hypothesis which depends on model.
        # d is a n-dimensioned vector [num_samples, 1].

        return np.dot((self._h - self.y).T, self.X).mean()

    def __cost_function__(self):
        """cost function where params [1, num_features]
        X, y of shape [num_samples, num_features], X includes intercept already
        """

        if self._linear:
            # h produces column vector of shape [num_samples, 1]
            # GLM cost function is mean squared error over 2.
            J = ((self._h - self.y) ** 2).mean() / 2

        if self._sigmoid:
            # h produces column vector of real numbers between (0, 1) of shape
            # [num_samples, 1]
            # logistic (sigmoid) cost function, y is gronud truth of 0 or 1
            J = (-np.dot(self.y.T, np.log(self._h))-np.dot((1 - self.y).T,
                 np.log(1 - self._h))).mean()

        return np.sum(J)

    def __processing__(self):

        # initiating a count number
        count = 0

        cost = self.__cost_function__()  # initial J(theta)
        prev_cost = cost + 10
        costs = [cost]
        # thetas = [params]

        if self._display:
            print('beginning gradient decent algorithm...')

        while np.abs(prev_cost - cost) > self._conv_thres and \
                count <= self._max_epochs and \
                cost >= 0:

            prev_cost = cost

            self._params -= self._alpha * self.__partial_derivative__()

            # cost at each iteration
            cost = self.__cost_function__()
            costs.append(cost)
            count += 1
            if self._display:
                print(cost)
                print('number of iterations processed: {0}'.format(count))

        return costs

    def optimise(self):

        if not self._multi_class:

            self.costs = self.__processing__()
            self.thetas = self.params
            return self.thetas, self.costs

        else:

            n_samples, n_features = self.X.shape
            unique_classes = np.unique(self.y)
            master_params = np.empty(shape=(1, n_features))
            master_costs = list()

            for k, _class in enumerate(unique_classes):
                # one versus rest method handling multi-nominal classification
                _y = np.array(self.y == _class).astype(int)
                _params = np.matrix(self.params[k])

                costs = self.__processing__()

                master_costs.append(costs)
                master_params = \
                    np.append(master_params, np.array(_params), axis=0)

            self.thetas = master_params[1:]
            self.costs = master_costs[0]

        return self.thetas, self.costs
