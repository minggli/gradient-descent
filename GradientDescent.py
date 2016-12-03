import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

__author__ = 'Ming Li'


class GradientDescent(object):

    """
    This is a standard Gradient Descent algorithm to optimise parameters for General Linear Models and Logistic Regression.
    Multi-class Logistic Regression is supported
    """

    def __init__(self, alpha=.1, max_epochs=5000, conv_thres=.0001, display=False):

        self._alpha = alpha    # learning rate
        self._max_epochs = max_epochs  # max number of iterations
        self._conv_thres = conv_thres    # convergence threshold
        self._display = display
        self._multi_class = False
        self._sigmoid = None
        self._linear = None
        self.params = None
        self.X = None
        self.y = None
        self.thetas = None
        self.costs = None

    def fit(self, model, X, y):

        self.X = np.array(X)
        self.y = np.array(y).reshape(len(y), 1)

        if isinstance(model, sm.OLS) or isinstance(model, linear_model.LinearRegression):
            self._linear = True
            if hasattr(model, 'coef_'):
                self.params = np.array(np.matrix(model.coef_))
            if hasattr(model, 'params'):
                self.params = np.array(np.matrix(model.params))

        if isinstance(model, linear_model.LogisticRegression):
            self._sigmoid = True
            if hasattr(model, 'coef_'):
                self.params = np.array(np.matrix(model.coef_))

            unique_classes = np.unique(self.y)
            n = len(unique_classes)
            if n < 2:
                raise ValueError("Optimiser needs samples of at least 2 classes"
                                 " in the data, but the data contains only one"
                                 " class: {0}".format(unique_classes[0]))
            if n == 2:
                self._multi_class = False
            else:
                self._multi_class = True

        return self

    def __partial_derivative_cost__(self, params, X, y):

        J = 0
        m = len(X)

        if self._linear:
            h = np.dot(X, params.T)     # GLM hypothesis in linear algebra representation

        if self._sigmoid:
            h = 1 / (1 + np.exp(-np.dot(X, params.T)))     # logistic (sigmoid) model hypothesis

        J = np.dot((h - y).T, X) / m        # partial_derivative terms for either linear or logistic regression

        return J  # J is a n-dimensioned vector

    def __cost_function__(self, params, X, y):

        J = 0
        m = len(X)

        if self._linear:
            h = np.dot(X, params.T)
            # GLM hypothesis in linear algebra representation
            J = (h - y) ** 2
            J /= (2 * m)

        if self._sigmoid:
            h = 1 / (1 + np.exp(-np.dot(X, params.T)))
            # logistic (sigmoid) model hypothesis
            J = - np.dot(np.log(h).T, y) - np.dot(np.log(1 - h).T, (1 - y))
            J /= m

        return np.sum(J)

    def __processing__(self, params, X, y):

        alpha = self._alpha

        count = 0  # initiating a count number so once reaching max iterations will terminate

        cost = self.__cost_function__(params, X, y)  # initial J(theta)
        prev_cost = cost + 10
        costs = [cost]
        # thetas = [params]

        if self._display:
            print('beginning gradient decent algorithm...')

        while (np.abs(prev_cost - cost) > self._conv_thres) and (count <= self._max_epochs):
            prev_cost = cost
            params -= alpha * self.__partial_derivative_cost__(params, X, y)  # gradient descend
            # thetas.append(params)  # restoring historic parameters
            cost = self.__cost_function__(params, X, y)  # cost at each iteration
            costs.append(cost)
            count += 1
            if self._display:
                print('iterations have been processed: {0}'.format(count))

        return params, costs

    def optimise(self):

        X = self.X
        y = self.y
        params = self.params

        if not self._multi_class:

            new_thetas, costs = self.__processing__(params, X, y)

            self.thetas = new_thetas
            self.costs = costs

        else:

            n_samples, n_features = X.shape
            unique_classes = np.unique(y)
            master_params = np.empty(shape=(1, n_features))
            master_costs = list()

            for k, _class in enumerate(unique_classes):

                _y = np.array(y == _class).astype(int)  # one versus rest method handling multi-nominal classification
                _params = np.matrix(params[k])

                new_thetas, costs = self.__processing__(_params, X, _y)

                master_costs.append(costs)
                master_params = np.append(master_params, np.array(_params), axis=0)

            self.thetas = master_params[1:]
            self.costs = master_costs
