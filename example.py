import numpy as np

import matplotlib.pyplot as plt
from GradientDescent import GradientDescent
from sklearn.datasets import load_iris
from sklearn import (metrics, model_selection, pipeline, preprocessing,
                     linear_model)

np.random.seed(0)


def iris_visualisation(iris):

    X = iris.data  # we only take the first two features.
    Y = iris.target

    space = .5

    x_min, x_max = X[:, 0].min() - space, X[:, 0].max() + space
    y_min, y_max = X[:, 1].min() - space, X[:, 1].max() + space

    plt.figure(1, figsize=(12, 6))

    # Plot the training points
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel(iris['feature_names'][0])
    plt.ylabel(iris['feature_names'][1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks()
    plt.yticks()

    x_min, x_max = X[:, 2].min() - space, X[:, 2].max() + space
    y_min, y_max = X[:, 3].min() - space, X[:, 3].max() + space

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 2], X[:, 3], c=Y, cmap=plt.cm.Paired)
    plt.xlabel(iris['feature_names'][2])
    plt.ylabel(iris['feature_names'][3])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks()
    plt.yticks()

    plt.show()


def scaler_wrapper(estimator):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
    wrapper_estimator = pipeline.make_pipeline(scaler, estimator)
    return wrapper_estimator


def initiate_weights(estimator, mean=0, stddev=1):
    norm_dist = np.random.normal(loc=mean, scale=stddev, size=1000)
    boolean_index = abs(norm_dist - mean) < stddev * 2
    truncated_norm_dist = norm_dist[boolean_index]
    shape = estimator.coef_.shape
    sample = np.random.choice(truncated_norm_dist, size=np.prod(shape))
    return sample.reshape(shape)


data = load_iris()

# iris_visualisation(data)

feature = data['data']
label = data['target']

# logistic regression
lr = linear_model.LogisticRegression(fit_intercept=False,
                                     multi_class='multinomial',
                                     solver='newton-cg')

# add constant to feature set
constantfeature = np.hstack((np.ones([feature.shape[0], 1]), feature))
# Gradient Descent to fit model
lr.fit(constantfeature, label)
# initiate weights
lr.coef_ = initiate_weights(lr)
print(lr.coef_)

optimiser = GradientDescent(alpha=1e-1, max_epochs=1e5, conv_thres=1e-6,
                            display=False)
optimiser.fit(lr, constantfeature, label).optimise()
new_parameters = optimiser.thetas
lr.coef_ = new_parameters
print(lr.coef_)
y_pred = lr.predict(constantfeature)
f1_macro = metrics.f1_score(label, y_pred, average='macro')
accuracy = metrics.accuracy_score(label, y_pred)
print('Logistic Regression performance using Gradient Descent:\n'
      'F1 (macro): {:.4f}\nAccuracy: {:.4f}'.format(
        f1_macro.mean(), accuracy.mean()))


# cross-validation with KFold
kfold = model_selection.StratifiedKFold(n_splits=4, shuffle=True)
f1_macro = model_selection.cross_val_score(estimator=lr,
                                           X=constantfeature,
                                           y=label,
                                           scoring='f1_macro',
                                           cv=kfold)
accuracy = model_selection.cross_val_score(estimator=lr,
                                           X=constantfeature,
                                           y=label,
                                           scoring='accuracy',
                                           cv=kfold)

print('Logistic Regression CV performance:\n'
      'F1 (macro): {:.4f}\nAccuracy: {:.4f}'.format(
        f1_macro.mean(), accuracy.mean()))
