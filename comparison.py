#! /usr/bin/env python
#
# Extended from code by Gaël Varoquaux and Andreas Müller
#   https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
# This script provides a comparison across ten different kinds of classifiers
# using random two-dimensional data.
#
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

DEFAULT_RANDOM_SEED = 1

# Classifier names establish ordering for what follows:
CLASSIFIERS = [KNN, LINEAR_SVM, RBF_SVM, GAUSSIAN_PROCESS, DECISION_TREE, RANDOM_FOREST,
         NEURAL_NET, ADABOOST, NAIVE_BAYES, QUADRATIC_DISCRIMINANT] = \
        ["k-NN", "Linear SVM", "RBF SVM", "Gauss. Proc.", "Dec. Tree", "RF",
         "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
MODEL_CODES = 'KLRPDFNABQ'

MODEL_NAME = {'K': KNN,
              'L': LINEAR_SVM,
              'R': RBF_SVM,
              'P': GAUSSIAN_PROCESS,
              'D': DECISION_TREE,
              'F': RANDOM_FOREST,
              'N': NEURAL_NET,
              'A': ADABOOST,
              'B': NAIVE_BAYES,
              'Q': QUADRATIC_DISCRIMINANT,
              }


def model_factory(model_name, args):
    """Return a model of the appropriate type for the given model name."""
    if model_name == KNN:
        return KNeighborsClassifier(args.k)
    elif model_name == LINEAR_SVM:
        return SVC(kernel="linear", C=args.linear_c)
    elif model_name == RBF_SVM:
        return SVC(gamma=args.gamma, C=1)
    elif model_name == GAUSSIAN_PROCESS:
        return GaussianProcessClassifier(1.0 * RBF(args.gp_length))
    elif model_name == DECISION_TREE:
        return DecisionTreeClassifier(max_depth=args.dt_depth)
    elif model_name == RANDOM_FOREST:
        return RandomForestClassifier(max_depth=args.rf_depth, n_estimators=10, max_features=1)
    elif model_name == NEURAL_NET:
        return MLPClassifier(alpha=1, max_iter=1000)
    elif model_name == ADABOOST:
        return AdaBoostClassifier()
    elif model_name == NAIVE_BAYES:
        return GaussianNB()
    elif model_name == QUADRATIC_DISCRIMINANT:
        return QuadraticDiscriminantAnalysis()
    else:
        raise ValueError('Unrecognized model name {}'.format(model_name))

DESCRIPTION = """
This script provides a comparison across ten different kinds of classifiers using random two-dimensional data.

Possible models are:
    K = [K]-nearest-neighbors
    L = [L]inear SVM 
    R = [R]BF SVM 
    P = Gaussian [P]rocess 
    D = [D]ecision tree 
    F = Random [F]orest 
    N = [N]eural network 
    A = [A]daBoost 
    B = Naive [B]ayes 
    Q = [Q]uadratic discriminant
"""

MODEL_HELP = 'K=k-NN/L=Lin. SVM/R=RBF SVM/P=Gauss./D=D-tree/F=Rand. Forest/N=Neural net/A=AdaBoost/B=Naive Bayes/Q=QDA'

# Hack to use two formatters to get argparse to show description and help default values nicely:
class CombinedFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=CombinedFormatter)
parser.add_argument('-d', dest='dt_depth', help='Decision tree depth', type=int, default=5)
parser.add_argument('-k', help='k-NN parameter', type=int, default=3)
parser.add_argument('-m', dest='model_codes', help='Models to compare ' + MODEL_HELP, default=MODEL_CODES)
parser.add_argument('-n', help='Number of samples in dataset', type=int, default=100)
parser.add_argument('-r', dest='rf_depth', help='Random forest depth', type=int, default=5)
parser.add_argument('-s', dest='seed', help='Random seed', type=int, default=None)
parser.add_argument('-G', dest='gamma', help='RBF SVM gamma value', type=float, default=2.0)
parser.add_argument('-L', dest='linear_c', help='Linear SVM soft margin parameter (C-value)', type=float, default=0.025)
parser.add_argument('-P', dest='gp_length', help='Gaussian process RBF length parameter', type=float, default=1.0)
parser.add_argument('-v', help='verbose mode', action='store_true')

args = parser.parse_args()

model_codes = args.model_codes.upper()
for code in model_codes:
    if code not in MODEL_CODES:
        parser.print_help()
        print('\n** Unrecognized model symbol {}.  Valid codes are {}'.format(code, MODEL_CODES))
        sys.exit(1)

# Instantiate a model for each classifier type
selected_models = [MODEL_NAME[c] for c in model_codes]
classifiers = {}
for name in selected_models:
    classifiers[name] = model_factory(name, args)

# Generate linearly separable dataset:
X, y = make_classification(n_samples=args.n, n_features=2, n_redundant=0, n_informative=2, random_state=args.seed, n_clusters_per_class=1)
rng = np.random.RandomState(args.seed)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# Create a list of datasets with the linearly separable data plus two others:
ds_names = [LINEAR, MOONS, CIRCLES] = ['Linearly separable', 'Overlapping crescents', 'Concentric circles']
datasets = {LINEAR: linearly_separable,
            MOONS: make_moons(n_samples=args.n, noise=0.3, random_state=args.seed),
            CIRCLES: make_circles(n_samples=args.n, noise=0.2, factor=0.5, random_state=args.seed),
            }

rcParams['font.weight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'

full_width = (1 + len(selected_models)) * 2
figure = plt.figure(figsize=(full_width, 9))
mesh_stepsize = 0.02

# matplotlib plot index ranges from 1 (upper-left) to (# datasets x # classifiers+1) (lower-right)
# with default settings, this is 1-30
plot_index = 1

# Iterate over datasets
dataset_index = 0
for ds_name in ds_names:
    dataset_index += 1
    dataset = datasets[ds_name]
    if args.v:
        print('Dataset {}/{}: {} data'.format(dataset_index, len(ds_names), ds_name))

    # Preprocess dataset
    X, y = dataset
    X = StandardScaler().fit_transform(X)

    # Set plotting boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=args.seed)

    # Plot the dataset on the left
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, plot_index)
    if dataset_index == 1:
        ax.set_title("Input data")
    ax.set_ylabel(ds_name)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')

    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')

    # Establish plot image data for all axes
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize),
                         np.arange(y_min, y_max, mesh_stepsize))

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    plot_index += 1

    # Iterate over classifiers
    for name in selected_models:
        if args.v:
            print('  {}'.format(name))

        clf = classifiers[name]
        ax = plt.subplot(len(datasets), len(classifiers) + 1, plot_index)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if dataset_index == 1:
            ax.set_title(name)

        ax.text(xx.max() - 0.3, yy.min() + 0.3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
        plot_index += 1

plt.tight_layout()
plt.show()
