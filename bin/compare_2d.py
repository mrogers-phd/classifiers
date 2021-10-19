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

from comparison_utils import *

# Force argparse to show description as written and default values for help
class CombinedFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

DESCRIPTION = """
This script provides a comparison across ten different kinds of classifiers
using random two-dimensional data.

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

MODEL_HELP = 'K=k-NN; L=Lin. SVM; R=RBF SVM; P=Gauss.; D=D-tree; F=Rand. Forest; N=Neural net; A=AdaBoost; B=Naive Bayes; Q=QDA'

parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=CombinedFormatter)
parser.add_argument('-d', dest='dt_depth', help='Decision tree depth', type=int, default=5)
parser.add_argument('-e', dest='n_estimators', help='Number of estimators for random forest', type=int, default=10)
parser.add_argument('-k', help='k-NN parameter', type=int, default=3)
parser.add_argument('-m', dest='model_codes', help='Models to compare ' + MODEL_HELP, default=MODEL_CODES)
parser.add_argument('-n', help='Number of samples in dataset', type=int, default=100)
parser.add_argument('-o', dest='output', help='Render graphics to output file', default=None)
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
    classifiers[name] = model_factory(name,
                                      args.k,
                                      args.linear_c,
                                      args.gamma,
                                      args.gp_length,
                                      args.dt_depth,
                                      args.rf_depth,
                                      args.n_estimators)

# Generate three datasets for testing
datasets = generate_datasets(args.n, args.seed)

# Run the classifiers and render their results
compare_classifiers(datasets, classifiers, args.seed, args.output, args.v)
