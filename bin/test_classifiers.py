#!/usr/bin/python
from optparse import OptionParser, OptionGroup
import multiprocessing
import numpy
import os
import random
import sys

from sklearn import *
from sklearn_utils import *

# -------------------------------------------------------------------------------------------------------------
# The scikit-learn implementation of some random forest based models causes issues with multiprocessing.
# This raises UserWarning messages whenever we test a plain random forest ('F') or an extra trees model ('E'),
# as follows:
#
#   "UserWarning: Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1"
#
# As a work-around/hack, we just ignore these warnings:
import warnings
warnings.filterwarnings('ignore')
# -------------------------------------------------------------------------------------------------------------

# Default C values for linear kernels
C_VALUES = [10**x for x in range(-3, 1)]
CVAL_STRING = ','.join(['%.3g' % x for x in C_VALUES])

# Default gamma values for RBF kernels
G_VALUES = [10**x for x in range(-3, 1)]
GVAL_STRING = ','.join(['%.3g' % x for x in G_VALUES])

# Default # trees in random forest (RF, ET, GB models)
T_VALUES = [2**x for x in range(1, 7)]
TVAL_STRING = ','.join(['%d' % x for x in T_VALUES])

ALL_MODELS = 'ALL'


def classifier_factory(model_code, **args):
    """Given a model code, create a list of models for testing,
    and a corresponding list of model names and parameters.

    :param str model_code: single character representing an sklearn classifier type
    """
    cvalues = args.get('cvalues', C_VALUES)
    gvalues = args.get('gamma', G_VALUES)
    nprocs = args.get('nprocs', multiprocessing.cpu_count())
    depth = args.get('depth', 1)
    tree_sizes = args.get('trees', T_VALUES)
    verbose = args.get('verbose', False)

    # String for model names
    model_type = MODEL_NAME[model_code]

    names = []
    models = []
    if model_type == LINEAR_SVM:
        for cval in cvalues:
            names.append('%s (C=%.5g)' % (model_type, cval))
            models.append(svm.SVC(kernel='linear', C=cval, probability=True))
    elif model_type == RBF_SVM:
        for gamma in gvalues:
            names.append('%s (gamma=%.5g)' % (model_type, gamma))
            models.append(svm.SVC(kernel='rbf', C=1, gamma=gamma, probability=True))
    elif model_type == LOGREG:
        for cval in cvalues:
            names.append('%s (C=%.5g)' % (model_type, cval))
            models.append(linear_model.LogisticRegression(C=cval))
    elif model_type == RANDOM_FOREST:
        for p in tree_sizes:
            names.append('%s (N=%d)' % (model_type, p))
            models.append(ensemble.RandomForestClassifier(n_estimators=p,
                                                          random_state=1,
                                                          n_jobs=nprocs))
    elif model_type == EXTRA_TREE:
        for p in tree_sizes:
            names.append('%s (N=%d)' % (model_type, p))
            models.append(ensemble.ExtraTreesClassifier(n_estimators=p,
                                                        random_state=1,
                                                        n_jobs=nprocs))
    elif model_type == ADABOOST:
        for p in tree_sizes:
            names.append('%s (N=%d)' % (model_type, p))
            models.append(ensemble.AdaBoostClassifier(n_estimators=p, random_state=1))
    elif model_type == GRADIENT_BOOST:
        for p in tree_sizes:      # number of boosting stages
            names.append('%s (N=%d)' % (model_type, p))
            models.append(ensemble.GradientBoostingClassifier(n_estimators=p, max_depth=depth, subsample=0.5))
    elif model_type == NAIVE_BAYES:
        names.append(model_type)
        models.append(naive_bayes.GaussianNB())
    elif model_type == DECISION_TREE:
        names.append(model_type)
        models.append(tree.DecisionTreeClassifier(random_state=1))
    else:
        raise ValueError('Unrecognized model type: {}'.format(model_type))

    return names, models


def run_iteration():
    """"Multiprocessing-friendly method that runs a single iteration
    of cross-validation for the current global model."""
    return cross_validation(MODEL, DATA.features, DATA.labels, nfolds=opts.nfolds, shuffle=True)


def roc_name(s):
    """Converts a model name into an output file name, for ROC curves."""
    # e.g., 'Random forest (N=1000)'
    result = s.replace(' ', '_')
    result = result.replace('.', 'p')
    for c in '(,=)':
        result = result.replace(c, '')
    return result + '_roc.csv'


USAGE = """%prog CSV-file [options]

Run tests on a set of classification algorithms to determine which one performs
best on a given data set."""

# Establish command-line options:
parser = OptionParser(usage=USAGE)
parser.add_option('-n', dest='nfolds', default=5,
                  help='# folds for CV [default: %default]', type='int')
modelHelp = 'Codes of models to use: %s, or ALL for all of them' % \
            ', '.join(['%s=%s' % (k, MODEL_NAME[k]) for k in MODEL_CODES])
parser.add_option('-m', dest='models', default=DEFAULT_CODE,
                  help=modelHelp + ' [default: %default]')
parser.add_option('-S', dest='std', default=False,
                  help='Standardize data [default: %default]', action='store_true')
parser.add_option('-v', dest='verbose', default=False,
                  help='Verbose mode [default: %default]', action='store_true')
parser.add_option('--roc', dest='roc', default=False,
                 help='Write ROC files (score/label pairs) for each model [default: %default]', action='store_true')

svmopt = OptionGroup(parser, 'SVMs')
svmopt.add_option('-C', dest='cvals', default=CVAL_STRING,
                  help='comma-separated list of C-values [default: %default]')
svmopt.add_option('-G', dest='gamma', default=GVAL_STRING,
                  help='comma-separated list of gamma values (RBF kernel only) [default: %default]')
parser.add_option_group(svmopt)

rfopt = OptionGroup(parser, 'Tree-based classifiers')
rfopt.add_option('-T', dest='trees', default=TVAL_STRING,
                 help='CSV list of forest sizes [default: %default]')
rfopt.add_option('--depth', dest='depth', default=2,
                 help='Max. depth (gradient boosting only) [default: %default]', type='int')
parser.add_option_group(rfopt)

opts, args = parser.parse_args(sys.argv[1:])

MIN_ARGS = 1
if len(args) != MIN_ARGS:
    parser.print_help()
    if args:
        sys.stderr.write('\nExpected %d parameters; received %d:\n  %s\n' % (MIN_ARGS, len(args), '\n  '.join(args)))
    sys.exit(1)

if opts.models.upper() == ALL_MODELS:
    opts.models = MODEL_CODES

selected_models = opts.models.upper()
invalid = set(selected_models) - set(MODEL_CODES)
if invalid:
    parser.print_help()
    sys.stderr.write('\nInvalid model codes: %s\n' % ','.join(invalid))
    sys.exit(1)

csv_file = args[0]
validate_file(csv_file)

# Global values for multiprocessing:
NFOLDS = opts.nfolds

c_values = [float(x) for x in opts.cvals.split(',')]
gamma_values = [float(x) for x in opts.gamma.split(',')]
tree_sizes = [int(x) for x in opts.trees.split(',')]

DATA = load_csv(csv_file)

if opts.verbose:
    sys.stderr.write('Data: %s\n' % DATA)

# set the method(s) to test ...
names = {}
models = {}
for c in selected_models:
    names[c] = []
    models[c] = []
    # Factory returns a list of model names and a list of associated models
    n, m = classifier_factory(c, cvalues=c_values, gamma=gamma_values, trees=tree_sizes,
                              depth=opts.depth, verbose=opts.verbose)
    names[c].extend(n)
    models[c].extend(m)

if opts.verbose:
    sys.stderr.write('Testing the following models:\n')
    sys.stderr.write('  {}\n'.format(', '.join([MODEL_NAME[e] for e in names])))
    sys.stderr.flush()

train_accuracy = {}
accuracy = {}
bal_accuracy = {}
sensitivity = {}
specificity = {}
auc = {}
matthews = {}
ppv = {}
results = {}

# Iterate over each model type:
for c in selected_models:
    train_accuracy[c] = []
    accuracy[c] = []
    bal_accuracy[c] = []
    sensitivity[c] = []
    specificity[c] = []
    auc[c] = []
    matthews[c] = []
    ppv[c] = []
    results[c] = []

    if opts.verbose:
        sys.stderr.write('running {} on {} settings'.format(MODEL_NAME[c], len(names[c])))
        sys.stderr.flush()

    # Iterate over each parameter combination:
    for k in range(len(names[c])):
        name = names[c][k]
        MODEL = models[c][k]

        if opts.verbose:
            sys.stderr.write('.')
            sys.stderr.flush()

        train_result, test_result = run_iteration()

        # Training accuracy comes from the first half of each pair
        train_accuracy[c].append(train_result.accuracy())

        # Test statistics come from the second half of each pair
        accuracy[c].append(test_result.accuracy())
        bal_accuracy[c].append(test_result.balanced_accuracy())
        sensitivity[c].append(test_result.sensitivity())
        specificity[c].append(test_result.specificity())
        matthews[c].append(test_result.matthews())
        auc[c].append(test_result.auc())
        ppv[c].append(test_result.ppv())
        results[c].append(test_result)

    if opts.verbose:
        sys.stderr.write('\n')
        sys.stderr.flush()

best = []
for c in selected_models:
    # Sort models in descending order by balanced accuracy
    ranking = numpy.argsort(bal_accuracy[c])

    print('\n%-30s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' %
          (' ', 'TrnAcc', 'TstAcc', 'BalAcc', 'Sens.', 'Spec.', 'PPV', 'MCC', 'AUC'))

    for i in ranking[::-1]:
        print('%-30s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f' %
              (names[c][i], train_accuracy[c][i], accuracy[c][i], bal_accuracy[c][i], sensitivity[c][i], specificity[c][i], ppv[c][i], matthews[c][i], auc[c][i]))

    # save the best of each model:
    k = ranking[-1]
    stats_tuple = (names[c][k], train_accuracy[c][k], accuracy[c][k], bal_accuracy[c][k], sensitivity[c][k], specificity[c][k], ppv[c][k], matthews[c][k], auc[c][k])
    best.append(stats_tuple)

    if opts.roc :
        results[c][k].writeROC(roc_name(names[c][k]))


if len(selected_models) > 1:
    print('\nOverall rankings:')
    print('\n%-30s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s' %
          (' ', 'TrnAcc', 'TstAcc', 'BalAcc', 'Sens.', 'Spec.', 'PPV', 'MCC', 'AUC'))
    model_accuracy = [t[3] for t in best]
    ranking = numpy.argsort(model_accuracy)

    for i in ranking[::-1]:
        print('%-30s\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f' % best[i])

