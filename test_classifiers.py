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

# C values for linear kernels
C_VALUES = [10**x for x in range(-3, 1)]
CVAL_STRING = ','.join(['%.3g' % x for x in C_VALUES])

# Gamma values for RBF kernels
G_VALUES = [10**x for x in range(-3, 1)]
GVAL_STRING = ','.join(['%.3g' % x for x in G_VALUES])

# Trees in random forest (RF, ET, GB models)
T_VALUES = [10**x for x in range(2, 4)]
TVAL_STRING = ','.join(['%d' % x for x in T_VALUES])

ALL_MODELS = 'ALL'


def classifier_factory(model_code, **args):
    """Given a model code, create a list of models for testing,
    and a corresponding list of model names and parameters.

    :param str model_code: single character representing an sklearn classifier type
    """
    cvalues = args.get('cvalues', C_VALUES)
    bootstrap = args.get('bootstrap', False)
    gvalues = args.get('gamma', G_VALUES)
    nprocs = args.get('nprocs', 1)
    depth = args.get('depth', 3)
    squal = args.get('splitqual', GINI_TYPE)
    trees = args.get('trees', T_VALUES)
    verbose = args.get('verbose', False)

    intTrees = [int(x) for x in trees]
    model_type = MODEL_NAME[model_code]

    names = []
    models = []
    if model_type == LINEAR_SVM:
        for cval in cvalues:
            names.append('Linear SVM (C=%.5g)' % cval)
            models.append(svm.SVC(kernel='linear', C=cval, probability=True))
    elif model_type == RBF_SVM:
        for gamma in gvalues:
            names.append('RBF SVM (gamma=%.5g, C=1)' % gamma)
            models.append(svm.SVC(kernel='rbf', C=1, gamma=gamma, probability=True))
    elif model_type == LOGREG:
        for cval in cvalues:
            names.append('Logistic regression (C=%.5g)' % cval)
            models.append(linear_model.LogisticRegression(C=cval))
    elif model_type == RANDOM_FOREST:
        for p in intTrees:
            if bootstrap:
                names.append('Random forest (N=%d, %s, bootstrap)' % (p, squal))
            else:
                names.append('Random forest (N=%d, %s)' % (p, squal))
            models.append(ensemble.RandomForestClassifier(n_estimators=p,
                                                          bootstrap=bootstrap,
                                                          criterion=squal,
                                                          random_state=1,
                                                          n_jobs=nprocs))
    elif model_type == EXTRA_TREE:
        for p in intTrees:
            if bootstrap:
                names.append('Extra trees (N=%d, %s, bootstrap)' % (p, squal))
            else:
                names.append('Extra trees (N=%d, %s)' % (p, squal))
            models.append(ensemble.ExtraTreesClassifier(n_estimators=p,
                                                        bootstrap=bootstrap,
                                                        criterion=squal,
                                                        random_state=1,
                                                        n_jobs=nprocs))
    elif model_type == ADABOOST:
        for p in intTrees:
            names.append('Adaboost (N=%d)' % p)
            models.append(ensemble.AdaBoostClassifier(n_estimators=p, random_state=1))
    elif model_type == GRADIENT_BOOST:
        for p in intTrees:      # number of boosting stages
            names.append('Gradient boosting (N=%d)' % p)
            models.append(ensemble.GradientBoostingClassifier(n_estimators=p, max_depth=depth, subsample=0.5))
    elif model_type == NAIVE_BAYES:
        names.append('Naive Bayes')
        models.append(naive_bayes.GaussianNB())
    elif model_type == DECISION_TREE:
        names.append('Decision Tree')
        models.append(tree.DecisionTreeClassifier(random_state=1))
    else:
        raise ValueError('Unrecognized model type: {}'.format(model_type))

    return names, models


def run_iteration(iterId):
    """"Multiprocessing-friendly method that runs a single iteration
    of cross-validation for the current global model.  NB: iteration
    number indexes into the random number seed for this iteration."""
    return cross_validation(MODEL, DATA.features, DATA.labels, nfolds=opts.nfolds, shuffle=True, random_state=seeds[iterId])


def roc_name(s):
    """Converts a model name into an output file name, for ROC curves."""
    # e.g., 'Random forest (N=1000, gini, bootstrap)'
    result = s.replace(' ', '_')
    result = result.replace('.', 'p')
    for c in '(,=)':
        result = result.replace(c, '')
    return result + '_roc.csv'


def set_seed(x):
    # init seed so that performance can be reproduced
    random.seed(x)
    numpy.random.seed(x)


USAGE = """%prog CSV-file [options]

Run tests on a set of classification algorithms to determine which one performs
best on a given data set."""

# Establish command-line options:
parser = OptionParser(usage=USAGE)
parser.add_option('-i', dest='niter', default=5,
                  help='# iterations per classifier [default: %default]', type='int')
parser.add_option('-n', dest='nfolds', default=5,
                  help='# folds for CV [default: %default]', type='int')
modelHelp = 'Codes of models to use: %s, or ALL for all of them' % \
            ', '.join(['%s=%s' % (k, MODEL_NAME[k]) for k in MODEL_CODES])
parser.add_option('-m', dest='models', default=DEFAULT_CODE,
                  help=modelHelp + ' [default: %default]')
parser.add_option('-p', dest='nprocs', default=1,
                  help='# processors to use [default: %default]', type='int')
parser.add_option('-S', dest='std', default=False,
                  help='Standardize data [default: %default]', action='store_true')
parser.add_option('-v', dest='verbose', default=False,
                  help='Verbose mode [default: %default]', action='store_true')

svmopt = OptionGroup(parser, 'SVMs')
svmopt.add_option('-C', dest='cvals', default=CVAL_STRING,
                  help='CSV list of C-values (SVMs) [default: %default]')
svmopt.add_option('-G', dest='gamma', default=GVAL_STRING,
                  help='CSV list of gamma values (RBF kernels only) [default: %default]')
parser.add_option_group(svmopt)

rfopt = OptionGroup(parser, 'Tree-based parameters')
rfopt.add_option('-T', dest='trees', default=TVAL_STRING,
                 help='CSV list of forest sizes [default: %default]')
rfopt.add_option('-b', dest='bootstrap', default=False,
                 help='Use bootstrapping [default: %default]', action='store_true')
rfopt.add_option('--squal', dest='squal', default='gini', help='Criterion (gini/entropy) [default: %default]')
rfopt.add_option('--roc', dest='roc', default=False,
                 help='Write ROC files for each model [default: %default]', action='store_true')
rfopt.add_option('--depth', dest='depth', default=3,
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

mCodes = set(opts.models.upper())
invalid = mCodes - set(MODEL_CODES)
if invalid:
    parser.print_help()
    sys.stderr.write('\nInvalid model codes: %s\n' % ','.join(invalid))
    sys.exit(1)

if opts.nprocs > multiprocessing.cpu_count():
    sys.stderr.write('You asked for %d processors, but there are only %d available.\n' %
                     (opts.nprocs, multiprocessing.cpu_count()))
    sys.exit(1)

csvFile = args[0]
validate_file(csvFile)

# Global values for multiprocessing:
NFOLDS = opts.nfolds

cValues = [float(x) for x in opts.cvals.split(',')]
gValues = [float(x) for x in opts.gamma.split(',')]
tValues = [int(x) for x in opts.trees.split(',')]

set_seed(1)
seeds = [random.randint(0, 1000) for r in range(opts.niter)]

DATA = load_csv(csvFile)

if opts.verbose:
    sys.stderr.write('Data: %s\n' % DATA)
    sys.stderr.write('Model codes: %s\n' % ','.join(mCodes))

# set the method(s) to test ...
Names = []
models = []
for c in mCodes:
    n, m = classifier_factory(c, cvalues=cValues, gamma=gValues, trees=tValues,
                              bootstrap=opts.bootstrap,
                              splitqual=opts.squal,
                              depth=opts.depth,
                              nprocs=opts.nprocs,
                              verbose=opts.verbose)
    if opts.verbose:
        sys.stderr.write('adding model %s\n' % n)
    Names.extend(n)
    models.extend(m)

if opts.verbose:
    sys.stderr.write('Testing the following models:\n')
    for e in Names:
        sys.stderr.write('  %s\n' % e)

Scores = []
Stdev = []
for k in range(len(Names)):
    name = Names[k]
    MODEL = models[k]
    if opts.verbose:
        sys.stderr.write('%s\n' % time_string(name))

    meanScores = []
    iterRange = range(opts.niter)
    if opts.nprocs > 1:
        MP = multiprocessing.Pool(processes=opts.nprocs)
        results = MP.map(run_iteration, iterRange)
        MP.terminate()
    else:
        results = [r for r in map(run_iteration, iterRange)]

    bestIndex = 0
    bestScore = 0.0
    for j in iterRange:
        score = numpy.mean(results[j].accuracy())
        meanScores.append(score)
        if score > bestScore:
            bestIndex = j
            bestScore = score

    if opts.roc:
        results[bestIndex].writeROC(roc_name(name))

    Scores.append(numpy.mean(meanScores))
    Stdev.append(numpy.std(meanScores))

    if opts.verbose:
        sys.stderr.write('%s %.3f\n' % (name, Scores[-1]))

if opts.verbose:
    sys.stderr.write(time_string('--------\nFinished\n'))

print('Final rankings for %s:' % os.path.basename(csvFile))
print('\n%-40s\t%11s\t%11s' % ('Model', 'Avg.', 'Stdev.'))
ranking = numpy.argsort(Scores)
for i in ranking[::-1]:
    print('%-40s\t%10.5f\t%10.5f' % (Names[i], Scores[i], Stdev[i]))
