import gzip
import numpy
import os
import random
import sklearn
import sys
import time

MODELS = [LINEAR_SVM, RBF_SVM, RANDOM_FOREST, NAIVE_BAYES, DECISION_TREE,
          EXTRA_TREE, ADABOOST, GRADIENT_BOOST, LOGREG] = \
         ['Linear SVM', 'RBF SVM', 'Random Forest', 'Naive Bayes', 'Decision Tree', 'Extra Trees', 'AdaBoost', 'Gradient Boosting', 'Logistic Regression']

MODEL_CODES = [LS_CODE, RS_CODE, RF_CODE, NB_CODE, DT_CODE, ET_CODE, AB_CODE, GB_CODE, LR_CODE] = 'LRFNDEAGO'
MODEL_NAME = dict(zip(MODEL_CODES, MODELS))
DEFAULT_CODE = 'G'
SVM_MODELS = 'LRO'
TREE_MODELS = 'FEAG'
EXTENDED_MODELS = 'FE'
NOPARAM_MODELS = 'ND'

# Reasonable default parameters for each type of model
DEFAULT_PARAM = {LS_CODE: 0.001,
                 RS_CODE: 1.0,
                 RF_CODE: 16,
                 NB_CODE: None,
                 DT_CODE: None,
                 ET_CODE: 16,
                 AB_CODE: 16,
                 GB_CODE: 16,
                 LR_CODE: 1.0,
                 }

# Binary labels:
NEG_LABEL = 0
POS_LABEL = 1

# Split quality types
SQ_TYPES = [GINI_TYPE, ENTROPY_TYPE] = ['gini', 'entropy']

# Optimization metrics (NB: not using AUC as it can't be pickled)
METRICS = [ACC_METRIC, PPV_METRIC, SENS_METRIC, SPEC_METRIC] = ['acc', 'ppv', 'sens', 'spec']
METRIC_OPTIONS = '/'.join(METRICS)


class ClassifierData(object):
    def __init__(self, **args):
        self.labels = []
        self.binary = []
        self.features = []
        self.scaler = None

        if ('labels' in args) and ('features' in args):
            self.setData(args['labels'], args['features'])
        elif ('labels' in args) or ('features' in args):
            raise ValueError('Both labels and features must be provided, or neither of them')

        if 'scaler' in args:
            self.scaler = args['scaler']

    def __len__(self):
        return len(self.features)

    def applyScaler(self, features):
        """Applies standardization (subtract mean, divide by SD)."""
        return self.scaler.transform(features)

    def clone(self):
        """Creates a duplicate of the current instance."""
        return ClassifierData(labels=list(self.labels), features=list(self.features), scaler=self.scaler)

    def computeBinaryLabels(self):
        """Creates a list of 0/1 labels regardless of the original labels."""
        self.binary = [binary_label(x) for x in self.labels]

    def createScaler(self):
        """Creates a standardising scaler (subtract mean, divide by SD) but does NOT apply it
        to the current instance.  Standardisation must be applied by clients before training/testing."""
        # transform features to zero mean and unit variance
        # NB: client methods must apply the scaler; here we only train it.
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(self.features)

    def merge(self, other):
        """Merges the data from another ClassifierData instance into this one."""
        if list(other.labels) != list(self.labels):
            raise ValueError('Invalid ClassifierData object: labels do not match current object')
        features = []
        for i in range(len(self.features)):
            fA = list(self.features[i])
            fB = list(other.features[i])
            features.append(fA+fB)
        self.features = numpy.asarray(features)
        self.createScaler()

    def __str__(self):
        return 'ClassifierData with %d features and %d labels' % (len(self.features[0]), len(set(self.labels)))

    def setData(self, labels, features):
        """Sets the label and feature lists for the instance."""
        if len(labels) != len(features):
            raise ValueError('Number of labels %d does not match number of feature vectors %d' %
                             (len(labels), len(features)))
        self.labels = labels
        self.features = features
        self.createScaler()
        self.computeBinaryLabels()


class Results(object):
    """Mini version of a results object similar to those in PyML but for sklearn CV methods."""
    def __init__(self):
        self.dfDict = {}
        self.labelDict = {}
        self.confDict = {}
        self.df = []
        self.labels = []
        self.threshold = None

    def add(self, foldId, df, labels):
        """Add a new fold result to the set of existing results."""
        if foldId in self.dfDict:
            raise ValueError('Fold %s already stored' % foldId)

        self.dfDict[foldId] = df
        self.labelDict[foldId] = labels
        self.confDict[foldId] = confusion_matrix(df, labels, threshold=self.threshold)

        self.updateDF()

    def addResult(self, foldId, other):
        """Add a new fold result to the set of existing results."""
        self.add(foldId, other.df, other.labels)

    def accuracy(self, verbose=False):
        """Compute the overall accuracy for the results represented in the instance."""
        return accuracy(self.df, self.labels, verbose=True, threshold=self.threshold)

    def accuracyDict(self):
        """Compute the current accuracy for the folds represented in the instance."""
        result = {}
        for k in self.dfDict.keys():
            result[k] = accuracy(self.dfDict[k], self.labelDict[k])
        return result

    def auc(self):
        """Compute the overall AUC score for the results represented in the instance."""
        return sklearn.metrics.roc_auc_score(self.labels, self.df)

    def aucDict(self):
        """Compute the AUC score for the folds represented in the instance."""
        result = {}
        for k in self.dfDict.keys():
            result[k] = sklearn.metrics.roc_auc_score(self.labelDict[k], self.dfDict[k])
        return result

    def compute(self, metric):
        """Factory single-source method for returning statistics.  NOTE: does not include
        AUC score since it relies on sklearn.metrics and cannot be pickled."""
        if metric not in METRICS:
            raise ValueError('Invalid metric %s not in %s' % (metric, METRIC_OPTIONS))
        if metric == ACC_METRIC:
            return self.accuracy()
        elif metric == PPV_METRIC:
            return self.ppv()
        elif metric == SENS_METRIC:
            return self.sensitivity()
        elif metric == SPEC_METRIC:
            return self.specificity()

    def confusion(self):
        """Returns the confusion matrix for this object."""
        return confusion_matrix(self.df, self.labels, threshold=self.threshold)

    def folds(self):
        """Return the set of folds represented in the results instance."""
        return self.dfDict.keys()

    def incrementResults(self, other):
        """Add a new fold result to the set of existing results."""
        if len(self.dfDict) == 0:
            foldId = 1
        else:
            foldId = max(self.dfDict.keys()) + 1
        self.add(foldId, other.df, other.labels)

    def __len__(self):
        return len(self.dfDict.keys())

    def matthews(self):
        """Compute the overall Matthews Correlation Coefficient for the results represented in the instance."""
        return matthews_correlation(self.df, self.labels)

    def ppv(self):
        """Compute the overall sensitivity for the results represented in the instance."""
        return ppv(self.df, self.labels, threshold=self.threshold)

    def resetThreshold(self):
        """Resets threshold to the default for confusion matrix statistics."""
        self.threshold = None

    def sensitivity(self):
        """Compute the overall sensitivity for the results represented in the instance."""
        return sensitivity(self.df, self.labels, threshold=self.threshold)

    def setThreshold(self, threshold):
        """Sets a threshold for confusion matrix statistics."""
        self.threshold = threshold

    def specificity(self):
        """Compute the overall specificity for the results represented in the instance."""
        return specificity(self.df, self.labels, threshold=self.threshold)

    def statisticsString(self):
        """Returns a string with all statistics for this instance."""
        return ','.join(['%s=%.5f' % (k, self.compute(k)) for k in METRICS])

    def __str__(self):
        return 'Results object (acc=%.4f, AUC=%.4f)' % (self.accuracy(), self.auc())

    def updateDF(self):
        """Updates results DF and labels."""
        self.df = []
        self.labels = []
        for k in self.dfDict.keys():
            self.df.extend(self.dfDict[k])
            self.labels.extend(self.labelDict[k])

    def writeROC(self, fileName):
        """Write df-label pairs to the given file."""
        rocStream = open(fileName, 'w')
        for k in self.dfDict.keys():
            for (x, y) in zip(self.dfDict[k], self.labelDict[k]):
                rocStream.write('%.8f,%d\n' % (x, y))
        rocStream.close()


def accuracy(df, given, verbose=False, threshold=None):
    """Computes balanced accuracy for a given TP, TN, FP and FN rate."""
    (TP, TN, FP, FN) = confusion_matrix(df, given, verbose=verbose, threshold=threshold)
    denom1 = TP + FN
    denom2 = TN + FP
    q1 = (0.5*TP)/denom1 if denom1 else 0.0
    q2 = (0.5*TN)/denom2 if denom2 else 0.0
    result = q1+q2
    return result


def binary_label(s):
    """Convenience method for assigning a value to string labels: -1 or 0 --> 0, 1 --> 1."""
    return int(int(s) > 0)


def confusion_matrix(df, given, verbose=False, threshold=None):
    """Computes components of confusion matrix: TP, TN, FP and FN results for a set of examples."""
    if len(df) != len(given):
        raise ValueError('Number of predictions must match number of training examples (%d != %d)' %
                         (len(df), len(given)))

    if len(given) == 0:
        return (0.0, 'empty')

    # Assumes values strictly in [0,1] are probabilities
    if threshold is not None:
        pass
    elif 0 <= min(df) and max(df) <= 1:
        threshold = 0.5
    else:
        threshold = 0.0

    P = [int(x > threshold) for x in df]
    G = [int(x > 0) for x in given]

    TP = TN = FP = FN = 0
    for i in range(len(df)):
        if P[i] == G[i]:
            if P[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if P[i] == 1:
                FP += 1
            else:
                FN += 1
    return (TP, TN, FP, FN)


def cross_validation(model, data, labels, nfolds=5, **args):
    """Runs cross-validation, creating a results object that permits access to a variety of CV statistics."""

    # Recent version of sklearn completely breaks this!  Rather than wading through sparse documentation
    # to reconfigure everything, and with the onerous possibility that it will change again later,
    # I wrote my own stinkin' method.  Not that I'm bitter...  ;-)
    # cvFolds = sklearn.model_selection.StratifiedKFold(labels, nfolds, **args)
    cvFolds = stratified_folds(labels, nfolds)

    test_results = Results()
    train_results = Results()
    ctr = 0
    # train, test are lists of array indexes
    for (train, test) in cvFolds:
        ctr += 1
        model.fit(data[train], labels[train])
        # training results
        train_df = df_factory(model, data[train])
        train_results.add(ctr, train_df, labels[train])
        # test results
        test_df = df_factory(model, data[test])
        test_results.add(ctr, test_df, labels[test])
    return (train_results, test_results)


def df_factory(model, data):
    """Factory method for obtaining a list of decision-function values."""
    # 'predict_proba' is available for random forest, extra trees,
    # and SVM when probability=True
    if hasattr(model, 'predict_proba'):
        class_probs = model.predict_proba(data)
        result = []
        for pair in class_probs:
            if type(pair) == numpy.float64:
                result.append(pair)
            else:
                result.append(pair[1])
        return result
    elif hasattr(model, 'decision_function'):
        # SVM, probability=False
        return model.decision_function(data)
    else:
        raise ValueError('Required function not available for %s!' % type(model))


def ezopen(file_name):
    """Allows clients to open files without regard for whether they're gzipped."""
    if not (os.path.exists(file_name) and os.path.isfile(file_name)):
        raise ValueError('file does not exist at %s' % file_name)

    handle = gzip.open(file_name, mode='rt')
    try:
        line = handle.readline()
        handle.close()
        return gzip.open(file_name, mode='rt')
    except Exception:
        return open(file_name)


def load_csv(csv_file):
    """Loads data from a CSV file into arrays for labels and features.
    Note: label must be the first column in each row."""
    labels = []
    features = []
    for line in ezopen(csv_file):
        parts = line.strip().split(',')
        labels.append(binary_label(parts[0]))
        vector = [float(x) for x in parts[1:]]
        features.append(vector)

    features = numpy.asarray(features)
    labels = numpy.asarray(labels)
    return ClassifierData(labels=labels, features=features)


def matthews_correlation(df, given, verbose=False, threshold=None):
    """Computes the Matthews Correlation Coefficient for a given TP, TN, FP and FN rate."""
    (TP, TN, FP, FN) = confusion_matrix(df, given, verbose=verbose, threshold=threshold)
    denom = (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
    if denom > 0: # numpy complains about sqrt(0)
        return float(TP*TN - FP*FN)/numpy.sqrt(denom)
    else:
        return 0.0


def ppv(df, given, threshold=None):
    """Computes positive predictive value (PPV) for a given TP and FP rate."""
    (TP, TN, FP, FN) = confusion_matrix(df, given, threshold=threshold)
    denom = TP + FP
    result = float(TP)/denom if denom else 0.0
    return result


def stratified_folds(labels, nfolds):
    """Returns pairs of index lists for each of the folds, stratified by labels.
    Assumes labels are positive/0 or positive/negative."""
    pos = []
    neg = []
    for i in range(len(labels)):
        if int(labels[i]) > 0:
            pos.append(i)
        else:
            neg.append(i)
    nPos = len(pos)
    nNeg = len(neg)

    if nPos < nfolds or nNeg < nfolds:
        raise ValueError('Insufficient positive (%d) or negative (%d) examples for %d-fold CV' % (nPos, nNeg, nfolds))

    pSize = float(nPos)/nfolds
    nSize = float(nNeg)/nfolds
    pSet = set(pos)
    nSet = set(neg)

    random.shuffle(pos)
    random.shuffle(neg)

    result = []
    pStart = nStart = 0
    for i in range(nfolds):
        pEnd = int(round((i+1)*pSize))
        nEnd = int(round((i+1)*nSize))
        pTest = set(pos[pStart:pEnd])
        nTest = set(neg[nStart:nEnd])
        pTrain = pSet - pTest
        nTrain = nSet - nTest
        test = sorted(pTest | nTest)
        train = sorted(pTrain | nTrain)
        result.append([train, test])
        pStart = pEnd
        nStart = nEnd

    return result


def sensitivity(df, given, threshold=None):
    """Computes sensitivity for a given TP, TN, FP and FN rate."""
    (TP, TN, FP, FN) = confusion_matrix(df, given, threshold=threshold)
    denom = TP + FN
    result = float(TP)/denom if denom else 0.0
    return result


def specificity(df, given, threshold=None):
    """Computes sensitivity for a given TP, TN, FP and FN rate."""
    (TP, TN, FP, FN) = confusion_matrix(df, given, threshold=threshold)
    denom = TN + FP
    result = float(TN)/denom if denom else 0.0
    return result


def time_string(s, format_string='%X', LF=False):
    """Returns the input string with user-readable a timestamp prefix."""
    timestamp = time.strftime(format_string, time.localtime())
    result = '%s %s' % (timestamp, s)
    if LF:
        result += '\n'
    return result


def validate_file(path):
    """Standard method for validating file paths."""
    if not path:
        raise Exception("'%s' is not a valid file path; exiting." % path)

    if not os.path.exists(path):
        raise Exception("File '%s' not found; exiting." % path)
