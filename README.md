# Classifier Sandbox
This repository contains python code for playing with machine learning classifiers.

## Dependencies
The scripts rely on two modules.  Software was confirmed to work on the versions indicated:
 - numpy (1.17.4)
 - matplotlib (3.1.2)
 - scikit-learn (0.23.1)

## 2-D dataset comparisons
The first script, `comparison.py`  uses a relatively simple 2-dimensional dataset to compare
the performance of different classifiers.  It was pulled from
[an example scikit-learn website](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
originally written by Gaël Varoquaux and Andreas Müller.
The script has been extended to include additional models;
to provide additional control over model parameters,
and to provide control over which models to compare.
There are default values for all models, so it is possible to try the script using simply:

```
$ comparison.py
```

The full set of options is available in the online help:
```
$ comparison.py -h
usage: comparison.py [-h] [-d DT_DEPTH] [-k K] [-m MODEL_CODES] [-n N] [-r RF_DEPTH] [-s SEED] [-G GAMMA] [-L LINEAR_C] [-P GP_LENGTH] [-v]

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

optional arguments:
  -h, --help      show this help message and exit
  -d DT_DEPTH     Decision tree depth (default: 5)
  -k K            k-NN parameter (default: 3)
  -m MODEL_CODES  Models to compare K=k-NN/L=Lin. SVM/R=RBF SVM/P=Gauss./D=D-tree/F=Rand. Forest/N=Neural net/A=AdaBoost/B=Naive Bayes/Q=QDA
                  (default: KLRPDFNABQ)
  -n N            Number of samples in dataset (default: 100)
  -r RF_DEPTH     Random forest depth (default: 5)
  -s SEED         Random seed (default: None)
  -G GAMMA        RBF SVM gamma value (default: 2.0)
  -L LINEAR_C     Linear SVM soft margin parameter (C-value) (default: 0.025)
  -P GP_LENGTH    Gaussian process RBF length parameter (default: 1.0)
  -v              verbose mode (default: False)
```


## High-dimensional dataset comparisons
A more realistic problem is represented in the datasets found in the `data` directory.
These files are related to the problem of discriminating between *pathogenic* and *benign*
single-nucleotide variants (SNVs) in the human genome.
SNVs are relatively rare single-nucleotide polymorphisms (SNPs), though what constitutes
"rare" changes from one publication to another.
Generally speaking, variants with an allele frequency below ~1% are considered SNVs,
while those with higher frequencies are considered SNPs.
For the data in our dataset, a cutoff of 1% was used.

Positive (pathogenic) examples here were derived from the
[Human Gene Mutation Database (HGMD)](http://www.hgmd.cf.ac.uk/ac/index.php),
while negative (presumed benign) examples were derived from the
[1000 Genomes Project](https://www.internationalgenome.org/).
(Note: these data were gathered back in 2014; since then, the HGMD has expanded considerably,
and other databases such as the
[Genome Aggregation Database (gnomAD)](https://gnomad.broadinstitute.org/)
may now provide better evidence for putative benign variants.)

### Conservation features
A key aspect of SNVs that facilitate classification is their impact in highly conserved or
weakly conserved regions of the genome.
A highly conserved region is one that changes very little from one species to another,
while a weakly conserved region is one that may change considerably between species, or may
disappear altogether.
Thus it seems intuitive that variants that appear within highly conserved regions are more
likely to cause problems (pathogenic) than those that appear in weakly conserved regions.
In fact, if we build a classifier using conservation-based features, we can achieve pretty good performance.

The `data` directory contains two files that contain measurements associated with several different
sequence conservation scoring methods:
 - `group_1_46-way.csv` represents measurements on a comparison of sequences across 46 different species.
 - `group_2_100-way.csv` contains similar measurements based on comparisions across 100 different species.
 - `group_1_2_merged.csv` combines all the features from the first two files.
 - `biased_46-way.csv` contains the 46-way data plus an extra, biased column that yields highly accurate but ultimately flawed classifiers.

