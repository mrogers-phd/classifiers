# Classifier Sandbox
This repository contains python code for playing with machine learning classifiers.
It includes two scripts designed to help gain an understanding of how different
classifiers perform with different data and parameters.

## Code layout
The code for demonstrating classifiers is written for Python 3 and is organized as follows:

### External dependencies
The scripts rely on the following modules.  Software was confirmed to work on the versions indicated:
 - numpy (1.17.4)
 - matplotlib (3.1.2)
 - scikit-learn (0.23.1)

### Python scripts
The main scripts reside in the `bin` directory, while the python modules they rely on are in
the `shared` directory.  The main scripts are:
 - `bin/compare_2d.py` compares a variety of classifiers using 2-D data
 - `bin/test_classifiers.py` compares a variety of classifiers on realistic data.

Files in the `shared` directory are not proper python modules; the code has been split out to
make it easier to demonstrate the code in a Jupyter notebook.  These files are:
 - `shared/comparison_utils.py` has methods for the `compare_2d.py` script
 - `shared/sklearn_utils.py` has methods and classes for the `test_classifiers.py` script

## Quick Start
There are a few scripts to help you get started as quickly as possible.

### Running Jupyter notebooks
If you do not have Python 3 or the external dependencies listed above, this is probably the simplest
way to start:

 1. `build_image.sh` builds a Docker image with the required dependencies
 2. `run_image.sh` runs the Docker container and starts a Jupyter notebook

Running the image will kick off a Jupyter notebook with instructions for how to view it
in a web browser such as Chrome.  Simply copy and paste one of the `http:` links into your
browser and then select one of the notebooks (`*.ipynb`) to get started.

Here is an example of what you can expect to see:
```
$ run_container.sh
[I 12:51:18.901 NotebookApp] Writing notebook server cookie secret to /home/classifiers/.local/share/jupyter/runtime/notebook_cookie_secret
[I 12:51:19.115 NotebookApp] Serving notebooks from local directory: /home/classifiers
[I 12:51:19.115 NotebookApp] Jupyter Notebook 6.4.4 is running at:
[I 12:51:19.115 NotebookApp] http://5462bb3c1b24:8888/?token=7324a10ee00ccc453aa6da75bbd8771b38393606a7afeebe
[I 12:51:19.115 NotebookApp]  or http://127.0.0.1:8888/?token=7324a10ee00ccc453aa6da75bbd8771b38393606a7afeebe
[I 12:51:19.115 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 12:51:19.118 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/classifiers/.local/share/jupyter/runtime/nbserver-1-open.html
    Or copy and paste one of these URLs:
        http://5462bb3c1b24:8888/?token=7324a10ee00ccc453aa6da75bbd8771b38393606a7afeebe
     or http://127.0.0.1:8888/?token=7324a10ee00ccc453aa6da75bbd8771b38393606a7afeebe
```

### Running scripts from the command line
If the external dependencies are already on your machine, then you can run the scripts directly
from the command line.  This will require an update to your `PYTHONPATH` and `PATH` variables
to include the `bin` and `shared` directories in your paths.  For convenience, the script
`setup_env.sh` has been included:
```
. ./setup_env.sh
```

## 2-D dataset comparisons
The first script, `compare_2d.py`  uses a relatively simple 2-dimensional dataset to compare
the performance of different classifiers.  It was pulled from
[an example scikit-learn website](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
originally written by Gaël Varoquaux and Andreas Müller.
The script has been extended to include additional models;
to provide additional control over model parameters,
and to provide control over which models to compare.
There are default values for all models, so it is possible to try the script using simply:

```
$ compare_2d.py
```

The full set of options is available in the online help:
```
$ compare_2d.py -h
usage: compare_2d.py [-h] [-d DT_DEPTH] [-k K] [-m MODEL_CODES] [-n N] [-r RF_DEPTH] [-s SEED] [-G GAMMA] [-L LINEAR_C] [-P GP_LENGTH] [-v]

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

# Statistical measures
