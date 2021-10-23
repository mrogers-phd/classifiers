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

### Running in Docker
If you do not have Python 3 or the external dependencies listed above, this is probably the simplest
way to start:

 1. `build_image.sh` builds a Docker image with the required dependencies
 2. `run_notebook.sh` runs the Docker container in a Jupyter notebook
 3. `run_container.sh` runs the Docker container in a shell

#### Using the Jupyter notebook
Running the image will kick off a Jupyter notebook with instructions for how to view it
in a web browser such as Chrome.  Simply copy and paste one of the `http:` links into your
browser and then select one of the notebooks (`*.ipynb`) to get started.

Here is an example of what you can expect to see:
```
$ run_notebook.sh
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

#### Using the Docker shell
There are notebooks for both the 2D and high-dimensional data examples, but in general it is easier
to run the high-dimensional script from the shell.

Here is an example of what you can expect to see:
```
$ run_container.sh
root@c96bcb434afc:~# test_classifiers.py data/group_1_46-way.csv -v
Using 3 CPUs
Data: ClassifierData with 8 features and 2 labels
Testing the following models:
  Gradient Boosting
running Gradient Boosting on 6 settings......

                              	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Gradient Boosting (N=64)      	0.834	0.849	0.820	0.825	0.669	0.908
Gradient Boosting (N=32)      	0.833	0.845	0.822	0.826	0.667	0.907
Gradient Boosting (N=16)      	0.832	0.842	0.822	0.826	0.665	0.902
Gradient Boosting (N=8)       	0.828	0.835	0.821	0.824	0.657	0.892
Gradient Boosting (N=4)       	0.827	0.833	0.821	0.823	0.653	0.886
Gradient Boosting (N=2)       	0.817	0.818	0.816	0.817	0.635	0.869
root@c96bcb434afc:~#
```

**Note:** _currently the scripts are not set up to allow the Docker container to render images
to your screen.  The solutions for allowing a container access to the host's DISPLAY were too
involved for a simple demonstration.  If you wish to display the 2D results outside of a Jupyter
notebook, the best course is to install the appropriate modules on your host computer and run
everything from the command line as described below._

### Running scripts from the command line
If the external dependencies are already on your machine, then you can run the scripts directly
from the command line.  This will require an update to your `PYTHONPATH` and `PATH` variables
to include the `bin` and `shared` directories in your paths.  For convenience, the script
`setup_env.sh` has been included:
```
. ./setup_env.sh
```

At this point, running the scripts should be a simple matter of entering the script on the command line:
```
$ test_classifiers.py data/group_2_100-way.csv -v
```
or
```
$ compare_2d.py 
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

## Statistical measures
 - **Balanced Accuracy**: provides a realistic measure of accuracy regardless of whether the data are balanced.
 - **Sensitivity**, or true positive rate (TPR), measures a model's ability to recognize the positive class correctly.
 - **Specificity**, or true negative rate (TNR), measures a model's ability to recognize the negative class correctly.
 - **Positive Predictive Value (PPV)**: probability that a positive prediction is really positive.
 - **Matthews Correlation Coefficient (MCC)**: measures the quality of binary predictions with values ranging from -1 to 1.
 - **Area under the ROC curve (AUC)**: measures how well a classifier ranks examples from strongest positive score to strongest negative score.

## Example
Below is output from an example run on four different classifiers:
```
$ bin/test_classifiers.py -p 8 -m DNOG data/group_1_46-way.csv

                                        	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Decision Tree                           	0.778	0.779	0.777	0.777	0.556	0.778

                                        	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Naive Bayes                             	0.805	0.855	0.756	0.778	0.614	0.886

                                        	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Logistic Regression (C=1)               	0.832	0.857	0.806	0.816	0.665	0.909
Logistic Regression (C=0.1)             	0.831	0.839	0.824	0.826	0.663	0.902
Logistic Regression (C=0.01)            	0.823	0.813	0.832	0.829	0.645	0.893
Logistic Regression (C=0.001)           	0.805	0.796	0.815	0.811	0.611	0.881

                                        	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Gradient Boosting (N=64)                	0.837	0.851	0.823	0.828	0.674	0.908
Gradient Boosting (N=32)                	0.834	0.844	0.825	0.828	0.669	0.908
Gradient Boosting (N=16)                	0.831	0.840	0.821	0.824	0.662	0.902
Gradient Boosting (N=8)                 	0.830	0.836	0.825	0.827	0.661	0.896
Gradient Boosting (N=4)                 	0.825	0.836	0.815	0.819	0.651	0.887
Gradient Boosting (N=2)                 	0.821	0.831	0.810	0.814	0.642	0.871

Overall rankings:
                                        	Acc.  	Sens. 	Spec. 	PPV   	MCC   	AUC
Gradient Boosting (N=64)                	0.837	0.851	0.823	0.828	0.674	0.908
Logistic Regression (C=1)               	0.832	0.857	0.806	0.816	0.665	0.909
Naive Bayes                             	0.805	0.855	0.756	0.778	0.614	0.886
Decision Tree                           	0.778	0.779	0.777	0.777	0.556	0.778
```
