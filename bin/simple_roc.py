#! /usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import os
import sys

from matplotlib import rcParams
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay


def load_csv_pairs(csv_file):
    """Loads prediction score and label pairs from a CSV file."""
    y_test = []
    y_pred = []
    for line in open(csv_file):
        [pred_str, label_str] = line.strip().split(',')
        y_test.append(int(label_str))
        y_pred.append(float(pred_str))
    return y_test, y_pred


DESCRIPTION = """
Plots an ROC curve based on data in a CSV file.  Data must have the form:
    score,label

Where score is output from some classifier, and label is the expected label associated with the score.
"""

parser = argparse.ArgumentParser(description=DESCRIPTION)
parser.add_argument('csv_files', help='File containing score,label pairs', nargs='+')
parser.add_argument('-F', dest='fontsize', help='Font size', default=10.0)
parser.add_argument('-H', dest='height', help='Plot height', default=6.0)
parser.add_argument('-n', dest='names', help='Comma-separated list of model names corresponding to the input files', default=None)
parser.add_argument('-o', dest='output', help='Output file [default=screen]', default=None)
parser.add_argument('-W', dest='width', help='Plot width', default=6.0)
parser.add_argument('-t', dest='title', help='Plot title', default=None)
parser.add_argument('-v', help='verbose mode', action='store_true')

args = parser.parse_args()

if args.names is None:
    names = args.csv_files
else:
    names = args.names.split(',')

if len(names) != len(args.csv_files):
    parser.print_help()
    sys.stderr.write('\n** You provided {} files but {} names'.format(len(args.csv_files), len(names)))
    sys.exit(1)

rcParams['figure.figsize']   = args.width, args.height
rcParams['font.size']        = args.fontsize
rcParams['axes.titlesize']   = 1.2*args.fontsize
rcParams['axes.labelsize']   = args.fontsize
rcParams['xtick.labelsize']  = args.fontsize
rcParams['ytick.labelsize']  = args.fontsize
rcParams['legend.fontsize']  = 0.8*args.fontsize
rcParams['font.weight']      = 'bold'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'

# Avoid Type 3 fonts 
rcParams['figure.dpi']   = 150
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype']  = 42

ax = None

for i in range(len(args.csv_files)):
    f = args.csv_files[i]
    y_test, y_pred = load_csv_pairs(f)
    if ax is None:
        RocCurveDisplay.from_predictions(y_test, y_pred, name=names[i])
        ax = plt.gca()
    else:
        RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax, name=names[i])

# Eliminate extra axis padding
ax.set_ylim(0,1)
ax.set_xlim(0,1)

ax.set_xlabel('FP')
ax.set_ylabel('TP')

# Show diagonal
ax.plot([0,1], [0,1], 'k--')

# Grey background
ax.set_facecolor([0.85, 0.85, 0.85])

if args.title is not None:
    ax.set_title(args.title)

plt.tight_layout(pad=2.0, w_pad=0.0, h_pad=1.0)
if not args.output :
    plt.show()
else :
    plt.savefig(args.output, dpi=150)
    plt.close()

