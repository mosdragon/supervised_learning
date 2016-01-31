## Supervised Learning Algorithms

This is a convenient sample of some of the most common supervised
learning algorithms, provided by scikit-learn and PyBrain. Also provided are
some helpful utils in __data_utils.py__.

### Disclaimer
I give no guarantees as to what will work and what won't. I'm a student writing
this for my own project, and I've made this with my limited understanding of
the scikit-learn framework.

__Read the comments for each function before you use it.__

__And backup your data just in case.__

### Dependencies
You will need Python. I ran this on 2.7, but I'm sure the libraries are available
for Python 3.x as well.

You also need the following packages:
- numpy
- scipy
- pandas
- scikit-learn
- PyBrain

It's easiest to install it with pip. Note, you may need to use "sudo" to install
using pip, depending on your machine.

```bash
pip install numpy scipy pandas scikit-learn pybrain
```

### Feature Extraction
If your dataset contains attributes that are text, with the exception of the
output column, you will need to do some feature extraction for your data to be
compatible with scikit-learn and Pybrain's algorithms. For convenience, I've
added a simple helper method in __data_util.py__ to do this.

Usage:
```python
import data_util

# Note, must be a csv
INPUT_DATASET = 'path/to/in_dataset.csv'
OUTPUT_DATASET = 'path/to/out_dataset.csv'

data_util.clean_data(INPUT_DATASET, OUTPUT_DATASET)
```

You will now find a csv where you specified. This csv will now have, at most,
one column with text attributes, which is the very last column.


### Usage
At the top of every file for the algorithms, you'll find parameters to change
so you can run the algorithms with your own specified configurations. This is
what you'll want to change to get the desired performance from these algorithms.

Example usage from svm.py:

```python
# Data params
FILENAME = 'investing/cleaned/class_sentiments.csv'
CLASSIFIER_FIELD = 'S&P500 Next Week Positive Returns'

PERCENTAGE_TRAINING = 0.70

# Cross Validation Params
N_FOLDS = 10

# SVM Params
KERNEL = 'rbf'  # Options: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
DEGREE = 1
```
