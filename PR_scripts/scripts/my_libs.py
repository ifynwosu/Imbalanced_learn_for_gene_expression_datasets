import os, sys

import numpy as np
import pandas as pd

# from collections import Counter

# from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from imblearn.datasets import make_imbalance
from imblearn.metrics import classification_report_imbalanced

from imblearn.metrics import make_index_balanced_accuracy
# from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import NearMiss


########################
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from imblearn.pipeline import make_pipeline

from sklearn.metrics import classification_report