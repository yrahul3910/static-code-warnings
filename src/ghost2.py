import os
import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob
from ivis import Ivis
from ghost import BinaryGHOST
from raise_utils.learners import Autoencoder, RandomForest, LogisticRegressionClassifier
from raise_utils.hyperparams import DODGE
from raise_utils.transforms import Transform
from raise_utils.data import DataLoader, Data
from raise_utils.metrics import ClassificationMetrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
from scipy.spatial import KDTree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_path = '../data/reimplemented_2016_manual/'
datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr',  'tomcat']


def remove_labels(data):
    """
    Keep only sqrt(n) of the real labels. To find the others, use k=sqrt(sqrt(n))
    nearest neighbors from the labels we know, and use the mode.
    """
    # "Remove" labels
    # lost_idx = np.random.choice(
    #    len(data.y_train), size=int(len(data.y_train) - np.sqrt(len(data.y_train))))
    lost_idx = np.random.choice(
        len(data.y_train), size=int(0.63 * len(data.y_train)), replace=False)
    X_lost = data.x_train[lost_idx]
    X_rest = np.delete(data.x_train, lost_idx, axis=0)
    y_lost = data.y_train[lost_idx]
    y_rest = np.delete(data.y_train, lost_idx, axis=0)

    if len(X_lost.shape) == 1:
        X_lost = X_lost.reshape(1, -1)
    if len(X_rest.shape) == 1:
        X_rest = X_rest.reshape(1, -1)

    # Impute data
    for i in range(len(X_lost)):
        tree = KDTree(X_rest)
        d, idx = tree.query([X_lost[i]], k=int(np.sqrt(len(X_rest))), p=1)
        y_lost[i] = mode(y_rest[idx][0])[0][0]

    print('Ratio =', round(len(X_rest) / len(data.y_train), 2))
    print('Total =', len(X_lost) + len(X_rest))
    data.x_train = np.concatenate((X_lost, X_rest), axis=0)
    data.y_train = np.concatenate((y_lost, y_rest), axis=0)
    return data, 0.8 * len(X_rest) / (len(X_rest) + len(X_lost))


results = []
ratios = []
for dataset in datasets:
    print(dataset)
    print('=' * len(dataset))

    train_file = base_path + 'train/' + dataset + '_B_features.csv'
    test_file = base_path + 'test/' + dataset + '_C_features.csv'

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    df = pd.concat((train_df, test_df), join='inner')
    df.sample(frac=1)  # shuffle data

    X = df.drop('category', axis=1)
    y = df['category']

    y[y == 'close'] = 1
    y[y == 'open'] = 0

    y = np.array(y, dtype=np.float32)

    X = X.select_dtypes(
        exclude=['object']).astype(np.float32)

    data = Data(*train_test_split(X, y, test_size=.2))
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)
    data, ratio = remove_labels(data)
    ratios.append(ratio)

    try:
        transform = Transform('smote')
        transform.apply(data)
    except ValueError:
        pass

    ghost = BinaryGHOST(['pd-pf', 'f1', 'accuracy', 'pd', 'pf',
                         'prec', 'auc'], autoencode=False, name=dataset)
    ghost.set_data(*data)
    results.append(ghost.fit())

results = np.array(results)
print(np.median(results, axis=0))
print(np.median(ratios))