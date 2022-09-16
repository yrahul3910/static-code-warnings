import os

import numpy as np
import pandas as pd
import tensorflow as tf
from raise_utils.data import Data
from raise_utils.hyperparams import DODGE
from raise_utils.learners import Learner
from raise_utils.transforms import Transform
from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras.models import Sequential

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_path = '../data/reimplemented_2016_manual/'
datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven',  'tomcat']


class CNN(Learner):
    def __init__(self, n_blocks=1, dropout_prob=0.2, n_filters=32, kernel_size=64, verbose=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learner = self
        self.model = Sequential()

        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.dropout_prob = dropout_prob
        self.kernel_size = kernel_size
        self.verbose = verbose

        self.random_map = {
            'n_blocks': (1, 4),
            'n_filters': [4, 8, 16, 32, 64],
            'dropout_prob': (0.05, 0.5),
            'kernel_size': [16, 32, 64]
        }
        self._instantiate_random_vals()

    def set_data(self, x_train, y_train, x_test, y_test):
        super().set_data(x_train, y_train, x_test, y_test)

        self.x_train = np.array(self.x_train).reshape(
            (*self.x_train.shape, 1, 1))
        self.x_test = np.array(self.x_test).reshape((*self.x_test.shape, 1, 1))
        self.y_train = np.array(self.y_train).squeeze()
        self.y_test = np.array(self.y_test).squeeze()

        if tf.__version__ >= '2.0.0':
            # We are running TF 2.0, so need to type cast.
            self.y_train = self.y_train.astype('float32')

    def fit(self):
        self._check_data()
        print(self.n_filters, self.dropout_prob,
              self.kernel_size, self.n_blocks)

        callback = EarlyStopping(monitor='loss', patience=10)

        self.model.add(Conv2D(self.n_filters, self.kernel_size, padding='same',
                              input_shape=self.x_train.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout_prob))

        for block in range(self.n_blocks - 1):
            # Order taken from https://stackoverflow.com/a/40295999/2713263.
            self.model.add(
                Conv2D(self.n_filters, self.kernel_size, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(Activation('relu'))
            self.model.add(Dropout(self.dropout_prob))

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

        self.model.fit(self.x_train, self.y_train, epochs=100,
                       batch_size=64, callbacks=[callback], verbose=self.verbose)

    def predict(self, x_test) -> np.ndarray:
        """
        Makes predictions
        :param x_test: Test data
        :return: np.ndarray
        """
        return (self.model.predict(x_test.reshape((*x_test.shape, 1, 1))) > 0.5).astype('int32')


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

    X = df.drop('category', axis=1)
    y = df['category']

    y[y == 'close'] = 1
    y[y == 'open'] = 0

    y = np.array(y, dtype=np.float32)

    X = X.select_dtypes(
        exclude=['object']).astype(np.float32)

    if dataset == 'maven':
        data = Data(*train_test_split(X, y, test_size=.5))
    else:
        data = Data(*train_test_split(X, y, test_size=.2))
    print(len(data.x_train), len(data.x_test))
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)
    data, ratio = remove_labels(data)
    # ratios.append(ratio)

    try:
        transform = Transform('smote')
        transform.apply(data)
    except ValueError:
        pass

    dodge_config = {
        'n_runs': 1,
        'transforms': ['standardize', 'normalize', 'minmax', 'maxabs'] * 30,
        'metrics': ['pd-pf', 'accuracy', 'pd', 'pf', 'auc', 'prec'],
        'random': True,
        'log_path': './log_dodge',
        'learners': [],
        'data': [data],
        'n_iters': 30,
        'name': dataset
    }

    for _ in range(30):
        dodge_config['learners'].append(
            CNN(random=True)
        )
    dodge = DODGE(dodge_config)
    try:
        dodge.optimize()
    except:
        pass
