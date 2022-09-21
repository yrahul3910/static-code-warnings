import os

import numpy as np
import pandas as pd
from raise_utils.data import Data
from raise_utils.hyperparams import BinaryGHOST
from raise_utils.learners import *
from raise_utils.transforms import Transform
from scipy.spatial import KDTree
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from lime.lime_tabular import LimeTabularExplainer
import lime


def remove_labels(data):
    """
    Keep only sqrt(n) of the real labels. To find the others, use k=sqrt(sqrt(n))
    nearest neighbors from the labels we know, and use the mode.
    """
    # "Remove" labels
    lost_idx = np.random.choice(
        len(data.y_train), size=int(len(data.y_train) - np.sqrt(len(data.y_train))), replace=False)
    # lost_idx = np.random.choice(
    #    len(data.y_train), size=int(0.63 * len(data.y_train)), replace=False)
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


class _SVM(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learner = SVC(C=1., kernel='rbf')
        self.random_map = {
            'C': [0.1, 1., 10., 100.],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        self._instantiate_random_vals()


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_path = '../data/reimplemented_2016_manual/'
#datasets = ['ant', 'cassandra', 'commons', 'derby',
#            'jmeter', 'lucene-solr', 'maven', 'tomcat']
datasets = ['ant']

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
        data = Data(*train_test_split(X, y, test_size=.5, shuffle=False))
    else:
        data = Data(*train_test_split(X, y, test_size=.2, shuffle=False))
    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)
    data, ratio = remove_labels(data)

    try:
        transform = Transform('wfo')
        transform.apply(data)
        transform.apply(data)
        transform = Transform('smote')
        transform.apply(data)
    except ValueError:
        pass

    ghost = BinaryGHOST(['pd-pf', 'pd', 'pf',
                         'prec', 'auc'], n_runs=1, smote=True, autoencode=False,  name=dataset)
    ghost.set_data(*data)
    ghost.fit()

    best_learner = ghost.dodge.best_learner[1].model
    explainer = LimeTabularExplainer(
        training_data=data.x_train,
        feature_names=range(data.x_train.shape[1]),
        class_names=['non-actionable', 'actionable'],
        mode='classification'
    )
    explanation = explainer.explain_instance(
        data_row=data.x_test.iloc[np.random.choice(range(len(data.x_test))),:],
        predict_fn=best_learner.predict
    )
    explanation.save_to_file('explanation.html')
    """
    dodge_config = {
        'n_runs': 1,
        'transforms': ['standardize', 'normalize', 'minmax', 'maxabs'] * 30,
        'metrics': ['f1', 'accuracy', 'pd', 'pf', 'auc', 'prec'],
        'random': True,
        'log_path': './log_dodge',
        'learners': [],
        'data': [data],
        'n_iters': 30,
        'name': dataset
    }

    for _ in range(30):
        dodge_config['learners'].extend([
            _SVM(random=True),
            LogisticRegressionClassifier(random=True),
            RandomForest(random=True),
            NaiveBayes(),
            DecisionTree(random=True)
        ])

    dodge = DODGE(dodge_config)
    try:
        dodge.optimize()
    except ValueError:
        print('AUC cannot be computed.')
    """
