import os

import numpy as np
import pandas as pd
from raise_utils.data import Data
from raise_utils.learners import Learner
from raise_utils.metrics import ClassificationMetrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_path = '../data/reimplemented_2016_manual/'
datasets = ['ant', 'cassandra', 'commons', 'derby',
            'jmeter', 'lucene-solr', 'maven',  'tomcat']


class _SVM(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learner = SVC(class_weight='balanced', kernel='rbf')
        self.random_map = {
            'C': [0.1, 1., 10., 100.],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        self._instantiate_random_vals()


results = []
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

    data = Data(
        *train_test_split(X, y, test_size=.2 if dataset != 'maven' else .5))

    data.x_train = np.array(data.x_train)
    data.y_train = np.array(data.y_train)

    #transform = Transform('normalize')
    # transform.apply(data)
    ghost = _SVM()
    ghost.set_data(*data)

    try:
        ghost.fit()
        m = ClassificationMetrics(data.y_test, ghost.predict(data.x_test))
        m.add_metrics(['prec', 'auc', 'pf', 'pd'])
        res = m.get_metrics()
        print(res)
        results.append(res)
    except:
        pass

results = np.array(results)
print(np.median(results, axis=0))
