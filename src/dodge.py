import numpy as np
from raise_utils.data import Data
from raise_utils.hyperparams import DODGE
from raise_utils.learners import *
from sklearn.svm import SVC


class _SVM(Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learner = SVC(C=1., kernel='rbf')
        self.random_map = {
            'C': [0.1, 1., 10., 100.],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        self._instantiate_random_vals()


class _DODGE(Learner):
    """
    Implements the original, 2-class GHOST algorithm.
    """

    def __init__(self, metrics: list, max_evals: int = 30, name='experiment', *args, **kwargs):
        """
        Initializes the GHOST algorithm. Several of these are internal parameters exposed for completeness.
        If you do not understand what a parameter does, the default value should work.

        :param metrics: A list of metrics supplied by raise_utils.metrics to print out.
        :param max_evals: The max number of hyper-parameter evaluations.
        :param name: A name for the DODGE runs.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.metrics = metrics
        self.max_evals = max_evals

    def fit(self):
        self._check_data()

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        data = Data(self.x_train, self.x_test, self.y_train, self.y_test)

        dodge_config = {
            'n_runs': 1,
            'data': [data],
            'metrics': self.metrics,
            'n_iters': self.max_evals,
            'learners': [],
            'log_path': './log/',
            'transforms': ['standardize', 'normalize', 'minmax'] * 30,
            'random': True,
            'name': self.name
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
        return dodge.optimize()

    def predict(self, x_test):
        """
        Makes predictions on x_test.
        """
        raise NotImplementedError
