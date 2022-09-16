import numpy as np
from raise_utils.data import Data
from raise_utils.hyperparams import DODGE
from raise_utils.learners import *


class DODGELearner(Learner):
    """
    Implements DODGE with standard learners..
    """

    def __init__(self, metrics: list, max_evals: int = 30, name='experiment', *args, **kwargs):
        """
        Initializes the DODGE algorithm. Several of these are internal parameters exposed for completeness.

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
                RandomForest(random=True),
                LogisticRegressionClassifier(random=True),
                DecisionTree(random=True),
                NaiveBayes(random=True)
            ])

        dodge = DODGE(dodge_config)
        return dodge.optimize()

    def predict(self, x_test):
        """
        Makes predictions on x_test.
        """
        raise NotImplementedError
