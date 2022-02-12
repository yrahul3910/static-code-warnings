from __future__ import division, print_function

import pandas as pd
import numpy as np
import random
import pdb
from demos import cmd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor

import glob
import copy
import collections

try:
    import cPickle as pickle
except:
    import pickle
from learners import Treatment, TM, SVM, RF, DT, NB, LR
from dnn_5by5 import *
import warnings
from raise_utils.data import Data
from raise_utils.transforms import Transform
warnings.filterwarnings('ignore')

BUDGET = 50
POOL_SIZE = 10000
INIT_POOL_SIZE = 10
SIZE = .1
np.random.seed(4789)


def load_updated_csv():
    data_types = ["FG"]
    datasets = ['ant', 'cassandra', 'commons', 'derby',
                'jmeter', 'lucene-solr', 'maven', 'tomcat']
    #path = "../../data/static_analysis-main/preprocessed_sherry/"
    path = '../../data/reimplemented_2016_manual/'
    final_data = {}
    """
    for t in data_types:
        train_data = glob.glob(f'{path}{t}_train_sampled/*.csv')
        final_data[t] = []
        for td in train_data:
            train_df = pd.read_csv(td)
            test_df = pd.read_csv(f'{path}{t}_test_common.csv')

            # training set
            training_x = train_df.iloc[:, :-1]  # pandas.core.frame.DataFrame
            training_y = train_df.iloc[:, -1]  # pandas.core.series.Series
            # testing set
            testset_x = test_df.iloc[:, :-1]
            testset_y = test_df.iloc[:, -1]
            training_y, training_x = preprocess1(training_y, training_x)
            testset_y, testset_x = preprocess1(testset_y, testset_x)
            training_x = training_x.select_dtypes(exclude=['object'])
            testset_x = testset_x.select_dtypes(exclude=['object'])
            # pdb.set_trace()
            final_data[t].append(
                (training_x, training_y, testset_x, pd.Series(testset_y)))
    return final_data
    """
    for dataset in datasets:
        print(dataset)
        print('=' * len(dataset))
        train_df = pd.read_csv(f'{path}train/{dataset}_B_features.csv')
        test_df = pd.read_csv(f'{path}test/{dataset}_C_features.csv')

        df = pd.concat((train_df, test_df), join='inner')
        X = df.drop('category', axis=1)
        y = df['category']

        y[y == 'close'] = 1.
        y[y == 'open'] = 0.

        X = X.select_dtypes(exclude=['object'])

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        yield {'FG': [(X_train, y_train, X_test, y_test)]}


def load_smoted_csv():
    data_types = ["FBC", "FG", "FC", "FBG"]
    path = "../data/static_analysis/preprocessed_sherry/"
    final_data = {}
    for t in data_types:
        train_data = glob.glob(f'{path}{t}_train_sampled/*.csv')
        final_data[t] = []
        for td in train_data:
            train_df = pd.read_csv(td)
            test_df = pd.read_csv(f'{path}{t}_test_common.csv')

            training_y, training_x = preprocess1(
                train_df.iloc[:, -1], train_df.iloc[:, :-1])
            testset_y, testset_x = preprocess1(
                test_df.iloc[:, -1], test_df.iloc[:, :-1])
            training_x = training_x.select_dtypes(exclude=['object'])
            testset_x = testset_x.select_dtypes(exclude=['object'])
            data = Data(training_x, testset_x, training_y, testset_y)
            transform = Transform('smote')
            transform.apply(data)
            x_train = pd.DataFrame(data.x_train, columns=training_x.columns)
            final_data[t].append(
                (x_train, data.y_train, data.x_test, pd.Series(data.y_test)))
    return final_data


def getHigherValueCutoffs(data, percentileCutoff):
    '''
        Parameters
        ----------
        data : in pandas format
        percentileCutoff : in integer
        class_category : [TODO] not needed

        Returns
        -------
        '''
    # pdb.set_trace()
    abc = data.quantile(float(percentileCutoff) / 100)
    abc = np.array(abc.values)[:-1]
    if abc.shape[0] == 0:
        abc = []
        for c in data.columns[:-1]:
            abc.append(np.percentile(data[c].values, percentileCutoff))
        abc = np.array(abc)
    return abc


def filter_row_by_value(row, cutoffsForHigherValuesOfAttribute):
    '''
        Shortcut to filter by rows in pandas
        sum all the attribute values that is higher than the cutoff
        ----------
        row
        cutoffsForHigherValuesOfAttribute

        Returns
        -------
        '''
    rr = row[:-1]
    condition = np.greater(rr, cutoffsForHigherValuesOfAttribute)
    res = np.count_nonzero(condition)
    return res


def getInstancesByCLA(data, percentileCutOff, positiveLabel):
    '''
        - unsupervised clustering by median per attribute
        ----------
        data
        percentileCutOff
        positiveLabel

        Returns
        -------

        '''
    # pdb.set_trace()
    # get cutoff per fixed percentile for all the attributes
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(
        data, percentileCutOff)
    # get K for all the rows
    K = data.apply(lambda row: filter_row_by_value(
        row, cutoffsForHigherValuesOfAttribute), axis=1)
    # cutoff for the cluster to be partitioned into
    cutoffOfKForTopClusters = np.percentile(K, percentileCutOff)
    instances = [1 if x > cutoffOfKForTopClusters else 0 for x in K]
    data["CLA"] = instances
    data["K"] = K
    return data


def getInstancesByRemovingSpecificAttributes(data, attributeIndices, invertSelection, label="Label"):
    '''
        removing the attributes
        ----------
        data
        attributeIndices
        invertSelection

        Returns
        -------
        '''
    # attributeIndices = data.columns[attributeIndices]
    if not invertSelection:
        data_res = data.drop(data.columns[attributeIndices], axis=1)
    else:
        # invertedIndices = np.in1d(range(len(attributeIndices)), attributeIndices)
        # data.drop(data.columns[invertedIndices], axis=1, inplace=True)
        data_res = data[attributeIndices]
        data_res['Label'] = data[label].values
    return data_res


def getInstancesByRemovingSpecificInstances(data, instanceIndices, invertSelection):
    '''
        removing instances
        ----------
        data
        instanceIndices
        invertSelection

        Returns
        -------

        '''
    if not invertSelection:
        data.drop(instanceIndices, axis=0, inplace=True)
    else:
        invertedIndices = np.in1d(range(data.shape[0]), instanceIndices)
        data.drop(invertedIndices, axis=0, inplace=True)
    return data


def getSelectedInstances(data, cutoffsForHigherValuesOfAttribute, positiveLabel):
    '''
        select the instances that violate the assumption
        ----------
        data
        cutoffsForHigherValuesOfAttribute
        positiveLabel

        Returns
        -------
        '''
    violations = data.apply(lambda r: getViolationScores(r,
                                                         data['Label'],
                                                         cutoffsForHigherValuesOfAttribute),
                            axis=1)
    violations = violations.values
    # get indices of the violated instances
    selectedInstances = (violations > 0).nonzero()[0]
    selectedInstances = data.index.values[selectedInstances]
    # remove randomly 90% of the instances that violate the assumptions
    # selectedInstances = np.random.choice(selectedInstances, int(selectedInstances.shape[0] * 0.9), replace=False)
    # for index in range(data.shape[0]):
    # 	if violations[index] > 0:
    # 		selectedInstances.append(index)
    try:
        tmp = data.loc[selectedInstances]
    except:
        tmp = data.loc[selectedInstances]
    if tmp[tmp["Label"] == 1].shape[0] < 10 or tmp[tmp["Label"] == 0].shape[0] < 10:
        # print("not enough data after removing instances")
        # category = 1 if tmp[tmp["Label"] == 1].shape[0] < 10 else 0
        len_0 = selectedInstances.shape[0]
        # len_0 -= data[data["Label"] == 1].shape[0]
        selectedInstances = np.random.choice(
            selectedInstances, int(len_0 * 0.9), replace=False)

    return selectedInstances


def CLA_SL(data, target, percentileCutoff, model="RF", seed=0, both=False, stats={"tp": 0, "p": 0}):
    try:
        traindata = copy.deepcopy(data["train"])
        testdata = copy.deepcopy(data[target])
    except:
        traindata = copy.deepcopy(data)
        testdata = copy.deepcopy(target)
    final_data = getInstancesByCLA(traindata, percentileCutoff, None)
    final_data["Label"] = final_data["CLA"]
    final_data.drop(["CLA", "K"], axis=1, inplace=True)
    results = training_CLAMI(final_data, testdata, target, model, stats=stats)
    return results


def get_CLAGRID(seed=4747, input="../new_data/corrected/", output="../results/CLAGRID_025_"):
    treatments = ["CLAMI", "CLA+SUP", "CLA"]
    # treatments = ["FRUGAL"] # ["CLAMI", "CLA+SUP", "CLA"]
    seed = int(time.time() * 1000) % (2 ** 32 - 1)

    for data in load_updated_csv():
        columns = ["Treatment"] + list(data.keys())
        print(output)
        # Supervised Learning Results
        result = {}
        result["Treatment"] = treatments
        for target in data:
            print(target)
            # dataset = data[target]
            result[target] = [updated_tuning(
                data, target, method="CLAMI", seed=seed)]
            result[target].append(updated_tuning(
                data, target, method="CLASUP", seed=seed))
            result[target].append(updated_tuning(
                data, target, method="CLA", seed=seed))
            # Output results to tables
            metrics = result[target][0].keys()
            print(result[target])
            for metric in metrics:
                df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in
                      result}
                pd.DataFrame(df, columns=columns).to_csv(output + metric + ".csv", line_terminator="\r\n",
                                                         index=False)


def get_smoted_CLAGRID(seed=4747, input="../new_data/corrected/", output="../results/CLAGRID_SMOTE_025_"):
    treatments = ["CLAMI", "CLA+SUP", "CLA"]
    # treatments = ["FRUGAL"] # ["CLAMI", "CLA+SUP", "CLA"]
    seed = int(time.time() * 1000) % (2 ** 32 - 1)
    data = load_smoted_csv()
    columns = ["Treatment"] + list(data.keys())
    print(output)
    # Supervised Learning Results
    result = {}
    result["Treatment"] = treatments
    for target in data:
        print(target)
        # dataset = data[target]
        # result[target] = [tuning_frugal(data, target, seed=seed)]
        result[target] = [updated_tuning(
            data, target, method="CLAMI", seed=seed)]
        result[target].append(updated_tuning(
            data, target, method="CLASUP", seed=seed))
        result[target].append(updated_tuning(
            data, target, method="CLA", seed=seed))
        # Output results to tables
        metrics = result[target][0].keys()
        print(result[target])
        for metric in metrics:
            df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in
                  result}
            pd.DataFrame(df, columns=columns).to_csv(output + metric + ".csv", line_terminator="\r\n",
                                                     index=False)


def CLA(data, positiveLabel, percentileCutoff, suppress=0, experimental=0, both=False):
    try:
        treatment = Treatment({}, "")
    except:
        treatment = Treatment(data, "")
    final_data = getInstancesByCLA(data, percentileCutoff, positiveLabel)
    treatment.y_label = ["yes" if y ==
                         1 else "no" for y in final_data["Label"]]
    treatment.decisions = ["yes" if y ==
                           1 else "no" for y in final_data["CLA"]]
    summary = collections.Counter(treatment.decisions)
    # print(summary, summary["yes"] / (summary["yes"] + summary["no"]))
    treatment.probs = final_data["K"]
    results = treatment.eval()
    results["read"] = summary["yes"] / (summary["yes"] + summary["no"])
    return results


def KMEANS(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, both=False):
    treatment = Treatment(data, target)
    treatment.preprocess()
    testdata = treatment.full_test
    data = getInstancesByCLA(testdata, percentileCutoff, positiveLabel)
    treatment.y_label = ["yes" if y == 1 else "no" for y in data["Label"]]
    treatment.decisions = ["yes" if y == 1 else "no" for y in data["CLA"]]
    treatment.probs = data["K"]
    return treatment.eval()


def CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, stats={"tp": 0, "p": 0},
          label="Label"):
    '''
        CLAMI - Clustering, Labeling, Metric/Features Selection,
                        Instance selection, and Supervised Learning
        ----------

        Returns
        -------

        '''
    # pdb.set_trace()
    try:
        traindata = copy.deepcopy(data["train"])
        testdata = copy.deepcopy(data[target])
    except:
        traindata = copy.deepcopy(data)
        testdata = copy.deepcopy(target)
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(
        traindata, percentileCutoff)
    # print("get cutoffs")
    traindata = getInstancesByCLA(traindata, percentileCutoff, positiveLabel)
    # print("get CLA instances")

    metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(traindata,
                                                                                 cutoffsForHigherValuesOfAttribute,
                                                                                 positiveLabel, label=label)
    # print("get Features and the violation scores")
    # pdb.set_trace()
    keys = list(metricIdxWithTheSameViolationScores.keys())
    # start with the features that have the lowest violation scores
    keys.sort()
    for i in range(len(keys)):
        k = keys[i]
        selectedMetricIndices = metricIdxWithTheSameViolationScores[k]
        # while len(selectedMetricIndices) < 3:
        # 	index = i + 1
        # 	selectedMetricIndices += metricIdxWithTheSameViolationScores[keys[index]]
        # print(selectedMetricIndices)
        # pick those features for both train and test sets
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(traindata,
                                                                            selectedMetricIndices, True, label=label)
        newTestInstances = getInstancesByRemovingSpecificAttributes(testdata,
                                                                    selectedMetricIndices, True, label="Label")
        # restart looking for the cutoffs in the train set
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                                  percentileCutoff)
        # get instaces that violated the assumption in the train set
        instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                       cutoffsForHigherValuesOfAttribute,
                                                       positiveLabel)
        # remove the violated instances
        trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                           instIndicesNeedToRemove, False)

        # make sure that there are both classes data in the training set
        zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
        one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
        if zero_count > 0 and one_count > 0:
            break
    return CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats=stats)


def percentile_tuning(func, train, tune):
    percentiles = range(5, 100, 5)
    results = []
    for p in percentiles:
        if func == "CLA":
            res = CLA(tune, None, p)
        elif func == "CLAMI":
            res = CLAMI(train, tune, None, p, label="CLA")
        elif func == "CLASUP":
            res = CLA_SL(train, tune, p, model="RF")
        results.append([res, p])
    return results


def run_method(func, train, test, metric, perc=50):
    if func == "CLA":
        res = CLA(test, None, perc)
    elif func == "CLAMI":
        res = CLAMI(train, test, None, perc, label="CLA")
    elif func == "CLASUP":
        res = CLA_SL(train, test, perc, model="RF")
    return res[metric]


def executing_methods(data, target, method="CLA", seed=0, perc=50):
    np.random.seed(seed)
    random.seed(seed)
    metrics = ["recall", "AUC", "fall-out"]
    results = {m: [] for m in metrics}
    for package in data[target]:
        x_0, x_1, y_0, y_1 = package
        x_0, y_0 = x_0.astype(float), y_0.astype(float)
        y_0["Label"] = y_1.astype(int).values
        x_0["Label"] = x_1

        traindata = copy.deepcopy(x_0)
        testdata = copy.deepcopy(y_0)
        for m in metrics:
            res = run_method(method, traindata, testdata, m, perc=perc)
            results[m].append(res)
    for m in metrics:
        res = np.array(results[m])
        results[m] = res[res != 0].median()
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results


def tuning(data, target, method="CLA", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    #metrics = ["precision", "f1"]
    metrics = ["f1", "accuracy", "g1", "AUC"]
    # metrics = ["recall", "AUC", "fall-out"]
    results = {m: [] for m in metrics}
    index = 0
    for package in data[target]:
        x_0, x_1, y_0, y_1 = package
        x_0, y_0 = x_0.astype(float), y_0.astype(float)
        y_0["Label"] = y_1.astype(int).values
        x_0["Label"] = x_1

        traindata = copy.deepcopy(x_0)
        testdata = copy.deepcopy(y_0)

        sss = StratifiedShuffleSplit(
            n_splits=5, test_size=SIZE, random_state=seed)
        X, y = traindata[traindata.columns[:-1]
                         ], traindata[traindata.columns[-1]]

        for train_index, tune_index in sss.split(X, y):
            # pdb.set_trace()
            print("Iteration = ", index)
            train_df = traindata.iloc[train_index]
            tune_df = traindata.iloc[tune_index]
            percentile_res = percentile_tuning(method, train_df, tune_df)
            for m in metrics:
                m_res = [[x[0][m], x[1]] for x in percentile_res]
                m_res.sort(key=lambda x: x[0])
                percentile = m_res[-1][1] if m != "fall-out" else m_res[0][1]
                res = run_method(method, train_df, testdata,
                                 m, perc=percentile)
                results[m].append(res)
            index += 1
    for m in metrics:
        res = np.array(results[m])
        res = res[~np.isnan(res)]
        results[m] = res.median()
        # results[m] = res[res!=0].mean()
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results


def updated_tuning(data, target, method="CLA", seed=0):
    np.random.seed(seed)
    random.seed(seed)
    metrics = ["f1", "precision", "recall", "AUC", "accuracy", "fall-out"]
    # metrics = ["f1", "accuracy", "g1", "AUC", "recall", "fall-out"]
    # metrics = ["recall", "AUC", "fall-out"]
    results = {m: [] for m in metrics}
    index = 0
    for package in data[target]:
        x_0, x_1, y_0, y_1 = package
        x_0, y_0 = x_0.astype(float), y_0.astype(float)
        y_0["Label"] = y_1.astype(int).values
        x_0["Label"] = x_1

        traindata = copy.deepcopy(x_0)
        testdata = copy.deepcopy(y_0)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=SIZE, random_state=seed)
        X, y = traindata[traindata.columns[:-1]
                         ], traindata[traindata.columns[-1]]

        for train_index, tune_index in sss.split(X, y):
            # pdb.set_trace()
            print("Iteration = ", index)
            train_df = traindata.iloc[train_index]
            tune_df = traindata.iloc[tune_index]
            percentile_res = percentile_tuning(method, train_df, tune_df)
            for m in metrics:
                m_res = [[x[0][m], x[1]] for x in percentile_res]
                m_res.sort(key=lambda x: x[0])
                percentile = m_res[-1][1] if m != "fall-out" else m_res[0][1]
                res = run_method(method, train_df, testdata,
                                 m, perc=percentile)
                results[m].append(res)
            index += 1
    for m in metrics:
        res = np.array(results[m])
        res = res[~np.isnan(res)]
        results[m] = np.median(res)
        # results[m] = res[res!=0].mean()
    print("*"*50)
    print(method)
    print(results)
    print("*"*50)
    return results


def CLAMI_eval(trainingInstancesByCLAMI, newTestInstances, target, stats={"tp": 0, "p": 0}):
    results = []
    # treaments = ["LR", "SVM", "RF", "NB"]
    # treaments = ["RF", "NB"]
    treaments = ["RF"]
    for mlAlg in treaments:
        results.append(training_CLAMI(trainingInstancesByCLAMI,
                       newTestInstances, target, mlAlg, stats=stats))
    return results[-1]


def MI(data, tunedata, selectedMetricIndices, percentileCutoff, positiveLabel, target):
    print(selectedMetricIndices)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
                                                                        selectedMetricIndices, True, label="CLA")
    newTuneInstances = getInstancesByRemovingSpecificAttributes(tunedata,
                                                                selectedMetricIndices, True, label="Label")
    cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
                                                              percentileCutoff, "Label")
    instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
                                                   cutoffsForHigherValuesOfAttribute,
                                                   positiveLabel)
    trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
                                                                       instIndicesNeedToRemove, False)
    zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
    one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
    if zero_count > 0 and one_count > 0:
        return selectedMetricIndices, training_CLAMI(trainingInstancesByCLAMI, newTuneInstances, target, "RF")
    else:
        return -1, -1


def transform_metric_indices(shape, indices):
    array = np.array([0] * shape)
    array[indices] = 1
    return array


def tune_CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, metric="APFD"):
    treatment = Treatment(data, target)
    treatment.preprocess()
    data = treatment.full_train
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=47)
    testdata = treatment.full_test
    X, y = data[data.columns[:-1]], data[data.columns[-1]]
    for train_index, tune_index in sss.split(X, y):
        train_df = data.iloc[train_index]
        tune_df = data.iloc[tune_index]
        train_df.reset_index(drop=True, inplace=True)
        tune_df.reset_index(drop=True, inplace=True)
        cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(
            train_df, percentileCutoff, "Label")
        print("get cutoffs")
        train_df = getInstancesByCLA(train_df, percentileCutoff, positiveLabel)
        print("get CLA instances")

        metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(train_df,
                                                                                     cutoffsForHigherValuesOfAttribute,
                                                                                     positiveLabel)
        # pdb.set_trace()
        keys = list(metricIdxWithTheSameViolationScores.keys())
        # keys.sort()
        evaluated_configs = random.sample(keys, INIT_POOL_SIZE * 2)
        evaluated_configs = [metricIdxWithTheSameViolationScores[k]
                             for k in evaluated_configs]

        tmp_scores = []
        tmp_configs = []
        for selectedMetricIndices in evaluated_configs:
            selectedMetricIndices, res = MI(train_df, tune_df, selectedMetricIndices,
                                            percentileCutoff, positiveLabel, target)
            if isinstance(res, dict):
                tmp_configs.append(transform_metric_indices(
                    data.shape[1], selectedMetricIndices))
                tmp_scores.append(res)

        ids = np.argsort([x[metric] for x in tmp_scores])[::-1][:1]
        best_res = tmp_scores[ids[0]]
        best_config = np.where(tmp_configs[ids[0]] == 1)[0]

        # number of eval
        this_budget = BUDGET
        eval = 0
        lives = 5
        print("Initial Population: %s" % len(tmp_scores))
        searchspace = [transform_metric_indices(data.shape[1], metricIdxWithTheSameViolationScores[k])
                       for k in keys]
        while this_budget > 0:
            cart_model = DecisionTreeRegressor()
            cart_model.fit(tmp_configs, [x[metric] for x in tmp_scores])

            cart_models = []
            cart_models.append(cart_model)
            next_config_id = acquisition_fn(searchspace, cart_models)
            next_config = metricIdxWithTheSameViolationScores[keys.pop(
                next_config_id)]
            searchspace.pop(next_config_id)
            next_config, next_res = MI(train_df, tune_df,
                                       next_config, percentileCutoff,
                                       positiveLabel, target)
            if not isinstance(next_res, dict):
                continue

            next_config_normal = transform_metric_indices(
                data.shape[1], next_config)
            tmp_scores.append(next_res)
            tmp_configs.append(next_config_normal)
            try:
                if abs(next_res[metric] - best_res[metric]) >= 0.03:
                    lives = 5
                else:
                    lives -= 1

                # pdb.set_trace()
                if isBetter(next_res, best_res, metric):
                    best_config = next_config
                    best_res = next_res

                if lives == 0:
                    print("***" * 5)
                    print("EARLY STOPPING!")
                    print("***" * 5)
                    break

                this_budget -= 1
                eval += 1
            except:
                pdb.set_trace()
    _, res = MI(train_df, testdata, best_config,
                percentileCutoff, positiveLabel, target)
    return res


def training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, model, all=True, stats={"tp": 0, "p": 0}):
    try:
        newTestInstances.drop(["CLA", "K"], axis=1, inplace=True)
    except:
        pass

    treatments = {"RF": RF, "SVM": SVM, "LR": LR, "NB": NB, "DT": DT, "TM": TM}
    treatment = treatments[model]
    clf = treatment({}, "")
    # print(target, model)
    clf.test_data = newTestInstances[newTestInstances.columns.difference([
                                                                         'Label'])].values
    clf.y_label = np.array(
        ["yes" if x == 1 else "no" for x in newTestInstances["Label"].values])

    try:
        clf.train_data = trainingInstancesByCLAMI.values[:, :-1]
        clf.x_label = np.array(
            ["yes" if x == 1 else "no" for x in trainingInstancesByCLAMI['Label']])
        clf.train()
        clf.stats = stats

        summary = collections.Counter(clf.decisions)
        # print(summary, summary["yes"] / (summary["yes"] + summary["no"]))
        results = clf.eval()
        results["read"] = summary["yes"] / (summary["yes"] + summary["no"])
        if all:
            return results
        else:
            return results["APFD"] + results["f1"]
    except:
        raise
        # pdb.set_trace()
        results = {"AUC": 0, "recall": 0, "precision": 0,
                   "f1": 0, "fall-out": 1, "accuracy": 0, "g1": 0}
        return results


def getViolationScores(data, labels, cutoffsForHigherValuesOfAttribute, key=-1):
    '''
        get violation scores
        ----------
        data
        labels
        cutoffsForHigherValuesOfAttribute
        key

        Returns
        -------

        '''
    violation_score = 0
    if key not in ["Label", "K", "CLA"]:
        if key != -1:
            # violation score by columns
            categories = labels.values
            cutoff = cutoffsForHigherValuesOfAttribute[key]
            # violation: less than a median and class = 1 or vice-versa
            violation_score += np.count_nonzero(np.logical_and(
                categories == 0, np.greater(data.values, cutoff)))
            violation_score += np.count_nonzero(np.logical_and(
                categories == 1, np.less_equal(data.values, cutoff)))
        else:
            # violation score by rows
            row = data.values
            row_data, row_label = row[:-1], row[-1]
            # violation: less than a median and class = 1 or vice-versa

            row_label_0 = np.array(row_label == 0).tolist() * row_data.shape[0]
            # randomness = random.random()
            # if randomness > 0.5:
            violation_score += np.count_nonzero(np.logical_and(row_label_0,
                                                               np.greater(row_data, cutoffsForHigherValuesOfAttribute)))
            row_label_1 = np.array(row_label == 0).tolist() * row_data.shape[0]
            violation_score += np.count_nonzero(np.logical_and(row_label_1,
                                                               np.less_equal(row_data,
                                                                             cutoffsForHigherValuesOfAttribute)))

    # for attrIdx in range(data.shape[1] - 3):
    # 	# if attrIdx not in ["Label", "CLA"]:
    # 	attr_data = data[attrIdx].values
    # 	cutoff = cutoffsForHigherValuesOfAttribute[attrIdx]
    # 	violations.append(getViolationScoreByColumn(attr_data, data["Label"], cutoff))
    return violation_score


def isBetter(new, old, metric):
    if metric == "d2h":
        return new[metric] < old[metric]
    else:
        return new[metric] > old[metric]


def getMetricIndicesWithTheViolationScores(data, cutoffsForHigherValuesOfAttribute, positiveLabel, label="Label"):
    '''
        get all the features that violated the assumption
        ----------
        data
        cutoffsForHigherValuesOfAttribute
        positiveLabel

        Returns
        -------

        '''
    # cutoffs for all the columns/features
    # pdb.set_trace()
    # cutoffsForHigherValuesOfAttribute = {i: x for i, x in enumerate(cutoffsForHigherValuesOfAttribute)}
    cutoffsForHigherValuesOfAttribute = {x: y for x, y in zip(
        data.columns, cutoffsForHigherValuesOfAttribute)}
    # use pandas apply per column to find the violation scores of all the features
    violations = data.apply(
        lambda col: getViolationScores(col, data[label],
                                       cutoffsForHigherValuesOfAttribute,
                                       key=col.name),
        axis=0)
    violations = violations.values
    metricIndicesWithTheSameViolationScores = collections.defaultdict(list)

    # store the violated features that share the same violation scores together
    for attrIdx in range(data.shape[1] - 3):
        key = violations[attrIdx]
        metricIndicesWithTheSameViolationScores[key].append(
            data.columns[attrIdx])
    return metricIndicesWithTheSameViolationScores


if __name__ == "__main__":
    eval(cmd())
