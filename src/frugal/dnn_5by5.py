# from __future__ import division
import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from statistics import mean
import time
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot
import pdb
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

""" NOTES:
      - requires Python 3.0 or greater
      - 5 by 5
"""

__author__ = 'Sherry'


def open(path):
    testFiles = path + "/test_set/totalFeatures5.csv"
    trainingFiles = glob.glob(path + "/training_set/*.csv")
    list_training = []  # training set list
    list_header = []  # get the list of headers of dfs for training & testing
    for file in trainingFiles:
        df = pd.read_csv(file, index_col=None, header=None, skiprows=0)

        head = df.iloc[0]
        head.tolist()
        list_header.append(head)  # list of header for training set
        list_training.append(df)  # list of training set

    testing = pd.read_csv(testFiles, index_col=None, header=None, skiprows=0)
    # training = pd.concat(list1, axis=0, ignore_index=True)
    list_header.append(testing.iloc[0].tolist())
    return testing, list_training, list_header

def data_preparing(path, stop_at, perc, seed=0):
    testing, list_training, list_header = open(path)
    # merge, common = merge1(testing, list_training, list_header)
    common_header = common_get(list_header)
    np.random.seed(seed)

    # training set
    training_trim = trim(list_training[0], common_header)  # ONLY USE THE VERSION 4 FOR TRAINING

    # testing set
    testing_trim = trim(testing, common_header)
    # testing_trim.iloc[0]   to check the type of each feature

    # training set
    training_x = training_trim.iloc[:, :-1]  # pandas.core.frame.DataFrame
    training_y = training_trim.iloc[:, -1]  # pandas.core.series.Series
    columns = training_x.columns
    # testing set
    testset_x = testing_trim.iloc[:, :-1]
    testset_y = testing_trim.iloc[:, -1]

    # remove the samples in training and test set with label "deleted"
    training_y, training_x = preprocess1(training_y, training_x)
    testset_y, testset_x = preprocess1(testset_y, testset_x)
    training_y, training_x, ratio = ratio_down(training_y, training_x, perc=perc)

    # using one-hot to preprepossing the label
    # training_y = to_categorical(training_y)
    # testset_y1 = to_categorical(testset_y)

    le = preprocessing.LabelEncoder()
    # kaggle's forums where to find valuable information

    # normalize the x for training and test sets
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = MinMaxScaler()

    testset_x = np.asarray(testset_x)  # convert dataframe into numpy.array to normalize
    # ########
    testset_x = min_max_scaler.fit_transform(testset_x)

    training_x = np.asarray(training_x)  # convert dataframe into numpy.array to normalize

    # ########   linear svm needs normalization to reduce the running cost on server
    training_x = min_max_scaler.fit_transform(training_x)

    return training_x, training_y, testset_x, testset_y, columns

def df_get(df):
    """
    :param df: a data frame with header
    :return: get rid of the header of data frame
    """
    header = df.iloc[0]
    # Create a new variable called 'header' from the first row of the dataset
    # Replace the dataframe with a new one which does not contain the first row
    df = df[1:]

    # Rename the dataframe's column values with the header variable
    df = df.rename(columns=header)
    h1 = list(df.columns.values)  # get the value of header of the df
    return df, h1


def common_get(list_header):
    """
    :param list_header: list of training & testing headers
    :return: common header
    """

    golden_fea = ["F116", "F115", "F117", "F120", "F123", "F110", "F105", "F68", "F101", "F104", "F65", "F22",
                  " F94", "F71", "F72", "F25", "F3-", "F15", "F126", "F41", "F77"]

    # mentioned in Dr. Wang's paper: Commonly-selected features (RQ1)

    golden_fea.append("category")
    golden_list = []
    count_list = []
    for header in list_header:
        golden = []
        count = 0
        for i in header:
            if i.startswith(tuple(golden_fea)):
                count += 1
                golden.append(i)
        # print("number of golden fea:", count)
        count_list.append(count)
        golden_list.append(golden)

    common = set(golden_list[0])
    for s in golden_list[1:]:
        common.intersection_update(s)
    return common


def trim(df, common):
    df1, header = df_get(df)
    for element in header:
        if element not in common:
            df1 = df1.drop(element, axis=1)
    df_trim = df1
    return df_trim


def is_number(df):
    '''
    :param df: input should be training_x, testset_x(type: data frame)
    :return: return is index of numeric features
    '''

    index = []
    position = 0
    for i in range(len(df.iloc[0])):
        s = df.iloc[0, i]
        try:
            float(s)  # for int, long and float
            index.append(i)
        except ValueError:
            position += 1
    return index


def preprocess1(Y, X):
    index = []
    label = []
    # index_target = []
    for i in range(0, len(Y)):
        # y = Y[0][i]
        y = Y.iloc[i]
        if y == "close":
            # y = "yes"
            y = 1
            # index_target.append(i)
        elif y == "open":
            # y = "no"
            y = 0
        elif y == "deleted":
            index.append(i)  # index is a list of index for deleted samples
        label.append(y)

    for i in sorted(index, reverse=True):  # delete samples with deleted label
        del label[i]
        del X[i]

    return label, X


def ratio_down(Y, X, perc):
    index_target = []
    for i in range(0, len(Y)):
        y = Y[i]
        if y == 1:
            index_target.append(i)

    percent = perc
    print('In training set:')
    print('number of target is', len(index_target) - round(len(index_target) * percent))
    print('number of non-target is', len(Y) - len(index_target))
    print('ratio of target is', (len(index_target) - round(len(index_target) * percent)) /
          (len(Y) - round(len(index_target) * percent)))

    for i in sorted(index_target[:round(len(index_target) * percent)], reverse=True):
        # remove the percent% of target samples
        del Y[i]
        # del X[i]        # error! cannot do this to dataframe
        X = X.drop(X.index[i])
    ratio = (len(index_target) - round(len(index_target) * percent)) / (len(Y) - round(len(index_target) * percent))

    return Y, X, ratio


def one_hot(df, index_num):
    """
    :param df: training_x or testset_x, type: data frame
    :param index_num: the index list of numerical features
    :return:
    """
    lb = LabelBinarizer()
    list_len = list(range(len(df.iloc[0])))
    index_onehot = list(set(list_len) - set(index_num))
    for i in index_onehot:
        df.iloc[:, i] = lb.fit_transform(df.iloc[:, 26]).tolist()
    return df


def distribution(y):
    '''
    :return:  get the distribution of y
    '''
    target_count = 0
    for i in y:
        if i == 1:
            target_count += 1
    ratio = target_count / len(y)

    return ratio


def model_init(testset_x):
    print("-----------------DNN----------------------")
    model = Sequential()  # creat model

    # get number of colums or features i testset data
    n_cols = testset_x.shape[1]

    # add the layers for model
    model.add(Dense(30, activation='relu', input_shape=(n_cols,)))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # compile model using accuracy to measure the performance
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model.summary()
    return model


def main(path, perc, seed=0):
    testing, list_training, list_header = open(path)
    # merge, common = merge1(testing, list_training, list_header)
    common_header = common_get(list_header)
    np.random.seed(seed)

    # training set
    training_trim = trim(list_training[0], common_header)  # ONLY USE THE VERSION 4 FOR TRAINING

    # testing set
    testing_trim = trim(testing, common_header)
    # testing_trim.iloc[0]   to check the type of each feature

    # training set
    training_x = training_trim.iloc[:, :-1]  # pandas.core.frame.DataFrame
    training_y = training_trim.iloc[:, -1]  # pandas.core.series.Series

    # testing set
    testset_x = testing_trim.iloc[:, :-1]
    testset_y = testing_trim.iloc[:, -1]

    # remove the samples in training and test set with label "deleted"
    training_y, training_x = preprocess1(training_y, training_x)
    testset_y, testset_x = preprocess1(testset_y, testset_x)
    training_y, training_x, ratio = ratio_down(training_y, training_x, perc=perc)

    le = preprocessing.LabelEncoder()
    # kaggle's forums where to find valuable information

    # normalize the x for training and test sets
    min_max_scaler = preprocessing.MinMaxScaler()
    scaler = MinMaxScaler()

    Recall = []
    AUC = []
    False_alarm = []
    F1 = []
    Accuracy = []
    Prec = []
    skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    skf1 = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
    # clf = model_init(testset_x)  # basic DNN
    svm = LinearSVC()
    clf = CalibratedClassifierCV(svm)
    for holdout_index, test_index in skf.split(testset_x, testset_y):
        x_holdout, x_test = testset_x.iloc[holdout_index, :], testset_x.iloc[test_index, :]
        y_holdout, y_test = pd.Series(testset_y)[holdout_index], pd.Series(testset_y)[test_index]

        '''
        # 1 . 80% randomly selected as training set in every loop
        l1 = list(range(0, len(training_x)))
        random.shuffle(l1)
        index = l1[:int(0.8*len(l1))]
        # get 80% of l1 as randomly selected index
        training_yinloop = [training_y[i] for i in index]
        training_x = np.asarray(training_x)
        training_xinloop = [training_x[i] for i in index]
        training_xinloop = np.array(training_xinloop)
        '''

        '''
        # 2 . training on whole set of release 4
        training_xinloop = training_x
        training_yinloop = training_y
        '''

        # 3. 5 by 5 designed by Sherry
        for training_index_train, holdout_index_train in skf1.split(training_x, training_y):
            training_xinloop, x_holdout_ = training_x.iloc[training_index_train, :], training_x.iloc[
                                                                                     holdout_index_train, :]
            training_yinloop, y_holdout_ = pd.Series(training_y)[training_index_train], pd.Series(training_y)[
                holdout_index_train]

            x_test = np.asarray(x_test).astype('float32')
            training_xinloop = np.asarray(training_xinloop).astype('float32')
            # training_yinloop = to_categorical(training_yinloop)
            clf.fit(training_xinloop, training_yinloop)
            # set early stopping monitor so the model stops training when it won't improve anymore
            early_stopping_monitor = EarlyStopping(patience=3)

            # class_weight = {0: 1.,
            #                 1: float(int(1 / ratio))}
            # clf.fit(training_xinloop, np.array(training_yinloop), epochs=100, validation_split=0.2,
            #         callbacks=[early_stopping_monitor])  # dnn model needs categorical label
            # validation_split means 20% for validation
            print("Training finished!")
            # visualization of training process
            # plot_curves(log)

            y_pred1 = clf.predict(x_test)
            # y_pred1 = clf.predict_classes(x_test)
            y_prob = clf.predict_proba(x_test)
            fpr, tpr, _ = metrics.roc_curve(y_test.values, y_prob[:, 1])
            auc = metrics.auc(fpr, tpr)
            # #######
            print(metrics.classification_report(y_test.values, y_pred1))
            # ##################
            y_pred1 = y_pred1.tolist()
            tn, fp, fn, tp = metrics.confusion_matrix(y_test.values.tolist(), y_pred1).ravel()
            precision = tp / (tp + fp + 1.e-5)
            recall = tp / (tp + fn + 1.e-5)
            pf = fp / (fp + tn + 1.e-5)  # false alarm
            f1 = 2 * precision * recall / (precision + recall + 1.e-5)
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # acc = accuracy_score(testset_y1, y_pred.round())
            Recall.append(recall)
            AUC.append(auc)
            False_alarm.append(pf)
            F1.append(f1)
            Accuracy.append(accuracy)
            Prec.append(precision)
    print("recall", Recall)
    print("f1", F1)
    print("false alarm", False_alarm)
    print("AUC score", AUC)
    print("accuracy", Accuracy)
    print("precision", Prec)

    auc = AUC[AUC != 0].mean()
    prec = Prec[Prec != 0].max()
    f1 = F1[F1 != 0].max()
    recall = Recall[Recall != 0].mean()
    false_alarm = False_alarm[False_alarm != 0].mean()

    print("average recall", recall)
    print("average f1", f1)
    print("average false alarm", false_alarm)
    print("average AUC score", auc)
    print("average accuracy", mean(Accuracy))
    print("average precision", prec)

    return auc, prec, f1, recall, false_alarm


if __name__ == "__main__":

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    projects = ['derby', 'mvn', 'lucence', 'phoenix', 'cass', 'jmeter', 'tomcat', 'ant', 'commons']
    # projects = ['derby', 'phoenix', 'cass', 'tomcat']
    repeated_time = 3
    # projects = ['lucence']
    stopats = [1]
    # how many percent of targets to cut down in training data to cut down
    perc_lists = [0]
    results = {}
    eval_metrics = ["auc", "precision", "f1", "recall", "false_alarm"]
    for stopat_id in stopats:
        # print("----------threshold stop at----------:", stopat_id)
        for project in projects:
            results[project] = {}
            for em in eval_metrics:
                results[project][em] = []
            start_time = time.time()
            # path = r'data/total_features/' + project
            path = r'data/' + project
            print("-----------------" + project + "----------------------")
            AUC = []
            cost = []
            Acc = []
            for i in range(1, repeated_time + 1):  # max-min times repeat
                svm = LinearSVC()
                clf = CalibratedClassifierCV(svm)
                seed = int(time.time() * 1000) % (2 ** 32 - 1)
                skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=seed)
                sss = StratifiedShuffleSplit(n_splits=5, test_size=.025, random_state=seed)
                for perc in perc_lists:
                    print("----------percentage of targets cut down----------:", perc * 100, "%")
                    # auc, prec, f1, recall, false_alarm = main(path, perc=perc,
                    #      seed=int(time.time() * 1000) % (2 ** 32 - 1))
                    training_x, training_y, testset_x, testset_y, columns = data_preparing(path, stop_at=stopat_id, perc=perc,
                                     seed=int(time.time() * 1000) % (2 ** 32 - 1))
                    training_x = pd.DataFrame(data=training_x, columns=columns)
                    testset_x = pd.DataFrame(data=testset_x, columns=columns)
                    for holdout_index, test_index in skf.split(testset_x, testset_y):
                        x_holdout, x_test = testset_x.iloc[holdout_index, :], testset_x.iloc[test_index, :]
                        y_holdout, y_test = pd.Series(testset_y)[holdout_index], pd.Series(testset_y)[test_index]

                        # 3. 5 by 5 designed by Sherry
                        for training_index_train, holdout_index_train in sss.split(training_x, training_y):
                            training_xinloop, x_holdout_ = training_x.iloc[training_index_train, :], training_x.iloc[
                                                                                                     holdout_index_train,
                                                                                                     :]
                            training_yinloop, y_holdout_ = pd.Series(training_y)[training_index_train], \
                                                           pd.Series(training_y)[
                                                               holdout_index_train]
                            try:
                                clf.fit(x_holdout_, y_holdout_)
                                # clf.fit(training_x, training_y)
                                print("Training finished!")
                                y_pred1 = clf.predict(testset_x)
                                # y_pred1 = clf.predict_classes(x_test)
                                y_prob = clf.predict_proba(testset_x)
                                fpr, tpr, _ = metrics.roc_curve(pd.Series(testset_y).values, y_prob[:, 1])
                                auc = metrics.auc(fpr, tpr)
                                # #######
                                print(metrics.classification_report(pd.Series(testset_y).values, y_pred1))
                                # ##################
                                y_pred1 = y_pred1.tolist()
                                tn, fp, fn, tp = metrics.confusion_matrix(pd.Series(testset_y).values.tolist(), y_pred1).ravel()
                                # y_pred1 = clf.predict(testset_x)
                                # y_prob = clf.predict_proba(testset_x)
                                # fpr, tpr, _ = metrics.roc_curve(testset_y, y_prob[:, 1])
                                # auc = metrics.auc(fpr, tpr)
                                # # #######
                                # print(metrics.classification_report(testset_y, y_pred1))
                                # # ##################
                                # y_pred1 = y_pred1.tolist()
                                # tn, fp, fn, tp = metrics.confusion_matrix(testset_y, y_pred1).ravel()
                                prec = tp / (tp + fp + 1.e-5)
                                recall = tp / (tp + fn + 1.e-5)
                                false_alarm = fp / (fp + tn + 1.e-5)  # false alarm
                                f1 = 2 * prec * recall / (prec + recall + 1.e-5)
                                accuracy = (tp + tn) / (tp + tn + fp + fn)
                            except:
                                auc = 0
                                prec = 0
                                recall = 0
                                false_alarm = 1
                                f1 = 0
                                accuracy = 0

                            results[project]["auc"].append(auc)
                            results[project]["precision"].append(prec)
                            results[project]["f1"].append(f1)
                            results[project]["recall"].append(recall)
                            results[project]["false_alarm"].append(false_alarm)
                print("running time", (time.time() - start_time) * 1 / repeated_time)
    print("*" * 50)
    for m in eval_metrics:
        print(m, end=", ")
    print()
    for p in projects:
        print(p, end=", ")
        for k in eval_metrics:
            print(np.mean(results[p][k]), end=", ")
        print()





