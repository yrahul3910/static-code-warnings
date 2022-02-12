from __future__ import division, print_function


import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from raise_utils.learners import LogisticRegressionClassifier, RandomForest
from raise_utils.metrics import ClassificationMetrics
from sklearn.model_selection import train_test_split

import random
import pdb
from collections import Counter
import re
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),\+!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\+", " \+ ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


class Treatment():

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = "Some Model"
        self.stats = {"tp": 0, "p": 0}

    def preprocess(self):
        self.x_content = []
        self.x_label = []
        self.y_label = []
        self.y_content = []
        for project in self.data:
            if project == self.target:
                self.y_label += [c for c in self.data[project]["label"]]
                self.y_content += [str(c)
                                   for c in self.data[project]["Abstract"]]
            else:
                self.x_content += [str(c)
                                   for c in self.data[project]["Abstract"]]
                self.x_label += [c for c in self.data[project]["label"]]
        # pdb.set_trace()
        # tfer = TfidfVectorizer(lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
        #                        sublinear_tf=False, decode_error="ignore",
        #                        max_features=5000)
        tfer = TfidfVectorizer(preprocessor=clean_str, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                               sublinear_tf=False, decode_error="ignore",
                               max_features=5000)
        self.train_data = tfer.fit_transform(self.x_content)
        self.test_data = tfer.transform(self.y_content)

        ascend = np.argsort(tfer.vocabulary_.values())
        self.voc = [list(tfer.vocabulary_.keys())[i] for i in ascend]
        # train_data_dense = list(sparse.csr_matrix(self.train_data))
        # train_data_dense = [np.array(x.todense()).flatten() for x in train_data_dense]
        # train_data_dense = np.array(train_data_dense)
        # self.full_train = pd.DataFrame(train_data_dense)
        # self.full_train["Label"] = [1 if x == "yes" else 0 for x in self.x_label]
        # self.full_test = pd.DataFrame(self.test_data)
        # self.full_test["Label"] = [1 if x == "yes" else 0 for x in self.y_label]
        self.full_train = self.convert_csr_to_dense(
            self.train_data, self.x_label)
        self.full_test = self.convert_csr_to_dense(
            self.test_data, self.y_label)

    def convert_csr_to_dense(self, data, label):
        data_dense = list(sparse.csr_matrix(data))
        data_dense = [np.array(x.todense()).flatten() for x in data_dense]
        data_dense = np.array(data_dense)
        data_dense_df = pd.DataFrame(data_dense)
        data_dense_df["Label"] = [1 if x == "yes" else 0 for x in label]
        return data_dense_df

    def get_raise_model(self):
        # if isinstance(self.model, LogisticRegression):
        if random.random() > 1.:
            return LogisticRegressionClassifier(random=True)
        # elif isinstance(self.model, RandomForestClassifier):
        else:
            return RandomForest(weighted=True, random=True)

    def train(self):
        # assert len(self.x_label)==len(self.x_content), "Size of training labels does not match training data."

        best_score = -1.
        best_model = None

        y = []
        for t in self.x_label:
            if t == 'yes':
                y.append(1)
            else:
                y.append(0)

        for _ in range(100):
            model = self.get_raise_model()
            model.set_data(self.train_data, y,
                           self.train_data, y)
            model.fit()
            preds = model.predict(self.train_data)
            m = ClassificationMetrics(y, preds)
            m.add_metric('f1')
            score = m.get_metrics()[0]

            if score > best_score:
                best_score = score
                best_model = model.learner
        self.model = model.learner

        self.model.fit(self.train_data, self.x_label)
        self.decisions = self.model.predict(self.test_data)
        pos_at = list(self.model.classes_).index("yes")
        try:
            self.probs = self.model.predict_proba(self.test_data)[:, pos_at]
        except:
            raise
            self.probs = self.model.decision_function(self.test_data)
            if pos_at == 0:
                self.probs = -self.probs

    def confusion(self, decisions):
        tp, fp, fn, tn = 0, 0, 0, 0
        if self.stats["p"] > 0:
            tp = self.stats['tp']
            fp = self.stats['p'] - tp
        for i, d in enumerate(decisions):
            gt = self.y_label[i]
            if d == "yes" and gt == "yes":
                tp += 1
            elif d == "yes" and gt == "no":
                fp += 1
            elif d == "no" and gt == "yes":
                fn += 1
            elif d == "no" and gt == "no":
                tn += 1
        return tp, fp, fn, tn

    def retrieval_curves(self, labels):
        stat = Counter(labels)
        t = stat["yes"]
        n = stat["no"]
        tp = 0
        fp = 0
        tn = n
        fn = t
        if self.stats["p"] > 0:
            tp = self.stats['tp']
            fp = self.stats['p'] - tp
        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]
        for label in labels:
            cost += 1.0
            costs.append(cost)
            if label == "yes":
                tp += 1.0
                fn -= 1.0
            else:
                fp += 1.0
                tn -= 1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)
        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps+fns)
        fpr = fps / (fps+tns)
        costr = costs / (t+n)
        return {"TPR": tpr, "FPR": fpr, "CostR": costr}

    def AUC(self, ys, xs):
        assert len(ys) == len(xs), "Size must match."
        x_last = 0
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[-1])
        auc = 0.0
        for i, x in enumerate(xs):
            y = ys[i]
            auc += y*(x-x_last)
            x_last = x
        return auc

    def eval(self):
        # assert len(self.y_label)==len(self.y_content), "Size of test labels does not match test data."
        tp, fp, fn, tn = self.confusion(self.decisions)
        result = {}
        if tp == 0:
            result["precision"] = 0
            result["recall"] = 0
            result["f1"] = 0
        else:
            result["precision"] = float(tp) / (tp+fp)
            result["recall"] = float(tp) / (tp+fn)
            result["f1"] = 2*result["precision"]*result["recall"] / \
                (result["precision"]+result["recall"])
        if fp == 0:
            result["fall-out"] = 0
        else:
            result["fall-out"] = float(fp) / (fp+tn)

        order = np.argsort(self.probs)[::-1]
        labels = np.array(self.y_label)[order]
        rates = self.retrieval_curves(labels)
        for r in rates:
            result[r] = rates[r]
        result["accuracy"] = float(tp + tn) / (tp + fp + tn + fn)
        result["AUC"] = self.AUC(rates["TPR"], rates["FPR"])
        result["APFD"] = self.AUC(rates["TPR"], rates["CostR"])
        result["p@10"] = Counter(labels[:10])["yes"] / float(len(labels[:10]))
        result["p@100"] = Counter(labels[:100])["yes"] / \
            float(len(labels[:100]))
        if result["recall"] + 1 - result["fall-out"] != 0:
            result["g1"] = (2 * result["recall"] * (1 - result["fall-out"])
                            ) / (result["recall"] + 1 - result["fall-out"])
        else:
            result["g1"] = 0
        return result


class SVM(Treatment):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = SGDClassifier(class_weight="balanced")


class RF(Treatment):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = RandomForestClassifier(class_weight="balanced_subsample")


class DT(Treatment):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = DecisionTreeClassifier(
            class_weight="balanced", max_depth=8)


class NB(Treatment):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = MultinomialNB()


class LR(Treatment):

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.model = LogisticRegression(class_weight="balanced")


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if len(t) < 20 and len(t) > 2]


class TM(Treatment):
    # Baseline model from Huang et al. 2018.
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def preprocess(self):
        self.x_content = []
        self.x_label = []

        for key in self.data:
            if key == self.target:
                self.y_content = [re.sub(r'[^a-zA-Z]', ' ', str(c))
                                  for c in self.data[key]["Abstract"]]
                self.y_label = self.data[key]["label"].tolist()
            else:
                self.x_content.append(
                    [re.sub(r'[^a-zA-Z]', ' ', str(c)) for c in self.data[key]["Abstract"]])
                self.x_label.append(self.data[key]["label"].tolist())
        '''
        for key in self.data:
            if key == self.target:
                self.y_content = [re.sub(r'[^a-zA-Z]', ' ', str(c)) for c in self.data[key]["Abstract"]]
                self.y_label = [self.data[key]["label"].tolist()]
            else:
                indices = []
                content = []
                index = 0
                for content in self.data[key]["Abstract"]:
                    try:
                        c = re.sub(r'[^a-zA-Z]', ' ', content)
                        if c.strip():
                            content.append(c)
                            indices.append(index)
                    except:
                        print("ill-formatted string", index)
                    index += 1
                self.x_label.append([self.data[key]["label"].tolist()])
                self.x_content.append([re.sub(r'[^a-zA-Z]', ' ', str(c)) for c in self.data[key]["Abstract"]])
        '''
        self.probs = np.array([0]*len(self.y_label))

    def train(self):
        def entropy(Nwt, N, Nt, Nw):
            return np.nan_to_num(Nwt/N*np.log2(Nwt*N/Nt/Nw))

        for i, content in enumerate(self.x_content):
            #  feature selection
            tfer = TfidfVectorizer(tokenizer=LemmaTokenizer(), lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                                   sublinear_tf=False, stop_words="english", decode_error="ignore")
            X = tfer.fit_transform(content)
            X[X != 0] = 1
            keys = np.array(list(tfer.vocabulary_.keys()))[
                np.argsort(list(tfer.vocabulary_.values()))]

            poses = np.where(np.array(self.x_label[i]) == "yes")[0]
            N = X.shape[0]
            Nt = len(poses)
            NT = N-Nt
            Nw = np.array(X.sum(axis=0))[0]
            NW = N-Nw
            Nwt = np.array(X[poses].sum(axis=0))[0]
            NWt = Nt-Nwt
            NwT = Nw-Nwt
            NWT = N-Nt-Nw+Nwt
            IG = entropy(Nwt, N, Nt, Nw)+entropy(NWt, N, Nt, NW) + \
                entropy(NwT, N, NT, Nw)+entropy(NWT, N, NT, NW)
            selected = keys[np.argsort(
                IG)[::-1][:int(X.shape[1]*0.1)]].tolist()

            # train model with selected features
            tfer = TfidfVectorizer(tokenizer=LemmaTokenizer(), lowercase=True, analyzer="word", norm=None, use_idf=False, smooth_idf=False,
                                   sublinear_tf=False, stop_words="english", decode_error="ignore", vocabulary=selected)
            X = tfer.fit_transform(content)
            X[X != 0] = 1
            Y = tfer.transform(self.y_content)
            Y[Y != 0] = 1
            model = MultinomialNB()
            model.fit(X, self.x_label[i])
            # get predictions
            self.probs += np.array([1 if l == "yes" else -
                                   1 for l in model.predict(Y)])

        self.decisions = ["yes" if votes > 0 else "no" for votes in self.probs]
