import numpy as np

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, positive_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall(y_true, y_pred, positive_label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(prec, rec):
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def macro_f1(y_true, y_pred, labels):
    f1_scores = []
    for label in labels:
        p = precision(y_true, y_pred, label)
        r = recall(y_true, y_pred, label)
        f1 = f1_score(p, r)
        f1_scores.append(f1)
    return np.mean(f1_scores)

def precision_per_class(y_true, y_pred, labels):
    precisions = []
    for label in labels:
        prec = precision(y_true, y_pred, label)
        precisions.append(prec)
    return precisions

def recall_per_class(y_true, y_pred, labels):
    recalls = []
    for label in labels:
        rec = recall(y_true, y_pred, label)
        recalls.append(rec)
    return recalls