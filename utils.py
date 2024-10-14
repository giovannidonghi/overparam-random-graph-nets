import os
from datetime import datetime
import torch
import numpy as np
import random
import torch
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression

from sklearn.svm import LinearSVC, SVC

def add_dummy_labels(data):
    if data.x is None:
        data.x = torch.ones(data.num_nodes, 1)
    return data

def split_ids(ids, folds=10):
    n = len(ids)
    stride = int(np.ceil(n / float(folds)))
    test_ids = [ids[i: i + stride] for i in range(0, n, stride)]

    assert np.all(
        np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
    assert len(test_ids) == folds, 'invalid test sets'
    valid_ids = []
    train_ids = []

    for fold in range(folds):
        valid_fold = []
        while len(valid_fold) < stride:
            id = random.choice(ids)
            if id not in test_ids[fold] and id not in valid_fold:
               valid_fold.append(id)

        valid_ids.append(np.asarray(valid_fold))
        train_ids.append(np.array([e for e in ids if e not in test_ids[fold] and e not in valid_ids[fold]]))
        assert len(train_ids[fold]) + len(test_ids[fold]) + len(valid_ids[fold]) == len(np.unique(list(train_ids[fold]) + list(test_ids[fold]) + list(valid_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids, valid_ids

def printParOnFile(test_name, log_dir, par_list):
    
    assert isinstance(par_list, dict), "par_list has to be a dictionary"
    with open(os.path.join(log_dir, test_name+".log"), 'w+') as f:
        f.write(test_name)
        f.write("\n")
        f.write(str(datetime.now().utcnow()))
        f.write("\n\n")
        for key, value in par_list.items():
            f.write(str(key)+": \t"+str(value))
            f.write("\n")

def train_rocket_fixed_alpha(X_train_transform,X_valid_transform, X_test_transform, y_train, y_valid, y_test,alpha=0, max_iter=100, optim="ridgeclassifier", eval=None):

  if optim == "ridgeclassifier" or optim == "ridgeclassifierzero":
    if alpha == 0:
       solver = "lsqr"
    else:
       solver = "auto"
    classifier = RidgeClassifier(alpha=alpha, max_iter=max_iter, solver=solver)
  elif optim == "linearSVC":
    classifier = LinearSVC(C=alpha, max_iter=max_iter)
  elif optim == "SGD":
    classifier = SGDClassifier(alpha=alpha, max_iter=max_iter)
  elif optim=="logisticregression":
    classifier = LogisticRegression(C=alpha, max_iter=max_iter)
  elif optim == "SVC":
    classifier = SVC(C=alpha, kernel='linear', max_iter=max_iter, gamma='auto') # rbf kernel


  classifier.fit(X_train_transform, y_train.ravel())
  #different scroing function for OGB
  if eval is not None:
        preds_val = classifier.predict(X_valid_transform).reshape(-1, 1)
        preds_test = classifier.predict(X_test_transform).reshape(-1, 1)
        val_score=eval(preds_val,y_valid)['rocauc']
        test_score=eval(preds_test,y_test)['rocauc']
  else:
        train_score= classifier.score(X_train_transform, y_train)
        val_score=classifier.score(X_valid_transform, y_valid)
        test_score=classifier.score(X_test_transform, y_test)
        coeff_norm= np.linalg.norm(classifier.coef_)
    
  return train_score, test_score, val_score, coeff_norm

def calculate_entropy_row(i, data_matrix, n_features):
    row = data_matrix[i, :]
    diff_matrix = row[:, np.newaxis] - row[np.newaxis, :]
    rv = norm(loc=0, scale=0.3 * row.std())
    entropy = -np.log(np.sum(rv.pdf(diff_matrix)) / n_features ** 2)
    return entropy

def renyi_entropy(data_matrix):
    n_samples, n_features = data_matrix.shape
    entropies = np.empty(n_samples)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_entropy_row, i, data_matrix, n_features) for i in range(n_samples)]

        for i, future in enumerate(futures):
            entropies[i] = future.result()

    return entropies.mean(), entropies.std()

def X_by_layer(X, n_layers):
  x1, x2 = np.hsplit(X, 2)
  X_new = np.empty((n_layers, X.shape[0], X.shape[1]//n_layers))
  l1 = np.hsplit(x1, n_layers)
  l2 = np.hsplit(x2, n_layers)
  for l in range(n_layers):
    X_new[l,:,:] = np.concatenate([l1[l], l2[l]], axis=1)
  return X_new

def random_subarray(original_array, row_fraction=0.5, max_columns=None, seed=None):
    np.random.seed(seed) if seed is not None else None

    num_rows = original_array.shape[0]
    selected_rows = np.random.choice(num_rows, size=int(row_fraction * num_rows), replace=False)
    num_columns = original_array.shape[1]
    if max_columns is None or max_columns >= num_columns:
        selected_columns = np.arange(num_columns)
    else:
        selected_columns = np.random.choice(num_columns, size=max_columns, replace=False)
    subarray = original_array[selected_rows][:, selected_columns]
    return subarray


