#!/usr/bin/env python
# coding: utf-8
from sklearn import preprocessing
import numpy as np
import os.path
import click
import pandas as pd
from tqdm import tqdm
import helper
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import sklearn.metrics
from joblib import dump, load


FILL_NA_WITH = -15
N_ESTIMATORS = 300
MAX_DEPTH = 10
MAX_FEATURES = 0.5

def main(data_folder, model, balance_method, output_folder, n_jobs, force, save):
    output_folder = os.path.abspath(output_folder)

    helper.mkdir_p(output_folder)
    X = pd.read_csv(os.path.join(data_folder, "X.csv"), index_col=0)
    X_columns = X.columns

    indices = X.index
    X = X.fillna(FILL_NA_WITH).values.astype(np.float32)
    y_true = pd.read_csv(os.path.join(data_folder, "y.csv"), index_col=0)
    sessions = pd.read_csv(os.path.join(data_folder, "sessions.csv"), index_col=0).values.squeeze()
    # tasks = pd.read_csv(os.path.join(data_folder, "tasks.csv"), index_col=0).values.squeeze()
    np.random.seed(1234)
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y_true.values.squeeze())
    group_cross_val = LeaveOneGroupOut()

    for (train_positions, val_positions) in tqdm(group_cross_val.split(X, y, groups=sessions), total=group_cross_val.get_n_splits(groups=sessions)):
        val_sess = set(sessions[val_positions])
        assert len(val_sess) == 1
        fold_id = "fold_for_session_{}".format(list(val_sess)[0])
        
        fold_folder = os.path.join(output_folder, fold_id)
        output_fname = os.path.join(fold_folder, "val_probas.csv")

        if not force and helper.exists(output_fname):
            helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
            continue

        helper.mkdir_p(fold_folder)
        helper.info(f"Saving models in {output_folder}")

        X_train = X[train_positions]
        y_train = y[train_positions]
        idx_train = indices[train_positions]

        if any([c.endswith("_A") for c in X_columns]):
            # Mirror channels
            X_train_mirror = X_train.copy()
            for col_index, col in enumerate(X_columns):
                if col.endswith("_A"):
                    other_col = col.replace("_A", "_B")
                    other_col_index = list(X_columns).index(other_col)
                    X_train_mirror[:, col_index] = X_train[:, other_col_index].copy()
                    X_train_mirror[:, other_col_index] = X_train[:, col_index].copy()
            X_train = np.concatenate([X_train, X_train_mirror])
            y_train = np.concatenate([y_train, y_train])
            idx_train = np.concatenate([idx_train, idx_train])
        X_val = X[val_positions]
        # y_val = y[val_positions]
        idx_val = indices[val_positions]

        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, max_features=MAX_FEATURES, n_jobs=n_jobs)

        if balance_method == "oversample":
            resampling_indices, _ = RandomOverSampler().fit_resample(X=np.arange(len(y_train)).reshape(-1, 1), y=y_train)
            X_train, y_train = X_train[resampling_indices.squeeze()], y_train[resampling_indices.squeeze()]
            idx_train = idx_train[resampling_indices.squeeze()]

        if balance_method == "smote":
            if min([v for (c, v) in Counter(y_train).items()]) < 6:
                for classs, count in Counter(y_train).items():
                    if count < 6:
                        helper.warning("could not apply SMOTE directly (#instances={}), RUNNING OVERSAMPLER FIRST".format(Counter(y_train)))
                        resampling_indices = np.concatenate([np.arange(len(y_train)), np.arange(len(y_train))[y_train == classs].repeat(6 // count)])
                        X_train, y_train = X_train[resampling_indices], y_train[resampling_indices]
                        idx_train = idx_train[resampling_indices]

            resampling_indices, _ = SMOTE().fit_resample(X=np.arange(len(y_train)).reshape(-1, 1), y=y_train)
            X_train, y_train = X_train[resampling_indices.squeeze()], y_train[resampling_indices.squeeze()]
            idx_train = idx_train[resampling_indices.squeeze()]

        if balance_method == "class_weights":
            weights = get_class_weights(y_train)
            print("weights", weights)
            clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight=weights, max_features=MAX_FEATURES, n_jobs=n_jobs)

        clf.fit(X_train, y_train)
        y_probas_val = clf.predict_proba(X_val)
        # y_probas_train = clf.predict_proba(X_train)
        val_probas = pd.DataFrame(y_probas_val, index=idx_val, columns=[le.classes_[c] for c in clf.classes_])
        for c in le.classes_:
            if c not in val_probas.columns:
                val_probas[c] = 0.
        # train_probas = pd.DataFrame(y_probas_train, index=idx_train, columns=le.classes_)

        val_probas.to_csv(output_fname)

        # train_probas.to_csv(os.path.join(fold_folder, "train_probas.csv"))
        if save:
            helper.info("Saving model in", os.path.join(fold_folder, 'clf.joblib'))
            dump(clf, os.path.join(fold_folder, 'clf.joblib'))

    y_true.to_csv(os.path.join(output_folder, 'y.csv'))
    text_file = open(os.path.join(output_folder, 'finished.txt'), "w")
    text_file.write("done")
    text_file.close()

    if save:
        clf.fit(X, y)
        dump(clf, os.path.join(output_folder, 'clf.joblib'))


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}



if __name__ == "__main__":
    main(data_folder="X_Y/no_overlap/dev/", model="no_overlap", balance_method="oversample", output_folder="Results/F1_NO_OVER", n_jobs=4, force=True,save=True)
    #main(data_folder="X_Y/overlap/dev/", model="overlap", balance_method="oversample", output_folder="Results/F1_OVER", n_jobs=4, force=True,save=True)
