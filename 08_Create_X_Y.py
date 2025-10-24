#!/usr/bin/env python
# coding: utf-8
import scipy.stats
import numpy as np
import os.path
import click
import pandas as pd
import configparser
import helper
from p_tqdm import p_map


WZ = 5  # Ms
TRACKS_SAMPLE_RATE = 0.0010
np.random.seed(1234)

def main(instances_list, output_folder, features_config, combination, time_interval, series_type, n_jobs, force):
    output_folder = os.path.abspath(output_folder)
    X_fname = os.path.join(output_folder, "X.csv")
    y_fname = os.path.join(output_folder, "y.csv")
    sessions_fname = os.path.join(output_folder, "sessions.csv")
    tasks_fname = os.path.join(output_folder, "tasks.csv")
    if not force and helper.exists(X_fname):
        helper.warning("{} already exist, skipping (use --force to overwrite)".format(X_fname))
        return

    helper.mkdir_p(output_folder)

    dev_instances = helper.read_list(instances_list)
    config = configparser.ConfigParser()
    config.read(features_config)
    
    # TEMP
    # VADS
    columns_temporal = config["features"]["TEMP"].split(" ")
    columns_temporal += config["features"]["VADS"].split(" ")

    # SUBJ
    # TASK
    # TURN
    # IPU
    columns_static = config["features"]["SUBJ"].split(" ")
    columns_static += config["features"]["TASK"].split(" ")
    columns_static += config["features"]["TURN"].split(" ")
    columns_static += config["features"]["IPU"].split(" ")
    columns_static += config["features"]["TRAN"].split(" ")

    features_combination = combination.split("+")
    static_features_to_use = []
    for categ in features_combination:
        if categ != "TEMP" and categ != "VADS":
            static_features_to_use += config["features"][categ].split(" ")

    if time_interval == "past":
        restrict_static_features = config["time"]["future"].split(" ")
    elif time_interval == "future":
        restrict_static_features = config["time"]["past"].split(" ")
    else:
        restrict_static_features = []

    for restricted in restrict_static_features:
        assert restricted in columns_static
        if restricted in static_features_to_use:
            static_features_to_use.remove(restricted)

    def process_instance_partial(x):
        (idx, fname_metadata) = x
        return process_instance(idx, fname_metadata, columns_static, static_features_to_use, features_combination, time_interval, config, series_type, columns_temporal)
    try:
        indices, X, y, sessions, tasks, x_i_colss = zip(*p_map(process_instance_partial, dev_instances, num_cpus=n_jobs))
    except:
        import ipdb; ipdb.set_trace()

    X = pd.DataFrame(X, index=indices, columns=x_i_colss[0])
    y = pd.DataFrame(y, index=indices)
    sessions = pd.DataFrame(sessions, index=indices)
    tasks = pd.DataFrame(tasks, index=indices)

    X.to_csv(X_fname)
    y.to_csv(y_fname)
    sessions.to_csv(sessions_fname)
    tasks.to_csv(tasks_fname)

def flatten_df(df, suffix=""):
    values = df.values.flatten("C")
    if df.shape[0] > 1:
        columns = ["{}-{}{}".format(i, c, suffix) for i in range(df.shape[0]) for c in df.columns]
    else:
        columns = df.columns
    return values, columns

def slope(x, y):
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]
    return scipy.stats.linregress(x, y)[0]


def sliding_window(df, window_size, func, step, min_points):
    res = []
    for i in range(0, len(df), window_size):
        rows = df.iloc[i:i + window_size]
        if func == "mean":
            res_i = rows.apply(lambda y: np.nanmean(y) if sum(~np.isnan(y)) >= min_points else np.nan)
        elif func == "slope":
            x = rows.index
            res_i = rows.apply(lambda y: slope(x, y) if sum(~np.isnan(y)) >= min_points else np.nan)
        res.append(res_i)
    res = pd.DataFrame(res)
    return res


def process_instance(idx, fname_metadata, columns_static, static_features_to_use, features_combination, time_interval, config, series_type, columns_temporal):
    metadata = pd.read_csv(fname_metadata, index_col="transition_idx").iloc[0]
    static_feats = pd.read_csv(metadata["static_features_fname"])
    assert len(set(static_feats.columns) - set(columns_static)) == 0, "diff: " + str(set(static_feats.columns) - set(columns_static))

    static_feats = static_feats.loc[:, static_features_to_use]

    x_i, x_i_cols = flatten_df(static_feats)

    if "TEMP" in features_combination or "VADS" in features_combination:
        temporal_feats = pd.read_csv(metadata["temporal_features_fname"], index_col="time")
        assert len(set(temporal_feats.columns) - set(columns_temporal)) == 0, "diff: " + str(set(temporal_feats.columns) - set(columns_temporal))

        if time_interval == "future":
            temporal_feats = temporal_feats.tail(temporal_feats.shape[0] // 2)

        if time_interval == "past":
            temporal_feats = temporal_feats.head(temporal_feats.shape[0] // 2)

        temporal_feats.index = np.arange(temporal_feats.shape[0])
        vad_columns = config["features"]["VADS"].split(" ")
        vads = temporal_feats.loc[:, vad_columns]
        temporal_feats = temporal_feats.drop(vad_columns, axis=1)

        if "VADS" in features_combination:
            x_i_vads, x_i_vads_cols = flatten_df(vads)
            x_i = np.concatenate([x_i, x_i_vads])
            x_i_cols = np.concatenate([x_i_cols, x_i_vads_cols])

        if "TEMP" in features_combination:
            if series_type == "windowed":
                means = sliding_window(temporal_feats, window_size=WZ, func="mean", step=WZ, min_points=2)
                means.index = np.arange(means.shape[0])
                x_i_means, x_i_means_cols = flatten_df(means, suffix="-mean")
                x_i = np.concatenate([x_i, x_i_means])
                x_i_cols = np.concatenate([x_i_cols, x_i_means_cols])

                slopes = sliding_window(temporal_feats, window_size=WZ, func="slope", step=WZ, min_points=2)
                slopes.index = np.arange(slopes.shape[0])
                x_i_slopes, x_i_slopes_cols = flatten_df(slopes, suffix="-slope")
                x_i = np.concatenate([x_i, x_i_slopes])
                x_i_cols = np.concatenate([x_i_cols, x_i_slopes_cols])
            else:
                x_i_temp, x_i_temp_cols = flatten_df(temporal_feats)
                x_i = np.concatenate([x_i, x_i_temp])
                x_i_cols = np.concatenate([x_i_cols, x_i_temp_cols])

    y_i = metadata.tt_label
    task_number = "{}-{}".format(metadata.session_number, metadata.task_number)
    session_number = metadata.session_number
    return (idx, x_i, y_i, session_number, task_number, x_i_cols)


if __name__ == "__main__":
    kind = "no_overlap"  # or "no_overlap" 
    main(instances_list="lists/dev_instances_overlap.lst", output_folder = "X_Y/overlap/dev", features_config="configs/features.ini", combination="IPU+SUBJ+TEMP+TURN", time_interval="both", series_type="raw", n_jobs=3, force=True)
