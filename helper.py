#funciones auxiliares
#!/usr/bin/env python
# coding: utf-8

import collections
import logging
import os
import os.path
import subprocess
from time import gmtime, strftime

import numpy as np
import pandas as pd
from tqdm import tqdm


# ------- LOGGING  ----------
def set_log_level(log_lvl):
    if log_lvl == "debug":
        logging.basicConfig(level=logging.DEBUG)
    if log_lvl == "info":
        logging.basicConfig(level=logging.INFO)
    if log_lvl == "warning":
        logging.basicConfig(level=logging.WARNING)
    if log_lvl == "error":
        logging.basicConfig(level=logging.ERROR)
    if log_lvl == "critical":
        logging.basicConfig(level=logging.CRITICAL)


def info(*msg):
    logger = logging.getLogger(__name__)
    logger.info(" ".join([str(x) for x in msg]))


def debug(*msg):
    logger = logging.getLogger(__name__)
    logger.debug(" ".join([str(x) for x in msg]))


def warning(*msg):
    logger = logging.getLogger(__name__)
    logger.warning(" ".join([str(x) for x in msg]))


def error(*msg):
    logger = logging.getLogger(__name__)
    logger.error(" ".join([str(x) for x in msg]))


def critical(*msg):
    global logger
    logger.critical(" ".join([str(x) for x in msg]))


set_log_level("info")

# ------- SYSTEM  ----------

def home():
    return os.path.expanduser("~")


def now():
    return strftime("%Y-%m-%d-%H:%M:%S", gmtime())


def mkdir_p(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def exists(fname):
    return os.path.exists(fname)


def run_command(cmd):
    try:
        return subprocess.check_output(cmd, shell=True).decode("utf-8")
    except subprocess.CalledProcessError as e:
        logger.error("ERROR running", str(cmd))
        raise e


def wav_duration(filename):
    # in seconds
    dur = run_command("soxi -D {}".format(filename))
    dur.strip()

    return float(dur)


# ------- LISTS  ----------

def read_list(filename, verbose=True):
    if not os.path.exists(filename):
        raise Exception("missing list: {}".format(filename))

    file = open(filename, "r")
    if verbose:
        debug("reading {}".format(filename))

    res = []
    lines = file.readlines()
    for line in lines:
        if line.strip() != "":
            chunks = line.split()
            if len(chunks) > 1:
                res.append(chunks)
            else:
                res.append(chunks[0])
    return list(res)


def save_list(lines, fname, verbose=True, separator="\t", append=False):
    def _multiple_line(line):
        return type(line) is list or type(line) is tuple or type(line) is np.ndarray
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(fname, mode) as fn:
        for line in lines:
            if _multiple_line(line):
                fn.write(separator.join([str(l) for l in line]) + "\n")
            else:
                fn.write(line + "\n")
    if verbose:
        debug("saving {}".format(fname))


# ------ PAPER SPECIFIC ------

def load_X_y(*args, **kargs):
    X, y, _ = load_X_y_and_instances(*args, **kargs)
    return X, y


def load_X_y_and_instances(instances_feature_list, y_column, columns_to_drop=[], value_for_nan=-15, limit=None):
    X = pd.DataFrame()
    y = pd.DataFrame()
    instances = []
    instances_list = read_list(instances_feature_list)
    for i, (instance_id, features_fname) in tqdm(enumerate(instances_list), desc="Loading dataset", total=len(instances_list)):
        instance = pd.read_csv(features_fname, index_col=None)
        y_i = instance.loc[:, y_column]
        X_i = instance.drop([c for c in columns_to_drop] + [y_column], axis=1)
        X = pd.concat((X, X_i))
        y = pd.concat((y, y_i))
        instances.append(instance_id)
        if limit and i > limit:
            break

    y.columns = [y_column]
    debug("Loaded instances and targets")
    debug("#Columns: ", len(X.columns))
    debug("Target: ", "\n".join(y.columns))
    debug("Target counts: ", collections.Counter(y[y_column].values))
    # debug("Missing percentages: ", 1 - (X.count() / len(X)))
    X = X.fillna(value_for_nan)
    debug("Filling missing values with: ", value_for_nan)
    y.reset_index(inplace=True, drop=True)
    X.reset_index(inplace=True, drop=True)

    instances = np.array(instances)
    return X, y, instances
