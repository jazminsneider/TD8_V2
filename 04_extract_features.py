#!/usr/bin/env python
# coding: utf-8

import os.path
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import helper
from tqdm import tqdm
import glob
import re
import os 
import ml.utils
import os
import subprocess
from pathlib import Path
import shutil
import pandas as pd
import ml.parsing.arff

import ml.opensmile
import os.path
import tempfile

import click
import ml.opensmile
import ml.parsing.arff
import ml.utils
import numpy as np
import pandas as pd
import helper
import configparser

#cambiar para adaptar:
from pathlib import Path
import tempfile
def main(ipus_table, output_folder, output_list, corpus, config, tracks_extraction_config, tasks_table, force):
    configuration = configparser.ConfigParser()
    configuration.read(config)
    corpus_config = configuration["corpus-{}".format(corpus)]

    output_folder = os.path.abspath(output_folder)
    helper.mkdir_p(output_folder)
    helper.mkdir_p(os.path.dirname(output_list))
    temp_folder = tempfile.mkdtemp()
    helper.mkdir_p(temp_folder)
    extraction_config = ml.utils.read_config(tracks_extraction_config)

    tasks = pd.read_csv(tasks_table)
    token_features_list = []
    ipus = pd.read_csv(ipus_table)

    for session_channel, group in ipus.groupby("session_channel"):
        session, channel = session_channel.split("_")
        session = int(session)
        tracks_output_filename = "{}/{}.csv".format(output_folder, session_channel)

        if not force and helper.exists(tracks_output_filename):
            helper.warning("{} already exist, skipping (use --force to overwrite)".format(tracks_output_filename))
        else:

            tracks = pd.DataFrame()
            for wav in sorted(set(group.wav_file)):
                print("Processing {}".format(wav))
                times, features = compute_tracks(wav, group.iloc[0].speaker_gender, extraction_config, temp_folder)
                tracks_i = pd.DataFrame(features)
                tracks_i["time"] = times
                tracks_i = tracks_i[~tracks_i.time.duplicated(keep="last")]
                tracks_i.set_index("time", inplace=True)

                vad_mask = np.zeros(tracks_i.index.shape)
                if "one_wav_per_task" in corpus_config:
                    task_id = int(wav.split("/")[-1].split(".")[-3])
                    group_for_mask = group[group.task == task_id]
                else:
                    group_for_mask = group

                for idx, ipu in group_for_mask.iterrows():
                    vad_mask[(tracks_i.index >= ipu.ipu_start_time) & (tracks_i.index <= ipu.ipu_end_time)] = 1
                tracks_i[vad_mask == 0] = np.nan
                tracks_i["vad"] = vad_mask
                tracks_i["time"] = tracks_i.index

                if "one_wav_per_task" in corpus_config:
                    tracks_i["task"] = task_id
                else:
                    tracks_i["task"] = None

                    for _, task in tasks[tasks.session == session].iterrows():
                        tracks_i.loc[(tracks_i.index >= task.t0) & (tracks_i.index <= task.tf), "task"] = task.task_number
                tracks = pd.concat([tracks, tracks_i], ignore_index=True)

            tracks = tracks[~tracks.task.isna()]
            tracks.to_csv(tracks_output_filename, index=False)

        token_features_list.append((session_channel, tracks_output_filename))

    helper.save_list(token_features_list, output_list)


def compute_tracks(filename, gender, extraction_config, temp_folder):
    if gender == "m":
        features_config = extraction_config["opensmile_config_male"]
    elif gender == "f":
        features_config = extraction_config["opensmile_config_female"]

    data = ml.opensmile.call_script(extraction_config["smile_extract_path"], temp_folder, features_config, filename)
    series = {}
    times = ml.parsing.arff.get_column(data, "frameTime")

    pitch = ml.parsing.arff.get_column(data, "F0final_sma")
    intensity = ml.parsing.arff.get_column(data, "pcm_intensity_sma")
    # loudness = ml.parsing.arff.get_column(data, "pcm_loudness_sma")
    jitter = ml.parsing.arff.get_column(data, "jitterLocal")
    shimmer = ml.parsing.arff.get_column(data, "shimmerLocal")
    logHNR = ml.parsing.arff.get_column(data, "logHNR")



    series["pitch"] = pitch
    series["intensity"] = intensity
    # series["loudness"] = loudness
    series["jitter"] = jitter
    series["shimmer"] = shimmer
    series["logHNR"] = logHNR

    return times, series


if __name__ == "__main__":
    main(ipus_table="csvs/ipus_uba.csv", output_folder="tracks", output_list="lists/tracks.lst", corpus="uba", config="configs/experiments.ini", tracks_extraction_config="configs/tracks_extraction.ini", tasks_table="csvs/tasks_uba.csv", force=True)