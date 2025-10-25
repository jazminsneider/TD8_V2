#!/usr/bin/env python
# coding: utf-8
import os.path
import click
import pandas as pd
import helper
import configparser
import numpy as np
from tqdm import tqdm

TIMEPOINTS_STEP = 0.01
def main(ipus_table, kind, tt_table, config, corpus, tasks_table, tasks_features_list, dev_output_folder, held_out_output_folder, dev_output_list, held_out_output_list, force):

    helper.mkdir_p(os.path.dirname(dev_output_list))
    helper.mkdir_p(os.path.dirname(held_out_output_list))
    dev_output_folder = os.path.abspath(dev_output_folder)
    held_out_output_folder = os.path.abspath(held_out_output_folder)

    helper.mkdir_p(dev_output_folder)
    helper.mkdir_p(held_out_output_folder)

    ipus = pd.read_csv(ipus_table, index_col="token_id")
    tasks = pd.read_csv(tasks_table)

    configuration = configparser.ConfigParser()
    configuration.read(config)
    corpus_config = configuration["corpus-{}".format(corpus)]
    allowed_tt_labels = configuration["global"]["allowed_tt_labels"].split(" ")

    if corpus_config["held_out_tasks"] != "":
        held_out_tasks = [(int(x.split("t")[0][1:]), int(x.split("t")[1])) for x in corpus_config["held_out_tasks"].split(" ")]
        held_out_sessions = [int(x[1:]) for x in corpus_config["held_out_sessions"].split(" ")]
    else:
        held_out_tasks = []
        held_out_sessions = []

    tt_table_df = pd.read_csv(tt_table)
    tt_table_df = tt_table_df[tt_table_df.tt_label.isin(allowed_tt_labels)]

    if kind == "overlap":
        tt_table_df = tt_table_df[tt_table_df.overlapped_transition]
    else:
        tt_table_df = tt_table_df[~tt_table_df.overlapped_transition]

    tt_table_dev_set = tt_table_df[tt_table_df.apply(lambda x: (x['session_number'] not in held_out_sessions) and ((x['session_number'], x['task']) not in held_out_tasks), axis=1)]
    tt_table_held_out_set = tt_table_df[tt_table_df.apply(lambda x: (x['session_number'] in held_out_sessions) or ((x['session_number'], x['task']) in held_out_tasks), axis=1)]

    instances_config = configuration["instances"]

    dev_list = create_instances(tt_table_dev_set, dev_output_folder, tasks_features_list, tasks, ipus, instances_config, kind, force)
    held_out_list = create_instances(tt_table_held_out_set, held_out_output_folder, tasks_features_list, tasks, ipus, instances_config, kind, force)
    helper.save_list(dev_list, dev_output_list)
    helper.save_list(held_out_list, held_out_output_list)


def create_instances(tt_table, output_folder, tasks_features_list, tasks, ipus, instances_config, kind, force):
    tokens_list = []

    for sess_task, sess_task_feats in tqdm(helper.read_list(tasks_features_list)):
        sess, task_id = sess_task.split("_")

        turn_transitions_for_task = tt_table[(tt_table.session_number == int(sess)) & (tt_table.task == int(task_id))]
        ipus_for_task = ipus[(ipus.session_number == int(sess)) & (ipus.task == int(task_id))]
        feats = pd.read_csv(sess_task_feats, index_col="time")

        task_t0, task_tf = feats.index.min(), feats.index.max()

        vads = {}
        vads["A"] = feats.vad_A
        vads["B"] = feats.vad_B

        # A turn is a maximal sequence of IPUs from one speaker, such that between any two
        # adjacent IPUs there is no speech from the interlocutor

        for idx, turn_transition in turn_transitions_for_task.iterrows():
            tt_label = turn_transition.tt_label
            speaker_2_channel = turn_transition.speaker2
            speaker_1_channel = "B" if speaker_2_channel == "A" else "A"

            assert speaker_2_channel == "A" or speaker_2_channel == "B"
            assert speaker_2_channel != speaker_1_channel, f"{speaker_1_channel}, {speaker_2_channel}"

            speaker_1_ipus = ipus_for_task[(ipus_for_task.channel == speaker_1_channel)]
            speaker_2_ipus = ipus_for_task[(ipus_for_task.channel == speaker_2_channel)]

            speaker_1_ipus = speaker_1_ipus.sort_values(by="ipu_start_time").copy()

            speaker_2_ipus = speaker_2_ipus.sort_values(by="ipu_start_time").copy()


            ipu2_t0 = turn_transition.ipu2_start_time
            ipu2_tf = turn_transition.ipu2_end_time

            speaker_1_previous_ipus = speaker_1_ipus[speaker_1_ipus.ipu_start_time < ipu2_t0]
            speaker_2_next_ipus = speaker_2_ipus[speaker_2_ipus.ipu_start_time >= ipu2_t0 - TIMEPOINTS_STEP]

            # speaker_2_previous_ipus = speaker_2_ipus[(ipus_for_task.ipu_start_time < ipu2_t0)]

            if len(speaker_1_previous_ipus) == 0:
                #helper.warning("Ignoring instance {} (no previous IPUs) (t={}, sess={}, task={}, tt_label={})".format(idx, ipu2_t0, sess, task_id, tt_label))
                continue

            assert len(speaker_2_next_ipus) != 0, "MISSING IPU2"

            # helper.info(f"TT_LABEL: {tt_label} IPU2_t0 = {ipu2_t0} ({sess_task})")

            speaker_1_prev_ipu = speaker_1_previous_ipus.tail(1).iloc[0]
            prev_ipu_tf = speaker_1_prev_ipu.ipu_end_time

            speaker_2_next_ipu = speaker_2_next_ipus.iloc[0]
            next_ipu_t0 = round(ipu2_t0, 4)
            next_ipu_tf = round(ipu2_tf, 4)

            if kind == "no_overlap" and not (prev_ipu_tf < next_ipu_t0):
                #helper.error("non-overlap transition with prev_ipu_tf > next_ipu_t0")
                continue

            if kind == "overlap" and not (prev_ipu_tf > next_ipu_t0):
                #helper.error("overlap transition with prev_ipu_tf < next_ipu_t0")
                continue

            # PREVIOUS AND NEXT TURN STATISTICS
            speaker_1_vad = vads[speaker_1_channel].astype(bool)
            speaker_2_vad = vads[speaker_2_channel].astype(bool)

            consistent = consistent_vads(speaker_1_vad, speaker_2_vad, prev_ipu_tf, next_ipu_t0, kind)
            if not consistent:
                helper.warning("Discarding instance (no consistent VAD)")
                continue

            fname = "s-{}_task-{}_ipu1tf-{}_ipu2t0-{}_speaker2-{}_ttlabel-{}.csv".format(turn_transition.session_number, turn_transition.task, prev_ipu_tf, next_ipu_t0, speaker_2_channel, tt_label)
            fname = os.path.join(output_folder, fname)

            output_instance_metadata_fname = fname.replace(".csv", ".metadata.csv")
            output_temporal_features_fname = fname.replace(".csv", ".temporal.csv")
            output_static_features_fname = fname.replace(".csv", ".static.csv")

            tokens_list.append((idx, output_instance_metadata_fname))

            if not force and helper.exists(output_instance_metadata_fname):
                helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_instance_metadata_fname))
                continue

            speaker_1_previous_turn_total_speech_time, speaker_1_previous_turn_ipus_count, speaker_1_previous_turn_duration = speaker_1_previous_turn_metrics(speaker_1_vad, speaker_2_vad, speaker_1_previous_ipus, prev_ipu_tf)
            speaker_2_next_turn_total_speech_time, speaker_2_next_turn_ipus_count, speaker_2_next_turn_duration = speaker_2_next_turn_metrics(speaker_1_vad, speaker_2_vad, speaker_2_next_ipus, next_ipu_t0)

            # static features
            static_features = {}
            static_features["speaker_1_gender_is_male"] = speaker_1_prev_ipu.speaker_gender == "m"
            static_features["speaker_2_gender_is_male"] = speaker_2_next_ipu.speaker_gender == "m"
            static_features["speaker_1_is_describing"] = speaker_1_prev_ipu.task_describer == speaker_1_prev_ipu.channel

            static_features["transition_direction"] = 1 if speaker_2_channel == "A" else -1

            if kind == "overlap":
                static_features["gap_duration"] = round(next_ipu_t0 - prev_ipu_tf, 4)
            else:
                static_features["gap_duration"] = round(next_ipu_t0 - min(prev_ipu_tf, next_ipu_tf), 4)

            if kind == "overlap":
                prev_ipu_tf = next_ipu_t0

            static_features["time_since_task_begining"] = prev_ipu_tf - task_t0
            static_features["time_to_task_ending"] = task_tf - next_ipu_t0

            # prev and next IPU information
            static_features["speaker_1_prev_ipu_duration"] = speaker_1_prev_ipu.duration
            static_features["speaker_1_prev_ipu_words_by_second"] = speaker_1_prev_ipu.words_by_second
            static_features["speaker_1_prev_ipu_phones_by_second"] = speaker_1_prev_ipu.phones_by_second

            static_features["speaker_2_next_ipu_duration"] = speaker_2_next_ipu.duration
            static_features["speaker_2_next_ipu_words_by_second"] = speaker_2_next_ipu.words_by_second
            static_features["speaker_2_next_ipu_phones_by_second"] = speaker_2_next_ipu.phones_by_second

            # prev and next TURN information
            static_features["speaker_1_previous_turn_duration"] = speaker_1_previous_turn_duration
            static_features["speaker_1_previous_turn_total_speech_time"] = speaker_1_previous_turn_total_speech_time
            static_features["speaker_1_previous_turn_ipus_count"] = speaker_1_previous_turn_ipus_count

            static_features["speaker_2_next_turn_total_speech_time"] = speaker_2_next_turn_total_speech_time
            static_features["speaker_2_next_turn_ipus_count"] = speaker_2_next_turn_ipus_count
            static_features["speaker_2_next_turn_duration"] = speaker_2_next_turn_duration

            fixed_window_temporal_features = fixed_window_tracks(next_ipu_t0, prev_ipu_tf, instances_config, feats)

            fixed_window_temporal_features.to_csv(output_temporal_features_fname)

            static_features = pd.DataFrame([static_features])
            static_features.to_csv(output_static_features_fname, index=False)

            instance_metadata = {}
            instance_metadata["tt_label"] = tt_label
            instance_metadata["transition_idx"] = idx
            instance_metadata["wav_file"] = speaker_1_prev_ipu.wav_file
            instance_metadata["speaker_1_channel"] = speaker_1_prev_ipu.channel
            instance_metadata["temporal_features_fname"] = output_temporal_features_fname
            instance_metadata["static_features_fname"] = output_static_features_fname
            instance_metadata["session_number"] = int(sess)
            instance_metadata["task_number"] = int(task_id)
            instance_metadata["prev_ipu_tf"] = prev_ipu_tf
            instance_metadata["next_ipu_t0"] = next_ipu_t0

            instance_metadata = pd.DataFrame([instance_metadata])
            instance_metadata.to_csv(output_instance_metadata_fname, index=False)

    return tokens_list


def consistent_vads(speaker_1_vad, speaker_2_vad, prev_ipu_tf, next_ipu_t0, kind):
    valid = True
    if kind == "overlap":
        valid = valid and (speaker_1_vad[speaker_1_vad.index <= next_ipu_t0].iloc[-1])  # Speech on last frame before turn exchange.
        valid = valid and (speaker_1_vad[speaker_1_vad.index > next_ipu_t0].iloc[0])
        valid = valid and (~speaker_2_vad[speaker_2_vad.index < next_ipu_t0].iloc[-1])
        valid = valid and (speaker_2_vad[speaker_2_vad.index >= next_ipu_t0].iloc[0])
    else:
        valid = valid and (speaker_1_vad[speaker_1_vad.index <= prev_ipu_tf].iloc[-1])  # Speech on last frame before turn exchange.
        valid = valid and (~speaker_1_vad[speaker_1_vad.index >= next_ipu_t0].iloc[0])
        valid = valid and (~speaker_2_vad[speaker_2_vad.index <= prev_ipu_tf].iloc[-1])
        valid = valid and (speaker_2_vad[speaker_2_vad.index >= next_ipu_t0].iloc[0])
    return valid


def fixed_window_tracks(from_t, to_t, instances_config, feats):
    seconds_before = float(instances_config["seconds_before"])
    seconds_after = float(instances_config["seconds_after"])

    from_t = round(from_t, 2)
    to_t = round(to_t, 2)

    fixed_window_t0 = round(to_t - seconds_before + TIMEPOINTS_STEP, 2)
    instance_prev = feats[fixed_window_t0:to_t]
    datapoints_prev = int(seconds_before * 1 / TIMEPOINTS_STEP)

    if len(instance_prev) != datapoints_prev:
        # PADDING LEFT
        new_index = []
        min_t = instance_prev.index.min()
        for _ in range(datapoints_prev - len(instance_prev)):
            min_t = min_t - TIMEPOINTS_STEP
            new_index.append(round(min_t, 2))
        new_index = [x for x in reversed(new_index)] + list(instance_prev.index)
        instance_prev = instance_prev.reindex(index=new_index, method="bfill")

    fixed_window_tf = round(from_t + seconds_after - TIMEPOINTS_STEP, 2)
    instance_post = feats[from_t:fixed_window_tf]
    datapoints_post = int(seconds_before * 1 / TIMEPOINTS_STEP)

    if len(instance_post) != datapoints_post:
        # PADDING RIGHT
        new_index = []
        max_t = instance_post.index.max()
        for _ in range(datapoints_post - len(instance_post)):
            max_t = max_t + TIMEPOINTS_STEP
            new_index.append(round(max_t, 2))

        new_index = [x for x in reversed(new_index)] + list(instance_post.index)
        instance_post = instance_post.reindex(index=new_index, method="ffill")

    instance = pd.concat([instance_prev, instance_post])
    if (instance.shape[0] != datapoints_prev + datapoints_post):
        helper.error("instance.shape[0] != datapoints_prev + datapoints_post")
        raise "Wrong dimensions"

    return instance


def speaker_1_previous_turn_metrics(speaker_1_vad, speaker_2_vad, speaker_1_previous_ipus, prev_ipu_tf):
    speaker_1_prev_vad = speaker_1_vad[:prev_ipu_tf]
    speaker_2_prev_vad = speaker_2_vad[:prev_ipu_tf]
    last_turn_interruption = speaker_1_prev_vad[speaker_2_prev_vad & ~speaker_1_prev_vad].index.max()  # Last time speaker 1 was "interrupted" during silence
    if np.isnan(last_turn_interruption):
        # No interruptions
        previous_turn_start = speaker_1_previous_ipus.ipu_start_time.min()
    else:
        activity_post_interruption = speaker_1_prev_vad[last_turn_interruption:]
        previous_turn_start = activity_post_interruption[activity_post_interruption].index.min()

    speaker_1_previous_turn_total_speech_time = speaker_1_prev_vad[previous_turn_start:].sum() * TIMEPOINTS_STEP
    speaker_1_previous_turn_ipus_count = len(speaker_1_previous_ipus[speaker_1_previous_ipus.ipu_start_time + TIMEPOINTS_STEP >= previous_turn_start])
    speaker_1_previous_turn_duration = prev_ipu_tf - previous_turn_start

    return speaker_1_previous_turn_total_speech_time, speaker_1_previous_turn_ipus_count, speaker_1_previous_turn_duration


def speaker_2_next_turn_metrics(speaker_1_vad, speaker_2_vad, speaker_2_next_ipus, next_ipu_t0):
    speaker_1_next_vad = speaker_1_vad[next_ipu_t0:]
    speaker_2_next_vad = speaker_2_vad[next_ipu_t0:]

    next_turn_interruption = speaker_2_next_vad[speaker_1_next_vad & ~speaker_2_next_vad].index.min()  # Last time speaker 1 was "interrupted" during silence
    if np.isnan(next_turn_interruption):
        next_turn_ending = speaker_2_next_ipus.ipu_start_time.min()
    else:
        activity_prev_interruption = speaker_2_next_vad[:next_turn_interruption]
        next_turn_ending = activity_prev_interruption[activity_prev_interruption].index.max()

    speaker_2_next_turn_total_speech_time = speaker_2_next_vad[:next_turn_ending].sum() * TIMEPOINTS_STEP
    speaker_2_next_turn_ipus_count = len(speaker_2_next_ipus[speaker_2_next_ipus.ipu_end_time - TIMEPOINTS_STEP <= next_turn_ending])
    speaker_2_next_turn_duration = next_turn_ending - next_ipu_t0

    return speaker_2_next_turn_total_speech_time, speaker_2_next_turn_ipus_count, speaker_2_next_turn_duration


if __name__ == "__main__":
    kind = "overlap"
    main(
        ipus_table="csvs/ipus_uba.csv",
        kind=kind,
        tt_table="csvs/tt-table.csv",
        config="configs/experiments.ini",
        corpus="uba",
        tasks_table="csvs/tasks_uba.csv",
        tasks_features_list="lists/tasks_features.lst",
        dev_output_folder=f"instances/{kind}/dev",
        held_out_output_folder=f"instances/{kind}/held_out",
        dev_output_list=f"lists/dev_instances_{kind}.lst",
        held_out_output_list=f"lists/held_out_instances_{kind}.lst",
        force=True
    )
