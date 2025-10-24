import configparser
import glob
import os
import pandas as pd
import helper
import ml.utils

def main(output_fname, corpus, tasks_table, ipus_table, config, force):

    configuration = configparser.ConfigParser()
    configuration.read(config)
    corpus_config = configuration["corpus-{}".format(corpus)]
    turns_files_regex = corpus_config["turns_files_regex"]

    if not force and helper.exists(output_fname):
        helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
        return

    helper.mkdir_p(os.path.dirname(output_fname))
    tasks = pd.read_csv(tasks_table)

    # folder in which the ".turns" files are located
    corpus_folder = corpus_config["folder"]
    turns_files = sorted(list(glob.glob(os.path.join(corpus_folder, turns_files_regex))))
    assert len(turns_files) != 0

    info = []
    ipus_table = pd.read_csv(ipus_table)

    for turns_file in turns_files:
        turns = ml.utils.read_turns(turns_file)
        session, _, object_id, channel, _ = turns_file.split("/")[-1].split(".")
        session = session[1:]  # elimina la "s" inicial
        tasks_session = tasks[tasks.session == int(session)]

        for (t0, tf, tt_label) in turns:
            if corpus == "eeg":
                task_id = object_id
            else:
                task = tasks_session[(tasks_session.t0 <= t0) & (tasks_session.tf >= t0)]
                if len(task) == 0:
                    helper.warning("ignoring out-of-task transition for s{} at {} ({})".format(session, t0, tt_label))
                    continue
                assert len(task) == 1
                task_id = task.iloc[0].task_number
            info.append(dict(tt_label=tt_label,
                             session_number=session,
                             speaker2=channel,
                             task=task_id,
                             ipu2_start_time=round(t0, 4),
                             ipu2_end_time=round(tf, 4),
                             overlapped_transition=tt_label in configuration["global"]["overlapped_transition"].split(" ")
                             ))
    df = pd.DataFrame(info)
    df.to_csv(output_fname, index=None)
    helper.info("File saved at {}".format(output_fname))
    helper.info(df.tt_label.value_counts())


if __name__ == "__main__":
    main(output_fname="csvs/tt-table.csv", 
         corpus="uba",
         tasks_table="csvs/tasks_uba.csv",
         ipus_table="csvs/ipus_uba.csv",
         config="configs/experiments.ini", 
         force=True)
