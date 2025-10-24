import pandas as pd
import helper
from tqdm import tqdm
import os.path

def main(z_scored_tracks_list, tasks_table, output_list, output_folder, force):
    helper.mkdir_p(output_folder)
    helper.mkdir_p(os.path.dirname(output_list))

    tasks = pd.read_csv(tasks_table)
    # tasks = pd.read_csv(tasks_table)
    tokens_list = []

    for sess_channel, track_fname in tqdm(helper.read_list(z_scored_tracks_list)):
        sess, channel = sess_channel.split("_")

        tasks_for_session = tasks[tasks.session == int(sess)]
        if channel == "A":
            features_A = pd.read_csv(track_fname, index_col="time")
            features_B = pd.read_csv(track_fname.replace("_A", "_B"), index_col="time")

            for _, task_info in tasks_for_session.iterrows():
                task_id = task_info.task_number

                sess_task = "{}_{}".format(sess, task_id)
                output_fname = "{}/{}_{}.csv".format(output_folder, sess, task_id)

                if not force and helper.exists(output_fname):
                    helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
                else:
                    features_task_A = features_A.loc[features_A.task == task_id, :].drop("task", axis=1)
                    features_task_B = features_B.loc[features_B.task == task_id, :].drop("task", axis=1)
                    joined = features_task_A.join(features_task_B, lsuffix="_A", rsuffix="_B")
                    joined.to_csv(output_fname)

                tokens_list.append((sess_task, output_fname))

    helper.save_list(tokens_list, output_list)


if __name__ == "__main__":
    main(z_scored_tracks_list="lists/z_scored_tracks.lst", tasks_table="csvs/tasks_uba.csv", output_list="lists/tasks_features.lst", output_folder="tasks_features/", force=True)
