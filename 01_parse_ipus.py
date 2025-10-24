import pandas as pd
import numpy as np
import glob
import os
import helper  # Asumiendo que helper.py está en el mismo directorio
import configparser
from tqdm import tqdm
import ml.utils

def main(output_fname, config, corpus, tasks_table, subjects_info_csv, force):
    if not force and helper.exists(output_fname):
        helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
        return

    helper.mkdir_p(os.path.dirname(output_fname))
    tasks = pd.read_csv(tasks_table)

    configuration = configparser.ConfigParser()
    configuration.read(config)
   

    corpus_config = configuration["corpus-{}".format(corpus)]
    speakers_info = pd.read_csv(subjects_info_csv, sep=";", index_col="sessionID")
    wavs_fname = corpus_config["wavs"]

    phrases_files = sorted(list(glob.glob("{}/{}".format(corpus_config["folder"], corpus_config["phrases_files_regex"]))))
    phonemes_dictionary = helper.read_list("{}/{}".format(corpus_config["folder"], corpus_config["phonetic_dict"]))

    words_phones_count = {}
    for row in phonemes_dictionary:
        word = row[1]
        phones = row[2:]
        words_phones_count[word] = len(phones)

    silence_tag = "#"
    ipus = []
    for filename in tqdm(phrases_files):
        file_id = ".".join(os.path.basename(filename).split(".")[:-1])
        file_id = file_id.replace("channel1", "A").replace("channel2", "B")
        assert("A" in file_id or "B" in file_id)

        channel = "A" if "A" in file_id else "B"
        session = file_id.split(".")[0][1:]
        session_tasks = tasks[tasks.session == int(session)]

        speaker = speakers_info.loc[int(session), "idSpeaker{}".format(channel)]
        speaker_gender = speakers_info.loc[int(session), "genderSpeaker{}".format(channel)]
        phrases = [x for x in ml.utils.read_wavesurfer(filename) if x[2] != silence_tag]
        for (ipu_t0, ipu_tf, phrase) in phrases:
            dur = ipu_tf - ipu_t0
            phones_in_phrase = 0
            words_in_phrase = 0
            words = phrase.split(" ")
            out_of_dict_words = [w for w in words if w not in words_phones_count]

            if dur == 0:
                helper.warning("Ignoring empty IPU")
                continue

            if len(out_of_dict_words) > 0:
                helper.info("missing words in dict {}".format(out_of_dict_words))
                phones_in_phrase = np.nan
            else:

                for w in words:
                    phones_in_phrase += words_phones_count[w]
                    words_in_phrase += 1

                # if phones_in_phrase == 0:
                #     helper.warning("skipping zero words phrase: {}".format(phrase))
                #     continue
            words_by_sec = round(words_in_phrase / dur, 4)
            phones_by_sec = round(phones_in_phrase / dur, 4)
            wav_file = wavs_fname.format(CORPUS_FOLDER=corpus_config["folder"], SESSION=session, FILE_ID=file_id)
            
            if corpus != "eeg":  # 1 file per task
                task_found = session_tasks.loc[(session_tasks.t0 <= ipu_t0) & (ipu_t0 < session_tasks.tf)]
                if len(task_found) == 0:
                    helper.warning("Ignoring IPU outside of task")
                    continue
            else:
                task_id = int(file_id.split(".")[2])
                task_found = session_tasks.loc[(session_tasks.task_number == task_id)]

            assert len(task_found) == 1
            task = task_found.iloc[0]

            row = dict(token_id="{}.task.{}.t0.{}.tf.{}".format(file_id, task.task_number, "%.4f" % ipu_t0, "%.4f" % ipu_tf),
                       ipu_start_time=round(ipu_t0, 4),
                       ipu_end_time=round(ipu_tf, 4),
                       task_start_time=task.t0,
                       task_end_time=task.tf,
                       task=task.task_number,
                       duration=round(dur, 4),
                       session_channel="{}_{}".format(session, channel),
                       session_channel_task="{}_{}_{}".format(session, channel, task.task_number),
                       session_task="{}_{}".format(session, task.task_number),
                       channel=channel,
                       task_describer=task.Describer,
                       task_target=task.Target,
                       task_score=task.Score,
                       corpus=corpus,
                       speaker=speaker,
                       speaker_gender=speaker_gender,
                       session_number=session,
                       words_count=words_in_phrase,
                       phones_count=phones_in_phrase,
                       words_by_second=round(words_by_sec, 4),
                       phones_by_second=round(phones_by_sec, 4),
                       wav_file = corpus_config["folder"] + "/b1-dialogue-wavs/" + file_id + ".wav"
)

            ipus.append(row)

    df = pd.DataFrame(ipus)
    df.to_csv(output_fname, index=False)


if __name__ == "__main__":
    main(output_fname="csvs/ipus_uba.csv", config="configs/experiments.ini", corpus="uba", tasks_table="csvs/tasks_uba.csv", subjects_info_csv='csvs/subjects_info_uba.csv', force=True)
