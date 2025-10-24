import glob
import pandas as pd
import os
import helper  
import configparser



def main(config, corpus, output_fname, force):
    if not force and helper.exists(output_fname):
        helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
        return
    helper.mkdir_p(os.path.dirname(output_fname))

    configuration = configparser.ConfigParser()
    configuration.read(config)
    corpus_config = configuration["corpus-{}".format(corpus)]
    normalize_task_time = "one_wav_per_task" in corpus_config

    res = []
    for session in corpus_config["sessions"].split(" "):
        tasks_files = glob.glob("{}/{}".format(corpus_config["folder"], corpus_config["tasks_files"]).format(SESSION=session))
        i = 1
        for tasks_file in sorted(tasks_files): 
            tasks = helper.read_list(tasks_file)
            
            tasks_info = [line[0:3] for line in tasks if len(line) >= 3 and "Images" in line[2]]
            for task_t0, task_tf, task_data in tasks_info:
                if "Images" in task_data and "Describer" in task_data:
                    
                    if task_data.endswith(";"):
                        task_data = task_data[:-1]
                    task_dict = dict([y.split(":") for y in task_data.split(";")])
                    task_dict["t0"] = float(task_t0) if not normalize_task_time else 0
                    
                    task_dict["tf"] = float(task_tf) if not normalize_task_time else float(task_tf) - float(task_t0)
                    
                    task_dict["session"] = session
                    
                    task_dict["task_number"] = i

                    res.append(task_dict)
                    i += 1
                else:
                    help.info("Ignoring task line: {}".format(task_data))

    df = pd.DataFrame(res)
    df.to_csv(output_fname, index=False)


if __name__ == "__main__":
    main(config="configs/experiments.ini", corpus="uba", output_fname="csvs/tasks_uba.csv", force=True)
