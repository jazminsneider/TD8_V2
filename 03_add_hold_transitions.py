import pandas as pd
from tqdm import tqdm
import helper

def main(output_fname, ipus_table, tt_table, force):
    if not force and helper.exists(output_fname):
        helper.warning("{} already exist, skipping (use --force to overwrite)".format(output_fname))
        return

    ipus = pd.read_csv(ipus_table, index_col="token_id")
    tt = pd.read_csv(tt_table)
    # tt_holds = tt[tt.tt_label == "H"]
    tt = tt[tt.tt_label != "H"]

    # tt.index = tt.apply(
    #     lambda x: "s{}.objects.{}.t0.{}.tf.{}".format(
    #         "%02d" % x.session_number,
    #         x.task,
    #         "%.4f" % x.ipu2_start_time,
    #         "%.4f" % x.ipu2_end_time
    #         ),
    #     axis=1
    # )

    # tt = tt.drop(["task", "session_number", "ipu2_start_time", "ipu2_end_time", "speaker2"], axis=1)
    # ipus.merge(tt, left_on=["session_number", "task", "ipu_start_time", "ipu_end_time"], right_on=["session_number", "task", "ipu2_start_time", "ipu2_end_time"])
    # import ipdb; ipdb.set_trace()
    # ipus = ipus.join(tt)
    # import ipdb; ipdb.set_trace()
    
    for name, group in tqdm(ipus.groupby(["session_number", "task"])):
        res = []
        for channel in ["A", "B"]:
            in_task_ipus_interlocutor = group[group.channel != channel].sort_values(by="ipu_start_time").copy()
            in_task_ipus_locutor = group[group.channel == channel].sort_values(by="ipu_start_time").copy()

            for i in range(0, len(in_task_ipus_locutor) - 1):
                ipu1 = in_task_ipus_locutor.iloc[i]
                ipu2 = in_task_ipus_locutor.iloc[i + 1]
                interruptions = in_task_ipus_interlocutor[~((in_task_ipus_interlocutor.ipu_end_time <= ipu1.ipu_end_time) | (in_task_ipus_interlocutor.ipu_start_time >= ipu2.ipu_start_time))]
                if len(interruptions) == 0:
                    res.append(dict(tt_label="H", session_number=ipu2.session_number, speaker2=ipu2.channel, task=ipu2.task, ipu2_start_time=ipu2.ipu_start_time, ipu2_end_time=ipu2.ipu_end_time, overlapped_transition=False))
            # assert ipu.channel in ["A", "B"]
            # if ipu.channel == "A":
            #     mask_A[int((ipu.ipu_start_time - minn) * 100):int((ipu.ipu_end_time - minn) * 100)] = True
            # else:
            #     mask_B[int((ipu.ipu_start_time - minn) * 100):int((ipu.ipu_end_time - minn) * 100)] = True

        tt = pd.concat([tt, pd.DataFrame(res)], ignore_index=True)

    tt.to_csv(output_fname, index=False)
    # tt_new_holds = tt[tt.tt_label == "H"]
    # diff1 = pd.DataFrame([row for row in tt_new_holds.values if not np.equal(row, tt_holds).all(axis=1).any()], columns=tt.columns)
    # diff2 = pd.DataFrame([row for row in tt_holds.values if not np.equal(row, tt_new_holds).all(axis=1).any()], columns=tt.columns)
    #
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main(output_fname='csvs/holds-tt-table.csv', ipus_table='csvs/ipus_uba.csv', tt_table='csvs/tt-table.csv', force=True)
