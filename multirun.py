##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##

#
# Compare running times of sorting and merging
# When the number of tables surpasses a threshold, sorting is faster than merging
# we would like to determine this threshold point with various number of columns in the tables
#

import pandas as pd
import subprocess
import os

# number of times a task will be executed and its running time is averaged
SINGLE_TASK_RUN_COUNT = 3
num_columns = [1, 2, 4, 8]

data_size = 1

single_run_result_file = "single_run.csv"
results_txt_file = "results.txt"
results_csv_file = "results.csv"


def run_task_multiple_times(args):
    durations = []

    for i in range(SINGLE_TASK_RUN_COUNT):
        print("/////////////////////////////////////////////////////////////////")
        print(f"running sequence: {i}")
        subprocess.run(args)
        with open(single_run_result_file) as f:
            duration = f.readline()
            durations.append(int(duration))
        os.remove(single_run_result_file)

    # write the results to a text file
    with open(results_txt_file, "a") as file_object:
        file_object.write(", ".join(args) + ": " + ", ".join(map(str, durations)) + "\n")

    # todo: check whether there is any anomaly in the results
    #       some values can be too high, low
    return sum(durations) / len(durations)


def run_task_for_ncolumns(exe_file, task_type, num_column, num_tables):
    delays = []
    for num_table in num_tables:
        print("##################################################################################################")
        print(f"running: {exe_file} with {task_type}, {data_size}GB, {num_column}, {num_table}")
        delay = run_task_multiple_times([exe_file, task_type, str(data_size) + "GB", str(num_column), str(num_table)])
        delays.append(delay)
    return delays


def run_task_for_type(exe_file, task_type, num_tables):
    delay_df = pd.DataFrame(index=num_tables)
    delay_df.index.name='num tables'
    for num_column in num_columns:
        delays = run_task_for_ncolumns(exe_file, task_type, num_column, num_tables)
        delay_df[str(num_column)+"-column"] = delays

    return delay_df


#####################################################

exefile = "build/bin/sorting"
task_types = ["sort", "merge"]
num_tables = [1, 2, 4, 8, 16, 32, 64, 128]

# delete previous sing-run-result file if exist
if os.path.exists(single_run_result_file):
    os.remove(single_run_result_file)

for tsk_type in task_types:
    # remove results files if exist
    csv_file = tsk_type + "-" + results_csv_file
    if os.path.exists(csv_file):
        os.remove(csv_file)
    txt_file = tsk_type + "-" + results_txt_file
    if os.path.exists(txt_file):
        os.remove(txt_file)

    print("=======================================================================================================")
    print(f"starting the tests for the task: {tsk_type}")
    df = run_task_for_type(exe_file=exefile, task_type=tsk_type, num_tables=num_tables)
    df.to_csv(csv_file, float_format='%.1f')
    os.rename(results_txt_file, tsk_type + "-" + results_txt_file)

