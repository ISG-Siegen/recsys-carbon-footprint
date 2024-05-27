import pandas as pd
from pathlib import Path
import json
import os

log_file_directory = 'experiment_logs/energy_logs'
experiment_log_file_directory = 'experiment_logs/energy'

with open("experiments_2023_workstation.json", "r") as file:
    experiment = json.load(file)

num_folds = 5
dataset = experiment["data_set_names"]
recommenders = experiment["algorithm_names"]
stages = ['evaluate', 'fit', 'predict']
metrics = ['NDCG@1', 'NDCG@3', 'NDCG@5', 'NDCG@10', 'NDCG@20']

combined_log_file = pd.DataFrame()

for filename in os.listdir(log_file_directory):
    if filename.startswith("log_") and filename.endswith(".csv"):
        f = os.path.join(log_file_directory, filename)
        combined_log_file = pd.concat([combined_log_file, pd.read_csv(f)])

mapped_csv = pd.DataFrame(columns=['dataset',
                                   'recommender',
                                   'fold',
                                   'stage',
                                   'start_time',
                                   'end_time',
                                   'duration',
                                   'total_energy_consumed (total energy consumed, in Wh)',
                                   'mean_power (mean energy draw in W)',
                                   'min_power (min energy draw in W)',
                                   'max_power (max energy draw in W)',
                                   'mean_voltage (mean voltage in V)',
                                   'min_voltage (min voltage in V)',
                                   'max_voltage (max voltage in V)',
                                   'mean_current (mean current in A)',
                                   'min_current (min current in A)',
                                   'max_current (max current in A)',
                                   'mean_temperature_tC (mean temperature in C)',
                                   'min_temperature_tC (min temperature in C)',
                                   'max_temperature_tC (max temperature in C)',
                                   'NDCG@1',
                                   'NDCG@3',
                                   'NDCG@5',
                                   'NDCG@10',
                                   'NDCG@20'])

for data_set_name in dataset:
    for num_fold in range(num_folds):
        for recommender in recommenders:
            for stage in stages:
                evaluate_log = json.load(
                    open(Path(f"experiment_logs/energy/{data_set_name}_{recommender}_0_{num_fold}_{stage}.json"), "r"))

                performance_log = json.load(
                    open(Path(
                        f"experiment_logs/checkpoints/{data_set_name}/checkpoint_{recommender}/config_0/fold_{num_fold}/evaluate_log.json"),
                        "r"))

                start_time = evaluate_log['start']
                end_time = evaluate_log['end']

                relevant_datapoints = combined_log_file[
                    combined_log_file['unix timestamp'].between(start_time, end_time)]

                duration = end_time - start_time
                total_energy_consumed = relevant_datapoints['aenergy_total (total energy consumed, in Wh)'].max() - \
                                        relevant_datapoints['aenergy_total (total energy consumed, in Wh)'].min()

                ndcg_1 = performance_log['NDCG@1']
                ndcg_3 = performance_log['NDCG@3']
                ndcg_5 = performance_log['NDCG@5']
                ndcg_10 = performance_log['NDCG@10']
                ndcg_20 = performance_log['NDCG@20']

                experiment_df = pd.DataFrame({'dataset': data_set_name,
                                              'recommender': recommender,
                                              'fold': num_fold,
                                              'stage': stage,
                                              'start_time': start_time,
                                              'end_time': end_time,
                                              'duration': duration,
                                              'total_energy_consumed (total energy consumed, in Wh)': total_energy_consumed,
                                              'mean_power (mean energy draw in W)': relevant_datapoints[
                                                  'apower (current energy draw in W)'].mean(),
                                              'min_power (min energy draw in W)': relevant_datapoints[
                                                  'apower (current energy draw in W)'].min(),
                                              'max_power (max energy draw in W)': relevant_datapoints[
                                                  'apower (current energy draw in W)'].max(),
                                              'mean_voltage (mean voltage in V)': relevant_datapoints[
                                                  'voltage (current voltage in V)'].mean(),
                                              'min_voltage (min voltage in V)': relevant_datapoints[
                                                  'voltage (current voltage in V)'].min(),
                                              'max_voltage (max voltage in V)': relevant_datapoints[
                                                  'voltage (current voltage in V)'].max(),
                                              'mean_current (mean current in A)': relevant_datapoints[
                                                  'current (current current in A)'].mean(),
                                              'min_current (min current in A)': relevant_datapoints[
                                                  'current (current current in A)'].min(),
                                              'max_current (max current in A)': relevant_datapoints[
                                                  'current (current current in A)'].max(),
                                              'mean_temperature_tC (mean temperature in C)': relevant_datapoints[
                                                  'temperature_tC (current temperature in C)'].mean(),
                                              'min_temperature_tC (min temperature in C)': relevant_datapoints[
                                                  'temperature_tC (current temperature in C)'].min(),
                                              'max_temperature_tC (max temperature in C)': relevant_datapoints[
                                                  'temperature_tC (current temperature in C)'].max(),
                                              'NDCG@1': ndcg_1,
                                              'NDCG@3': ndcg_3,
                                              'NDCG@5': ndcg_5,
                                              'NDCG@10': ndcg_10,
                                              'NDCG@20': ndcg_20
                                              }, index=[0])

                mapped_csv = pd.concat([mapped_csv, experiment_df])
mapped_csv.to_csv(f"experiment_logs/mapped_logs.csv", index=False)
