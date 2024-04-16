from log_power_consumption import log_power_consumption
import time
from live_plot import live_plot
from multiprocessing import Process
import argparse
import sys


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser("Log power consumption data")
    parser.add_argument("--plot_data",
                        type=int,
                        help="Flag to determine if data should be plotted. 1 for yes, 0 for no. Default is 0.",
                        default=0)
    parser.add_argument("--local",
                        dest="local",
                        type=int,
                        help="Flag to determine if data should be logged from local device. "
                             "1 for yes, 0 for no. Default is 1.",
                        default=1)
    args = parser.parse_args()

    # Capture start time
    start_timestamp = time.time()

    sys.stdout = open(f"{start_timestamp}_output.txt", "w")
    sys.stderr = open(f"{start_timestamp}_error.txt", "w")

    print('experiments started at:', start_timestamp)

    # Start logging data
    process_data_log = Process(target=log_power_consumption, args=(start_timestamp, args.local))
    process_data_log.start()

    # Start live plotting if flag is set
    if args.plot_data == 1:
        process_live_plot = Process(target=live_plot, args=(start_timestamp,))
        process_live_plot.start()

