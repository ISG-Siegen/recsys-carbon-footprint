import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pandas as pd
import time
from pathlib import Path


def live_plot(start_timestamp):
    """
    Live plot of power consumption data
    :param start_timestamp: Start time of the experiment
    """
    # Style setting
    style.use('fivethirtyeight')

    # create figure and subplots
    fig = plt.figure()
    ax_current_power = fig.add_subplot(2, 1, 1)
    ax_total_power = fig.add_subplot(2, 1, 2)

    # Animation function
    def animate(i, start_timestamp):
        if start_timestamp + 3600 <= time.time():
            start_timestamp = start_timestamp + 3600
        # Read data from file
        graph_data = pd.read_csv(f"log_{start_timestamp}.csv")
        if Path(f"log_{start_timestamp - 3600}.csv").exists():
            old_graph_data = pd.read_csv(f"log_{start_timestamp - 3600}.csv")
            graph_data = pd.concat([old_graph_data, graph_data])
        # Limit data to last 120 datapoints
        current_graph_data = graph_data.tail(120)
        total_graph_data = graph_data.tail(3600)
        # Create lists for x and y values
        x_current_list = []
        y_current_list = []
        x_total_list = []
        y_total_list = []
        # Iterate over rows and append to lists
        for row in current_graph_data.iterrows():
            # gather current power data
            x_current = row[1]['unix timestamp']
            y_current = row[1]['apower (current energy draw in W)']
            x_current_list.append(float(x_current))
            y_current_list.append(float(y_current))

        for row in total_graph_data.iterrows():
            # gather total power data
            x_total = row[1]['unix timestamp']
            y_total = row[1]['aenergy_total (total energy consumed, in Wh)']
            x_total_list.append(float(x_total))
            y_total_list.append(float(y_total))

        # Set labels
        ax_total_power.set_xlabel('Time')
        ax_current_power.set_ylabel('Power Consumption in W')

        # Clear the axis and plot the new data
        ax_current_power.clear()
        ax_current_power.plot(x_current_list, y_current_list)

        # Clear the axis and plot the new data
        ax_total_power.clear()
        ax_total_power.plot(x_total_list, y_total_list)

    # Create animation
    ani = animation.FuncAnimation(fig, animate, interval=1500, fargs=(start_timestamp,))
    # Show plot
    plt.tight_layout()
    plt.show()
    # close plot
    plt.close()
