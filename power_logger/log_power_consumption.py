from shelly_api import Shelly_API
from pathlib import Path
import csv
import time


def log_power_consumption(start_timestamp: float, local: bool = True):
    """
    Log power consumption data to a file
    :param start_timestamp: Name of the log file
    :param local: Flag to determine if the device is local or cloud based. Default is True.
    """
    # Create API object
    api = Shelly_API(local_flag=local)

    # Loop to get and log data
    while True:
        # Create log file if it does not exist
        if not Path(f"log_{start_timestamp}.csv").exists() or start_timestamp + 3600 <= time.time():
            if start_timestamp + 3600 <= time.time():
                start_timestamp = start_timestamp + 3600
            if local:
                with open(f"log_{start_timestamp}.csv", 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['unix timestamp',
                                     'aenergy_total (total energy consumed, in Wh)',
                                     'aenergy_minute_ts (total energy timestamp)',
                                     'apower (current energy draw in W)',
                                     'voltage (current voltage in V)',
                                     'current (current current in A)',
                                     'temperature_tC (current temperature in C)'])
            else:
                with open(f"log_{start_timestamp}.csv", 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['unix timestamp',
                                     '_updated (update time smart plug)',
                                     'aenergy_total (total energy consumed, in Wh)',
                                     'aenergy_minute_ts (total energy timestamp)',
                                     'apower (current energy draw in W)',
                                     'voltage (current voltage in V)',
                                     'current (current current in A)',
                                     'temperature_tC (current temperature in C)'])

        # Get data from API
        data = api.get_data()
        if local:
            electricity_info = data['result']
        else:
            data = data['data']
            # Extract relevant data
            device_data = data['device_status']
            updated_time = device_data['_updated']

            # Extract electricity info
            electricity_info = device_data['switch:0']

        # Extract total power consumption
        total_watt_hours = electricity_info['aenergy']['total']
        total_watt_hours_minute_ts = electricity_info['aenergy']['minute_ts']

        # Extract current power consumption
        current_enery_draw = electricity_info['apower']
        current_voltage = electricity_info['voltage']
        current_current = electricity_info['current']

        # Extract temperature
        current_temperature_C = electricity_info['temperature']['tC']

        # Append data to log file
        if local:
            with open(f"log_{start_timestamp}.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [time.time(), total_watt_hours, total_watt_hours_minute_ts, current_enery_draw,
                     current_voltage,
                     current_current, current_temperature_C])
                # cloud data pull frequency
                time.sleep(0.5)
        else:
            with open(f"log_{start_timestamp}.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [time.time(), updated_time, total_watt_hours, total_watt_hours_minute_ts, current_enery_draw,
                     current_voltage,
                     current_current, current_temperature_C])
            # cloud data pull frequency
            time.sleep(5)
