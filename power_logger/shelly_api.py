import requests
from static import *


def check_api_response_code(response):
    if response.status_code != 200:
        raise Exception(f"API call failed. Status code: {response.status_code} \n Response: {response}")


class Shelly_API():
    def __init__(self,
                 api_key=API_KEY,
                 api_control_url=API_CONTROL_URL,
                 local_flag=True):
        self.api_key = api_key
        self.api_control_url = api_control_url
        self.local = local_flag
        if self.local:
            self.api_status_url = API_LOCAL_URL
            self.device_id = LOCAL_DEVICE_ID
        else:
            self.api_status_url = API_CLOUD_URL
            self.device_id = CLOUD_DEVICE_ID

    def get_data(self):
        """
        Get data from the API
        :return: JSON response
        """
        if self.local:
            headers = {'Content-Type': 'application/x-www-form-urlencoded', }
            request_data = ('{"id":1, '
                            f'"src":"{self.device_id}", '
                            '"method":"Switch.GetStatus", '
                            '"params":{"id":0}}')
            response = requests.post(API_LOCAL_URL, headers=headers, data=request_data)
        else:
            request_data = {
                'id': self.device_id,
                'auth_key': self.api_key
            }
            response = requests.post(url=self.api_status_url, data=request_data)
            response_json = response.json()
            if not response_json['data']['device_status']['switch:0']['output']:
                raise Exception('Device is off, turn it on')

        check_api_response_code(response)
        return response.json()

    def turn_on(self):
        """
        Turn on the device
        """
        request_data = {
            'id': self.device_id,
            'auth_key': self.api_key,
            'channel': 0,
            'turn': 'on'
        }
        response = requests.post(url=self.api_control_url, data=request_data)
        check_api_response_code(response)

    def turn_off(self):
        """
        Turn off the device
        """
        request_data = {
            'id': self.device_id,
            'auth_key': self.api_key,
            'channel': 0,
            'turn': 'off'
        }
        response = requests.post(url=self.api_control_url, data=request_data)
        check_api_response_code(response)
