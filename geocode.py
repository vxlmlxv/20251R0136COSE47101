import requests
import pandas as pd
import json

# Load the CSV file
file_path = "cleaned_seoul_parking_info.csv"
data = pd.read_csv(file_path, encoding='utf-8')

# Function to get coordinates for a given address
def get_coordinates(address):
    apiurl = "https://api.vworld.kr/req/address?"
    apikey =  "D9C1CFC9-0FD3-3304-A3E7-131C33A91190"

    params = {
        "service": "address",
        "request": "getcoord",
        "crs": "epsg:4326",
        "address": address,
        "format": "json",
        "type": "parcel", # 도로명주소는 road, 지번주소는 parcel
        "key": apikey
    }
    response = requests.get(apiurl, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['response']['status'] == 'OK':
            point_info = data['response']['result']['point']
            return point_info['x'], point_info['y']
    return None, None

data[['경도', '위도']] = data['주소'].apply(lambda addr: pd.Series(get_coordinates(addr)))

# Display the updated dataframe
data.head()

# Save the updated dataframe to a new CSV file
data.to_csv('지오코딩_결과.csv', index=False, encoding='utf-8-sig')