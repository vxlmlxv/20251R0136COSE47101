import requests
import pandas as pd
import json

# apikey =  "D9C1CFC9-0FD3-3304-A3E7-131C33A91190"
# apiurl = "http://api.vworld.kr/req/address?"
# params = {
#     "service": "address",
#     "request": "getcoord",
#     "crs": "epsg:4326",
#     "address": "강서구 화곡동 980-19", #변수 이용할 경우 따옴표 없이 변수명만 
#     "format": "json",
#     "type": "parcel",
#     "key": apikey
# }
# response = requests.get(apiurl, params=params)
# if response.status_code == 200:
#     json_data = response.json()
#     print(json_data)

file_path = "지오코딩_결과.csv"
data = pd.read_csv(file_path, encoding='utf-8-sig')

# Find rows with missing values
missing_rows = data[data.isnull().any(axis=1)]

# Count the number of rows with missing values
missing_count = len(missing_rows)

print(f"Number of rows with missing values: {missing_count}")
