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

# file_path = "지오코딩_결과.csv"
# data = pd.read_csv(file_path, encoding='utf-8-sig')

# # Find rows with missing values
# missing_rows = data[data.isnull().any(axis=1)]

# # Count the number of rows with missing values
# missing_count = len(missing_rows)

# print(f"Number of rows with missing values: {missing_count}")

file_path = "parks.csv"
data = pd.read_csv(file_path, encoding='utf-8-sig')

print(data.shape)
print(data.columns)
# keep only the unique values in ['경도', '위도']
# extract all rows that have at least one duplicate in ['경도','위도']
dups = data[data.duplicated(subset=['경도', '위도'], keep=False)]
# remove duplicate entries for mapping
data = data.drop_duplicates(subset=['경도', '위도'])
print(data.shape)
print(dups.head())

data_map = data[['위도', '경도', '총주차면']]
# rename columns
data_map = data_map.rename(columns={'위도': 'latitude', '경도': 'longitude', '총주차면': 'area'})
print(data_map.head())
data_map.to_csv('data_map.csv', index=False, encoding='utf-8-sig')