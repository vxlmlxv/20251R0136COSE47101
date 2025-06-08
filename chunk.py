import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm

# Haversine 거리 계산 함수
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000  # meters
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 데이터 불러오기
parks = pd.read_csv("src/parks_mod.csv")
cctv = pd.read_csv("cctv.csv")
reports = pd.read_csv("reports.csv")

# 저장 폴더 생성
os.makedirs("result_chunks_filtered", exist_ok=True)

# 청크 분할
chunk_size = 100000
n_chunks = (len(reports) // chunk_size) + 1

for i in range(27, n_chunks):
    chunk = reports.iloc[i*chunk_size : (i+1)*chunk_size]
    results = []

    for _, report in tqdm(chunk.iterrows(), total=len(chunk), desc=f"Chunk {i+1} 처리 중"):
        try:
            r_lat, r_lon = report["위도"], report["경도"]
            day = report["요일"]
            hour = int(report["민원접수시간"].split(":")[0])

            dist_parks = haversine_np(r_lon, r_lat, parks["경도"], parks["위도"])
            dist_cctv = haversine_np(r_lon, r_lat, cctv["경도"], cctv["위도"])
            parks_within = parks[dist_parks <= 500].copy()

            if day == 'Weekday':
                parks_within = parks_within[
                    (parks_within['평일시작'] <= hour) & (parks_within['평일종료'] > hour)
                ]
                hours = parks_within['평일운영시간']
            elif day == 'Saturday':
                parks_within = parks_within[
                    (parks_within['토요일시작'] <= hour) & (parks_within['토요일종료'] > hour)
                ]
                hours = parks_within['토요일운영시간']
            elif day == 'Holiday':
                parks_within = parks_within[
                    (parks_within['공휴일시작'] <= hour) & (parks_within['공휴일종료'] > hour)
                ]
                hours = parks_within['공휴일운영시간']
            else:
                continue

            if len(parks_within) == 0:
                continue

            total_spaces = parks_within['총주차면'].sum()
            avg_fee = parks_within['1시간 요금'].replace(-4, np.nan).mean()
            avg_hours = hours.replace(-4, np.nan).mean()
            cctv_count = np.sum(dist_cctv <= 500)

            results.append({
                "총주차면수": total_spaces,
                "평균요금": avg_fee,
                "평균운영시간": avg_hours,
                "CCTV개수": cctv_count,
                "위도": r_lat,
                "경도": r_lon,
                "민원발생": 1
            })

        except Exception as e:
            continue

    pd.DataFrame(results).to_csv(f"result_chunks_filtered/results_part_{i}.csv", index=False)
