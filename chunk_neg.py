import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.neighbors import BallTree

# 🌐 Haversine 거리 기반 BallTree
EARTH_RADIUS = 6371000  # meters

def build_tree(df):
    coords_rad = np.radians(df[['위도', '경도']].values)
    return BallTree(coords_rad, metric='haversine')

def query_radius(tree, point, radius_m):
    point_rad = np.radians([point])
    ind = tree.query_radius(point_rad, r=radius_m / EARTH_RADIUS)
    return ind[0]

# 📂 데이터 로딩
parks = pd.read_csv("src/parks_mod.csv")
cctv = pd.read_csv("cctv.csv")

# 📌 결측치 처리
for col in ['평일시작', '평일종료', '토요일시작', '토요일종료', '공휴일시작', '공휴일종료']:
    parks[col] = parks[col].fillna(0)

# 📁 출력 폴더 생성
os.makedirs("result_chunks_label0_fast", exist_ok=True)

# 🧭 좌표 검색용 BallTree 구성
park_tree = build_tree(parks)
cctv_tree = build_tree(cctv)

# 🔧 설정값
chunk_size = 5000
num_samples = 300000
n_chunks = (num_samples // chunk_size) + 1
np.random.seed(42)

print("🚀 랜덤 label=0 샘플 생성 시작")

for i in range(n_chunks):
    results = []
    for _ in tqdm(range(chunk_size), desc=f"Chunk {i+1} 처리 중"):
        for _ in range(10):  # 최대 10회 재시도
            park = parks.sample(1).iloc[0]
            lat = park['위도'] + np.random.uniform(-0.0015, 0.0015)
            lon = park['경도'] + np.random.uniform(-0.0015, 0.0015)
            hour = np.random.randint(7, 23)
            day = np.random.choice(['Weekday', 'Saturday', 'Holiday'])

            idx = query_radius(park_tree, [lat, lon], 500)
            nearby_parks = parks.iloc[idx].copy()

            if day == 'Weekday':
                nearby_parks = nearby_parks[
                    (nearby_parks['평일시작'] <= hour) & (nearby_parks['평일종료'] > hour)
                ]
                hours = nearby_parks['평일운영시간']
            elif day == 'Saturday':
                nearby_parks = nearby_parks[
                    (nearby_parks['토요일시작'] <= hour) & (nearby_parks['토요일종료'] > hour)
                ]
                hours = nearby_parks['토요일운영시간']
            else:
                nearby_parks = nearby_parks[
                    (nearby_parks['공휴일시작'] <= hour) & (nearby_parks['공휴일종료'] > hour)
                ]
                hours = nearby_parks['공휴일운영시간']

            if len(nearby_parks) == 0:
                continue

            idx_cctv = query_radius(cctv_tree, [lat, lon], 500)
            cctv_count = len(idx_cctv)

            total_spaces = nearby_parks['총주차면'].sum()
            avg_fee = nearby_parks['1시간 요금'].replace(-4, np.nan).mean()
            avg_hours = hours.replace(-4, np.nan).mean()

            results.append({
                "총주차면수": total_spaces,
                "평균요금": avg_fee,
                "평균운영시간": avg_hours,
                "CCTV개수": cctv_count,
                "위도": lat,
                "경도": lon,
                "민원발생": 0
            })
            break  # 조건 만족 시 break

    pd.DataFrame(results).to_csv(f"result_chunks_label0_fast/results_part_{i}.csv", index=False)

print("✅ 모든 랜덤 샘플 생성 완료")
