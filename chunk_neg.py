import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import logging

# 설정
chunk_size = 100000
n_chunks = 30
output_dir = "random_chunks"
os.makedirs(output_dir, exist_ok=True)

# 로그 설정
logging.basicConfig(filename='log_random_chunk.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 데이터 로드
parks = pd.read_csv("parks.csv")
cctv = pd.read_csv("cctv.csv")

# 거리 계산 함수
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 생성 시작
np.random.seed(42)

for chunk_id in range(n_chunks):
    print(f"🚧 청크 {chunk_id} 처리 중...")
    start = time.time()

    # 무작위로 주차장 위치 선택
    sampled_parks = parks.sample(n=chunk_size, replace=True).reset_index(drop=True)
    lat_offset = np.random.uniform(-0.0015, 0.0015, chunk_size)
    lon_offset = np.random.uniform(-0.0015, 0.0015, chunk_size)

    points = pd.DataFrame({
        '위도': sampled_parks['위도'] + lat_offset,
        '경도': sampled_parks['경도'] + lon_offset,
        '요일': np.random.choice(['Weekday', 'Saturday', 'Holiday'], size=chunk_size)
    })

    results = []

    for i in tqdm(range(chunk_size), desc=f"청크 {chunk_id} 생성 중"):
        try:
            r_lat, r_lon = points.loc[i, '위도'], points.loc[i, '경도']
            d_parks = haversine_np(r_lon, r_lat, parks['경도'].values, parks['위도'].values)
            d_cctvs = haversine_np(r_lon, r_lat, cctv['경도'].values, cctv['위도'].values)
            nearby_parks = parks[d_parks <= 300]

            def get_fee_hours(row):
                if row['평일유료'] == 'Y':
                    return row['1시간 요금'], row['평일운영시간']
                return 0, 0

            fee_hours = nearby_parks.apply(get_fee_hours, axis=1, result_type='expand')
            avg_fee = fee_hours[0].mean() if not fee_hours.empty else 0
            avg_hours = fee_hours[1].mean() if not fee_hours.empty else 0
            total_spaces = nearby_parks['총주차면'].sum()
            cctv_count = np.sum(d_cctvs <= 300)

            results.append({
                '총주차면수': total_spaces,
                '평균요금': avg_fee,
                '평균운영시간': avg_hours,
                'CCTV개수': cctv_count,
                '민원발생': 1,
                '위도': r_lat,      # ✅ 추가
                '경도': r_lon       # ✅ 추가
            })

        except Exception as e:
            logging.error(f"[청크 {chunk_id} index {i}] 오류 발생: {e}")

    df_chunk = pd.DataFrame(results)
    df_chunk.to_csv(f"{output_dir}/negative_part_{chunk_id}.csv", index=False)
    print(f"✅ 청크 {chunk_id} 저장 완료 ({len(df_chunk)}개), 소요 시간: {time.time() - start:.2f}초")
