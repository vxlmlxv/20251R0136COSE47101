import pandas as pd
import numpy as np

# 필요한 파일 불러오기
df_chunk = pd.read_csv("result_chunks_2/results_part_0.csv")  # ⬅️ 분석 대상 청크
parks = pd.read_csv("parks.csv")

# Haversine 거리 계산 함수
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 운영시간 평균이 1.0인 민원 추출
df_target = df_chunk[df_chunk['평균운영시간'] == 1.0]

print(f"🎯 평균운영시간 1.0인 민원 수: {len(df_target)}")

for idx, row in df_target.iterrows():
    lat, lon = row['위도'], row['경도']
    distances = haversine_np(lon, lat, parks['경도'].values, parks['위도'].values)
    nearby_parks = parks[distances <= 500].copy()

    print(f"\n📍 민원 위치 (위도: {lat}, 경도: {lon})")
    print(f"반경 500m 이내 주차장 개수: {len(nearby_parks)}")

    if not nearby_parks.empty:
        display_cols = ['총주차면', '평일운영시간', '토요일운영시간', '공휴일운영시간',
                        '평일유료', '토요일유료', '공휴일유료', '1시간 요금']
        print(nearby_parks[display_cols])
