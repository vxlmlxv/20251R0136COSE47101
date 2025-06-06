import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import logging
import time
from datetime import datetime
import os

# 로그 설정
logging.basicConfig(filename='log_full_model.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start_all = time.time()

# 데이터 불러오기
parks = pd.read_csv("parks.csv")
cctv = pd.read_csv("cctv.csv")
reports = pd.read_csv("reports.csv")

# 민원 발생 지점 샘플링
n_samples = 10000
report_sample = reports.sample(n=n_samples, random_state=42)

# 민원 없는 지점 무작위 생성 (서울 위경도 대략 범위 사용)
np.random.seed(42)
lat_rand = np.random.uniform(37.45, 37.70, n_samples)
lon_rand = np.random.uniform(126.80, 127.10, n_samples)
fake_reports = pd.DataFrame({'위도': lat_rand, '경도': lon_rand, '요일': 'Weekday'})

# 거리 함수
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 공통 처리 함수
def extract_features(row, label):
    try:
        r_lat, r_lon = row['위도'], row['경도']
        d_parks = haversine_np(r_lon, r_lat, parks['경도'].values, parks['위도'].values)
        d_cctvs = haversine_np(r_lon, r_lat, cctv['경도'].values, cctv['위도'].values)
        nearby_parks = parks[d_parks <= 300]

        # 요일 = Weekday 가정
        def get_fee_hours(park_row):
            if park_row.get('평일유료') == 'Y':
                return (
                    park_row.get('1시간 요금', 0),
                    park_row.get('평일운영시간', 0)
                )
            return (0, 0)

        fees_hours = nearby_parks.apply(get_fee_hours, axis=1)
        total_spaces = nearby_parks['총주차면'].sum()
        avg_fee = np.mean([fh[0] for fh in fees_hours]) if len(fees_hours) > 0 else 0
        avg_hours = np.mean([fh[1] for fh in fees_hours]) if len(fees_hours) > 0 else 0
        cctv_count = np.sum(d_cctvs <= 300)

        return {
            '총주차면수': total_spaces,
            '평균요금': avg_fee,
            '평균운영시간': avg_hours,
            'CCTV개수': cctv_count,
            '민원발생': label
        }
    except Exception as e:
        logging.error(f"⚠️ 좌표 처리 오류: {e}")
        return None

# 분석
results = []
for _, r in tqdm(report_sample.iterrows(), total=n_samples, desc="✅ 민원발생 1 처리"):
    res = extract_features(r, 1)
    if res: results.append(res)

for _, r in tqdm(fake_reports.iterrows(), total=n_samples, desc="✅ 민원발생 0 처리"):
    res = extract_features(r, 0)
    if res: results.append(res)

# 데이터프레임 변환 및 회귀 분석
df = pd.DataFrame(results)
X = df[['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']]
X = sm.add_constant(X)
y = df['민원발생']

logit_model = sm.Logit(y, X)
result = logit_model.fit()

# 결과 출력
print(result.summary())

# 종료
end_all = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"✅ 전체 소요 시간: {end_all - start_all:.2f}초")
