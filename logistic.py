import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import logging
import time
from datetime import datetime

# 로그 설정
logging.basicConfig(
    filename='log_analysis.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 시간 측정 시작
print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start = time.time()
logging.info("🚀 분석 시작")

# Haversine 거리 계산 (벡터화)
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000  # 지구 반지름 (m)
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 데이터 불러오기
parks = pd.read_csv("parks.csv")
reports = pd.read_csv("reports.csv")
cctv = pd.read_csv("cctv.csv")

# 샘플링 (민원 10,000개만)
max_reports = 10000
report_sample = reports.sample(n=min(max_reports, len(reports)), random_state=42)
logging.info("✅ 전체 샘플링 수: %d", len(report_sample))
print(f"✅ 샘플링된 민원 수: {len(report_sample)}")

# 결과 저장
results = []

# 분석 반복
for idx, (_, report) in enumerate(tqdm(report_sample.iterrows(), total=len(report_sample), desc="Haversine 분석 중")):
    if idx % 1000 == 0 and idx > 0:
        logging.info("⏳ 현재 %d건 처리 중...", idx)

    try:
        r_lat, r_lon = report['위도'], report['경도']

        # 거리 계산
        d_parks = haversine_np(r_lon, r_lat, parks['경도'].values, parks['위도'].values)
        d_cctvs = haversine_np(r_lon, r_lat, cctv['경도'].values, cctv['위도'].values)

        # 300m 이내 주차장
        nearby_parks = parks[d_parks <= 300]

        # 요일별 요금/운영시간
        def get_fee_hours(park_row):
            day = report.get('요일')  # 'Weekday', 'Saturday', 'Holiday'
            try:
                if day == 'Weekday':
                    return (
                        park_row.get('1시간 요금', 0) if park_row.get('평일유료') == 'Y' else 0,
                        park_row.get('평일운영시간', 0) if park_row.get('평일유료') == 'Y' else 0
                    )
                elif day == 'Saturday':
                    return (
                        park_row.get('1시간 요금', 0) if park_row.get('토요일유료') == 'Y' else 0,
                        park_row.get('토요일운영시간', 0) if park_row.get('토요일유료') == 'Y' else 0
                    )
                elif day == 'Holiday':
                    return (
                        park_row.get('1시간 요금', 0) if park_row.get('공휴일유료') == 'Y' else 0,
                        park_row.get('공휴일운영시간', 0) if park_row.get('공휴일유료') == 'Y' else 0
                    )
            except:
                return (0, 0)
            return (0, 0)

        fees_hours = nearby_parks.apply(get_fee_hours, axis=1)
        total_spaces = nearby_parks['총주차면'].sum()
        avg_fee = np.mean([fh[0] for fh in fees_hours]) if len(fees_hours) > 0 else 0
        avg_hours = np.mean([fh[1] for fh in fees_hours]) if len(fees_hours) > 0 else 0

        # 300m 이내 CCTV
        cctv_count = np.sum(d_cctvs <= 300)

        results.append({
            '총주차면수': total_spaces,
            '평균요금': avg_fee,
            '평균운영시간': avg_hours,
            'CCTV개수': cctv_count,
            '민원발생': 1
        })

    except Exception as e:
        logging.error(f"[index={report.name}] 예외 발생: {repr(e)}")
        logging.error(f"해당 report 내용:\n{report.to_dict()}")

# 시간 측정 종료
end = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"✅ 총 소요 시간: {end - start:.2f}초")
logging.info("✅ 전체 처리 완료. 총 소요 시간: %.2f초", end - start)

# 회귀 분석
df = pd.DataFrame(results)
X = df[['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']]
X = sm.add_constant(X)
y = df['민원발생']

logging.info("📊 로지스틱 회귀 분석 시작 (총 %d건)", len(df))
logit_model = sm.Logit(y, X)
result = logit_model.fit()
logging.info("✅ 회귀 분석 완료")
print(result.summary())
