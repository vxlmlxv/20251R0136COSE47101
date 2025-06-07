import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import logging
import time
from datetime import datetime
import traceback

# 로그 설정
logging.basicConfig(filename='log_analysis.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Haversine 거리 계산
def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371000
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# 데이터 불러오기
parks = pd.read_csv("parks.csv")
cctv = pd.read_csv("cctv.csv")
reports = pd.read_csv("reports.csv")

# 시간 시작
print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start_all = time.time()
logging.info("🚀 전체 분석 시작")
logging.info("📄 전체 민원 수: %d", len(reports))

# 청크 단위 설정
chunk_size = 100000
n_chunks = (len(reports) // chunk_size) + 1
print(f"📦 총 청크 수: {n_chunks}")
logging.info("📦 총 청크 수: %d", n_chunks)

for i in range(n_chunks):
    logging.info("🔄 청크 %d 시작", i)
    print(f"\n🔄 청크 {i+1}/{n_chunks} 처리 중...")
    start_chunk = time.time()
    
    chunk = reports.iloc[i*chunk_size : (i+1)*chunk_size]
    results = []

    for idx, (_, report) in enumerate(tqdm(chunk.iterrows(), total=len(chunk), desc=f"Chunk {i+1} 분석")):
        if idx > 0 and idx % 1000 == 0:
            logging.info("⏳ 청크 %d - %d건 처리 중...", i, idx)

        try:
            r_lat, r_lon = report['위도'], report['경도']
            d_parks = haversine_np(r_lon, r_lat, parks['경도'].values, parks['위도'].values)
            d_cctvs = haversine_np(r_lon, r_lat, cctv['경도'].values, cctv['위도'].values)
            nearby_parks = parks[d_parks <= 300]

            # 요일 처리 함수 (Weekday, Saturday, Holiday만 처리)
            def get_fee_hours(park_row):
                day = report.get('요일')
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
            report_id = report.get('주소', f"(index={report.name})")
            logging.error(f"[청크 {i}] 민원 {report_id} 처리 실패: {repr(e)}")
            logging.error("traceback:\n%s", traceback.format_exc())

    # 저장
    df = pd.DataFrame(results)
    df.to_csv(f"results_part_{i}.csv", index=False)
    elapsed = time.time() - start_chunk
    print(f"✅ 청크 {i+1} 저장 완료 ({len(df)}건), 소요: {elapsed:.2f}초")
    logging.info("✅ 청크 %d 저장 완료 (%d건), 소요 시간: %.2f초", i, len(df), elapsed)

# 전체 시간 기록
end_all = time.time()
total_elapsed = end_all - start_all
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"✅ 전체 소요 시간: {total_elapsed:.2f}초")
logging.info("✅ 전체 분석 종료. 총 소요 시간: %.2f초", total_elapsed)
