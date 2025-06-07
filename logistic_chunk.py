import pandas as pd
import statsmodels.api as sm
import glob
from datetime import datetime
import time

# 시간 측정 시작
print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start_time = time.time()

# 🔍 파일 목록 불러오기
file_list = sorted(glob.glob("results_chunks/results_part_*.csv"))
print(f"📂 병합할 파일 수: {len(file_list)}")

# 🔗 CSV 병합
df_all = pd.concat([pd.read_csv(file) for file in file_list], ignore_index=True)
df_all.to_csv("results_all_combined.csv", index=False)
print(f"✅ 전체 데이터 수: {len(df_all)}")

# 🧼 결측치 제거
df_all = df_all.dropna()

# 🎯 로지스틱 회귀 분석
X = df_all[['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']]
X = sm.add_constant(X)
y = df_all['민원발생']

logit_model = sm.Logit(y, X)
result = logit_model.fit()

# 📊 결과 출력
print(result.summary())

# 시간 측정 종료
end_time = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"✅ 총 분석 소요 시간: {end_time - start_time:.2f}초")
