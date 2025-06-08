import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm

# 📁 label=1 (positive)와 label=0 (negative) 데이터 불러오기
positive_files = glob.glob("result_chunks_filtered/results_part_*.csv")
negative_files = glob.glob("result_chunks_label0/results_part_*.csv")

dfs = []
for file in tqdm(positive_files + negative_files, desc="CSV 파일 병합 중"):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"❌ 파일 {file} 오류: {e}")

# 🔗 전체 병합
full_df = pd.concat(dfs, ignore_index=True)

# 🧹 결측치 제거
clean_df = full_df.dropna(subset=['총주차면수', '평균요금', '평균운영시간', 'CCTV개수', '민원발생'])

# ➕ interaction term 추가
clean_df['요금_CCTV'] = clean_df['평균요금'] * clean_df['CCTV개수']
clean_df['요금_면수'] = clean_df['평균요금'] * clean_df['총주차면수']

# 🎯 로지스틱 회귀
feature_cols = [
    '총주차면수', '평균요금', '평균운영시간', 'CCTV개수',
    '요금_CCTV', '요금_면수'
]
X = clean_df[feature_cols]
y = clean_df['민원발생']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_scaled = sm.add_constant(X_scaled)

print("🔍 로지스틱 회귀 분석 시작...")
model = sm.Logit(y, X_scaled)
result = model.fit()

# 📊 결과 출력
print(result.summary())
