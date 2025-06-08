import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from datetime import datetime
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import glob
import folium
from folium.plugins import MarkerCluster

# 로그 설정
logging.basicConfig(filename='log_contribution_cluster.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start_all = time.time()

# 📁 파일 경로 설정
positive_files = glob.glob("result_chunks/results_part_*.csv")
negative_files = glob.glob("random_chunks/negative_part_*.csv")

# 🔄 데이터 병합
dfs = []
print(f"📂 민원 발생 파일 {len(positive_files)}개 병합 중...")
for file in tqdm(positive_files, desc="Positive"):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        logging.error(f"파일 {file} 읽기 실패: {e}")

print(f"📂 민원 없음 파일 {len(negative_files)}개 병합 중...")
for file in tqdm(negative_files, desc="Negative"):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        logging.error(f"파일 {file} 읽기 실패: {e}")

# 🔗 전체 병합
full_df = pd.concat(dfs, ignore_index=True)
print(f"✅ 총 데이터 수: {len(full_df)}")

# 🧹 결측값 제거
clean_df = full_df.dropna(subset=['총주차면수', '평균요금', '평균운영시간', 'CCTV개수', '민원발생', '위도', '경도'])

# 🧩 interaction terms 생성
clean_df['총주차면수_운영시간'] = clean_df['총주차면수'] * clean_df['평균운영시간']
clean_df['평균요금_운영시간'] = clean_df['평균요금'] * clean_df['평균운영시간']
clean_df['총주차면수_평균요금'] = clean_df['총주차면수'] * clean_df['평균요금']

# 🎯 로지스틱 회귀 전처리
feature_cols = [
    '총주차면수', '평균요금', '평균운영시간', 'CCTV개수',
    '총주차면수_운영시간', '평균요금_운영시간', '총주차면수_평균요금'
]
X_raw = clean_df[feature_cols]
y = clean_df['민원발생']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
X_scaled = sm.add_constant(X_scaled)

print("🔍 로지스틱 회귀 분석 중...")
try:
    model = sm.Logit(y, X_scaled).fit()
    print(model.summary())
except Exception as e:
    print(f"⚠️ 회귀 분석 실패: {e}")
    logging.error(f"회귀 분석 실패: {e}")
    exit()

# ✅ 계수 추출 (상수 제외)
coef = model.params.drop('const')
print("📌 회귀 계수:", coef.to_dict())

# 💡 기여도 계산 (민원발생 == 1인 지점만)
positive_only = clean_df[clean_df['민원발생'] == 1].copy()
X_pos_raw = positive_only[feature_cols]
X_pos_scaled = pd.DataFrame(scaler.transform(X_pos_raw), columns=feature_cols)

contribs = []
print("🧮 기여도 계산 중...")
for idx, row in tqdm(X_pos_scaled.iterrows(), total=len(X_pos_scaled)):
    contrib = {}
    for var in coef.index:
        contrib[f"기여도_{var}"] = coef[var] * row[var]
    contrib['위도'] = positive_only.iloc[idx]['위도']
    contrib['경도'] = positive_only.iloc[idx]['경도']
    contribs.append(contrib)

contrib_df = pd.DataFrame(contribs)
df_with_contrib = pd.concat([positive_only.reset_index(drop=True), contrib_df], axis=1)

# 🔗 클러스터링
contrib_only = contrib_df[[col for col in contrib_df.columns if col.startswith("기여도_")]]
kmeans = KMeans(n_clusters=3, random_state=42)
df_with_contrib['클러스터'] = kmeans.fit_predict(contrib_only)

# 💾 저장
df_with_contrib.to_csv("clustered_contribution_with_interaction.csv", index=False)
print("✅ 결과 저장 완료: clustered_contribution_with_interaction.csv")
