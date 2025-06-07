import pandas as pd
import glob
import statsmodels.api as sm
import time
from datetime import datetime
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans

# 로그 설정
logging.basicConfig(filename='log_contribution_cluster.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start_all = time.time()

# 📁 파일 경로 설정
positive_files = glob.glob("results_chunks/results_part_*.csv")
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
clean_df = full_df.dropna()

# 🎯 로지스틱 회귀
X = clean_df[['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']]
y = clean_df['민원발생']
X = sm.add_constant(X)

print("🔍 로지스틱 회귀 분석 중...")
try:
    model = sm.Logit(y, X).fit()
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

contribs = []
print("🧮 기여도 계산 중...")
for idx, row in tqdm(positive_only.iterrows(), total=len(positive_only)):
    contrib = {}
    for var in coef.index:
        contrib[f"기여도_{var}"] = coef[var] * row[var]
    contribs.append(contrib)

contrib_df = pd.DataFrame(contribs)
df_with_contrib = pd.concat([positive_only.reset_index(drop=True), contrib_df], axis=1)

# 📊 클러스터링 (기여도 기준)
kmeans = KMeans(n_clusters=3, random_state=42)
df_with_contrib['클러스터'] = kmeans.fit_predict(contrib_df)

# 💾 저장
output_file = "clustered_contribution_from_chunks.csv"
df_with_contrib.to_csv(output_file, index=False)
print(f"✅ 결과 저장 완료: {output_file}")

end_all = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"⏱ 전체 소요 시간: {end_all - start_all:.2f}초")
