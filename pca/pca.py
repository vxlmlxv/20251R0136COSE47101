import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# macOS 기준
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 로그 설정
logging.basicConfig(filename='pca_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 시간 측정 시작
start = time.time()
print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
logging.info("📊 PCA + 클러스터링 분석 시작")

# 1. 데이터 불러오기
df = pd.read_csv("results_all_combined.csv")
logging.info("✅ 데이터 로드 완료. 행 개수: %d", len(df))

# 2. 변수 정리
features = ['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']
X = df[features].dropna()
logging.info("✅ 분석 대상 변수 정리 완료. 유효 행 개수: %d", len(X))

# 3. 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logging.info("✅ 변수 표준화 완료")

# 4. PCA 수행
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
logging.info("✅ PCA 완료. PC1: %.2f%%, PC2: %.2f%% 설명", explained[0]*100, explained[1]*100)

# 5. 변수별 주성분 기여도 저장
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(features))], index=features)
loadings.to_csv("pca_variable_contributions.csv")
logging.info("📄 변수별 주성분 기여도 저장 완료")

# 6. 클러스터링
k = 3
logging.info("🔁 KMeans 클러스터링 시작 (k=%d)", k)
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_pca[:, :2])
logging.info("✅ 클러스터링 완료")

# 7. 클러스터 결과 요약
df_clustered = pd.DataFrame(X, columns=features)
df_clustered['클러스터'] = clusters
summary = df_clustered.groupby('클러스터').mean()
summary.to_csv("cluster_summary.csv")
logging.info("📄 클러스터별 평균 요약 저장 완료")

# 8. 시각화
plt.figure(figsize=(8, 4))
sns.heatmap(loadings, annot=True, cmap='coolwarm')
plt.title("변수별 주성분 기여도")
plt.tight_layout()
plt.savefig("pca_loadings_heatmap.png")
logging.info("🖼 기여도 히트맵 저장 완료")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='Set2', alpha=0.5)
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.title("민원 지점 클러스터링 (PCA 투영)")
plt.legend(title="클러스터")
plt.tight_layout()
plt.savefig("pca_cluster_plot.png")
logging.info("🖼 PCA 클러스터링 시각화 저장 완료")

# 시간 종료
end = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"✅ 전체 분석 소요 시간: {end - start:.2f}초")
logging.info("✅ 전체 분석 완료. 소요 시간: %.2f초", end - start)
