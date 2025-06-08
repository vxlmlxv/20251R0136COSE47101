import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# macOS 기준
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📁 파일 병합
positive_files = glob.glob("result_chunks_filtered/results_part_*.csv")
negative_files = glob.glob("random_chunks_label0/negative_part_*.csv")

dfs = []
for file in positive_files + negative_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except:
        continue

full_df = pd.concat(dfs, ignore_index=True).dropna()

# 🎯 관심 변수
cols = ['총주차면수', '평균요금', '평균운영시간', 'CCTV개수', '민원발생']

# 📌 상관관계 히트맵
sns.heatmap(full_df[cols].corr(), annot=True, cmap='coolwarm')
plt.title("📊 Pearson Correlation Heatmap")
plt.show()

# 📌 민원발생 기준 평균값 비교
grouped = full_df[cols].groupby('민원발생').mean().T
grouped.plot(kind='barh', figsize=(8, 5), legend=True, title='Mean by 민원발생')
plt.grid(True)
plt.tight_layout()
plt.show()

# 📌 산점도 행렬 (pairplot)
sns.pairplot(full_df[cols], hue="민원발생", palette="Set2")
plt.suptitle("🔍 Pairwise Scatter Matrix (민원발생 별)", y=1.02)
plt.show()

# 📌 조건부 민원 비율 분석 (예: 평균요금 x CCTV 개수)
full_df['요금_bin'] = pd.qcut(full_df['평균요금'], 3, labels=["낮음", "중간", "높음"])
full_df['CCTV_bin'] = pd.cut(full_df['CCTV개수'], bins=[-1,1,3,100], labels=["적음", "중간", "많음"])

pivot = pd.pivot_table(full_df, values='민원발생',
                       index='요금_bin', columns='CCTV_bin', aggfunc='mean')

sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".2f")
plt.title("📌 민원발생률 (평균요금 x CCTV 개수)")
plt.show()
