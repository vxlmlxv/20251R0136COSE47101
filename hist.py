import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm

# macOS 기준
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 📁 모든 청크 csv 불러오기
all_files = glob.glob("result_chunks/results_part_*.csv")
print(f"🔎 불러올 파일 수: {len(all_files)}")

df_list = []
for file in tqdm(all_files, desc="파일 통합 중"):
    df = pd.read_csv(file)
    df_list.append(df)

df_all = pd.concat(df_list, ignore_index=True)
print(f"✅ 전체 통합 완료. 총 행 수: {len(df_all)}")

# 📊 변수 목록
features = ['총주차면수', '평균요금', '평균운영시간', 'CCTV개수']

# 🎨 시각화 (히스토그램 + KDE)
plt.figure(figsize=(16, 10))
for i, var in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.histplot(df_all[var], bins=50, kde=True, color='skyblue')
    plt.title(f'{var} 분포')
    plt.xlabel(var)
    plt.ylabel('빈도수')
    plt.grid(True)

plt.tight_layout()
plt.savefig("변수별_분포_1.png", dpi=300)
print("📸 저장 완료: 변수별_분포_1.png")
