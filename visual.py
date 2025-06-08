import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
from datetime import datetime
import time
from tqdm import tqdm

print("⏱ 시작 시각:", datetime.now().strftime("%H:%M:%S"))
start = time.time()

# 📁 CSV 파일 불러오기
input_file = "clustered_contribution_with_interaction.csv"
df = pd.read_csv(input_file)

# 🧹 위도/경도 결측 제거 및 float 변환
df = df.dropna(subset=['위도', '경도'])
df['위도'] = df['위도'].astype(float)
df['경도'] = df['경도'].astype(float)

# 🎯 지도 중심 위치
map_center = [df['위도'].mean(), df['경도'].mean()]

# =====================
# 📍 1. 샘플링 마커 클러스터 시각화
# =====================
print("📍 샘플링 마커 클러스터 시각화 생성 중...")
df_sample = df.sample(n=100000, random_state=42)

map_sample = folium.Map(location=map_center, zoom_start=11)
marker_cluster = MarkerCluster().add_to(map_sample)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue']
for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="샘플 마커"):
    folium.CircleMarker(
        location=(row['위도'], row['경도']),
        radius=4,
        color=colors[int(row['클러스터']) % len(colors)],
        fill=True,
        fill_opacity=0.6,
        popup=f"Cluster {int(row['클러스터'])}"
    ).add_to(marker_cluster)

map_sample.save("cluster_map_sample.html")
print("✅ 샘플링 지도 저장 완료: cluster_map_sample.html")

# 종료 시간
end = time.time()
print("⏱ 종료 시각:", datetime.now().strftime("%H:%M:%S"))
print(f"⏱ 전체 소요 시간: {end - start:.2f}초")
