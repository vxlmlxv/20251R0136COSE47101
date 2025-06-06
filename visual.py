import pandas as pd
import numpy as np

# ① 데이터 로드
df = pd.read_csv("clustered_contribution_with_location.csv")

# ② 절대값 기준 최대 기여 요인 추출
contrib_cols = ['기여도_총주차면수', '기여도_평균요금', '기여도_평균운영시간', '기여도_CCTV개수']

# 절대값 기준 최대값을 가지는 컬럼명을 새 열로 추가
df['최대기여요인'] = df[contrib_cols].abs().idxmax(axis=1)

# 보기 좋게 이름 바꾸기
df['최대기여요인'] = df['최대기여요인'].map({
    '기여도_총주차면수': '총주차면수',
    '기여도_평균요금': '평균요금',
    '기여도_평균운영시간': '운영시간',
    '기여도_CCTV개수': 'CCTV개수'
})

import folium
from folium.plugins import MarkerCluster

# 색상 지정
color_map = {
    '총주차면수': 'blue',
    '평균요금': 'red',
    '운영시간': 'green',
    'CCTV개수': 'orange'
}

# 서울 중심 좌표 기준 지도 생성
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
marker_cluster = MarkerCluster().add_to(m)

# 마커 추가
for _, row in df.iterrows():
    if pd.notnull(row['위도']) and pd.notnull(row['경도']):
        color = color_map.get(row['최대기여요인'], 'gray')
        popup_text = (
            f"<b>최대 기여 요인:</b> {row['최대기여요인']}<br>"
            f"총주차면수: {row['기여도_총주차면수']:.2f}<br>"
            f"평균요금: {row['기여도_평균요금']:.2f}<br>"
            f"운영시간: {row['기여도_평균운영시간']:.2f}<br>"
            f"CCTV개수: {row['기여도_CCTV개수']:.2f}"
        )
        folium.CircleMarker(
            location=[row['위도'], row['경도']],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(marker_cluster)

# 저장
m.save("map_max_contribution.html")
print("✅ 지도 저장 완료: map_max_contribution.html")
