import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# === 1. 파일 불러오기 ===
print("[STEP 1] Loading files...")
parks_df = pd.read_csv("parks.csv")
reports_df = pd.read_csv("reports.csv")

# === 2. datetime 컬럼 생성 ===
print("[STEP 2] Creating datetime column...")
reports_df["datetime"] = pd.to_datetime(reports_df["민원접수일"] + " " + reports_df["민원접수시간"])

# === 3. 거리 및 운영시간 판단 함수 정의 ===
def simple_distance(lat1, lon1, lat2, lon2):
    return (lat1 - lat2)**2 + (lon1 - lon2)**2

def is_within_operating_hours(report_time, start_str, end_str):
    if start_str == "00:00:00" and end_str == "00:00:00":
        return False
    try:
        start = datetime.strptime(start_str, "%H:%M:%S").time()
        end = datetime.strptime(end_str, "%H:%M:%S").time()
        return start <= report_time.time() <= end
    except:
        return False

# === 4. 민원별로 가장 가까운 주차장 찾고, 운영시간 여부 판단 ===
print("[STEP 3] Matching each report to nearest park...")
results = []
for i, (_, report) in enumerate(tqdm(reports_df.head(5).iterrows(), total=5)):  # 테스트용으로 5개만
    min_dist = float('inf')
    closest_park = None
    for _, park in parks_df.iterrows():
        dist = simple_distance(report["위도"], report["경도"], park["위도"], park["경도"])
        if dist < min_dist:
            min_dist = dist
            closest_park = park

    in_hours = is_within_operating_hours(report["datetime"], closest_park["평일시작"], closest_park["평일종료"])
    
    results.append({
        "민원시간": report["datetime"],
        "민원주소": report["주소"],
        "가까운주차장": closest_park["주소"],
        "distance": min_dist,
        "평일시작": closest_park["평일시작"],
        "평일종료": closest_park["평일종료"],
        "1시간요금": closest_park["1시간 요금"],
        "운영시간내": in_hours
    })

results_df = pd.DataFrame(results)

# === 5. 시간 문자열을 숫자로 변환 ===
print("[STEP 4] Converting time strings to numeric values...")
def time_to_float(tstr):
    try:
        t = datetime.strptime(tstr, "%H:%M:%S").time()
        return t.hour + t.minute / 60
    except:
        return 0

results_df["시작시간"] = results_df["평일시작"].apply(time_to_float)
results_df["종료시간"] = results_df["평일종료"].apply(time_to_float)
results_df["요금"] = results_df["1시간요금"].fillna(0)
results_df["label"] = results_df["운영시간내"].astype(int)

# 테스트용: 라벨 하나를 강제로 1로 설정
results_df.loc[0, "label"] = 1

# === 6. 로지스틱 회귀 실행 ===
print("[STEP 5] Running logistic regression...")
X = results_df[["distance", "시작시간", "종료시간", "요금"]]
y = results_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === 7. 결과 출력 ===
print("\n[STEP 6] Logistic Regression Results:")
print(classification_report(y_test, y_pred))
