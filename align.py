import pandas as pd
import os

# 기준 파일 로드
rf_parksdf = pd.read_csv("src/RF_parksdf.csv")
target_columns = [col for col in rf_parksdf.columns if col != "Unnamed: 0"]

# 컬럼명 매핑
column_rename_map = {
    '주차장코드': 'id',
    '주소': 'address',
    '주차장종류': 'parking_type',
    '운영구분': 'operation_type',
    '총주차면': 'total_parking_spaces',
    '평일유료': 'Weekday_paid',
    '토요일유료': 'Saturday_paid',
    '공휴일유료': 'Holiday_paid',
    '평일시작': 'Weekday_start',
    '평일종료': 'Weekday_end',
    '토요일시작': 'Saturday_start',
    '토요일종료': 'Saturday_end',
    '공휴일시작': 'Holiday_start',
    '공휴일종료': 'Holiday_end',
    '기본주차요금': 'base_parking_fee',
    '기본주차시간': 'base_parking_time',
    '추가단위요금': 'additional_unit_fee',
    '추가단위시간': 'additional_unit_time',
    '평일운영시간': 'Weekday_operatingHours',
    '토요일운영시간': 'Saturday_operatingHours',
    '공휴일운영시간': 'Holiday_operatingHours',
    '1시간 요금': 'fee1H',
    'complaints_r300_평일': 'complaints_r300',
    'complaints_r300_토요일': 'complaints_r300',
    'complaints_r300_공휴일': 'complaints_r300',
    'lon': 'longitude',
    'lat': 'latitude'
}

# 변환 대상 파일들
file_map = {
    "src/parks_weekday.csv": "src/RF_parksdf_weekday.csv",
    "src/parks_saturday.csv": "src/RF_parksdf_saturday.csv",
    "src/parks_holiday.csv": "src/RF_parksdf_holiday.csv"
}

# 변환 루프
for input_path, output_path in file_map.items():
    df = pd.read_csv(input_path)
    
    # 기존 complaints_r300이 있으면 삭제
    if "complaints_r300" in df.columns:
        df = df.drop(columns=["complaints_r300"])
    
    # 컬럼 정리 및 변환
    df.columns = df.columns.str.strip()
    df = df.rename(columns=column_rename_map)
    
    # 누락 컬럼 추가 (NaN), 순서 정렬
    for col in target_columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[target_columns]
    
    df.to_csv(output_path, index=False)
    print(f"✔ 저장 완료: {output_path}")
