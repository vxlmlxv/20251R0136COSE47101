import pandas as pd

# 파일 경로 지정
weekday_path = "src/RF_parksdf_weekday.csv"
saturday_path = "src/RF_parksdf_saturday.csv"
holiday_path = "src/RF_parksdf_holiday.csv"

# 공통 컬럼
base_columns = [
    'address', 'parking_type', 'operation_type', 'total_parking_spaces',
    'base_parking_fee', 'base_parking_time', 'additional_unit_fee', 'additional_unit_time',
    'fee1H', 'cctv_r300', 'bus_r300', 'complaints_r300'
]

# 요일별 추가 컬럼 정의
weekday_cols = base_columns + [
    'Weekday_paid', 'Weekday_start', 'Weekday_end', 'Weekday_operatingHours'
]
saturday_cols = base_columns + [
    'Saturday_paid', 'Saturday_start', 'Saturday_end', 'Saturday_operatingHours'
]
holiday_cols = base_columns + [
    'Holiday_paid', 'Holiday_start', 'Holiday_end', 'Holiday_operatingHours'
]

# 파일 열기 및 컬럼 필터링
weekday_df = pd.read_csv(weekday_path)[weekday_cols]
saturday_df = pd.read_csv(saturday_path)[saturday_cols]
holiday_df = pd.read_csv(holiday_path)[holiday_cols]

# 저장 경로
weekday_out = "src/RF_parksdf_week.csv"
saturday_out = "src/RF_parksdf_sat.csv"
holiday_out = "src/RF_parksdf_holi.csv"

# 파일 저장
weekday_df.to_csv(weekday_out, index=False)
saturday_df.to_csv(saturday_out, index=False)
holiday_df.to_csv(holiday_out, index=False)

print("✔ 요일별 컬럼만 추출하여 저장 완료")