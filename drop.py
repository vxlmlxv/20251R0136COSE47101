import pandas as pd

parks_data = "seoul_parking_info.csv"
report_data = "illegal_parking_reports.csv"
df_parks = pd.read_csv(parks_data, encoding='utf-8')
df_reports = pd.read_csv(report_data, encoding='utf-8')
# Display rows with missing values in df_reports
print("Rows with missing values in df_reports:")
print(df_reports[df_reports.isnull().any(axis=1)])

# Remove rows with missing values in df_reports
df_reports.dropna(inplace=True)

# Display rows with missing values in df_parks
print("\nRows with missing values in df_parks:")
print(df_parks[df_parks.isnull().any(axis=1)])

# Remove rows with missing values in df_parks
df_parks.dropna(inplace=True)

# Save the cleaned dataframes to new CSV files
df_parks.to_csv('cleaned_seoul_parking_info.csv', index=False, encoding='utf-8-sig')
df_reports.to_csv('cleaned_illegal_parking_reports.csv', index=False, encoding='utf-8-sig')

