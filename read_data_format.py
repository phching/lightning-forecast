import os
import pandas as pd
from datetime import datetime, timedelta

folder_path = "llis_20250315"

# Function to check if a file can be opened and read, and show details
def check_file_readability(file_path):
    try:
        df = pd.read_csv(file_path, sep=r'\s+', names=[
            'version', 'year', 'month', 'day', 'hour', 'minutes', 'seconds', 'nanoseconds',
            'latitude', 'longitude', 'peak_current','multiplicity','number_of_sensors', 'degrees_freedom', 
            'ellipse_angle', 'semi_major_axis','semi-minor_axis', 'chi_square', 'rise_time', 'peak_to_zero_time',
            'max_rate_of_rise', 'cloud_indicator', 'angle_indicator', 'signal_indicator', 'timing_indicator'
        ])

        print(f"\nSuccessfully opened and read: {file_path} (Rows: {len(df)})")
        # Show detailed information
        print(f"\n--- Details for {file_path} ---")
        # Preview of the first 5 rows
        print("\nPreview of first 5 rows:")
        print(df.head().to_string(index=False))
        
        # 2. Data types of columns
        print("\nColumn data types:")
        print(df.dtypes)
        
        # 3. Specific details for key columns
        if not df.empty:
            print("\nKey column ranges:")
            print(f"Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
            print(f"Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
            print(f"Peak current range: {df['peak_current'].min():.2f} to {df['peak_current'].max():.2f}")
            print(f"Number of sensors range: {df['number_of_sensors'].min()} to {df['number_of_sensors'].max()}")
        
        return True
    except Exception as e:
        print(f"Failed to open or read {file_path}: {str(e)}")
        return False

# Generate the list of expected filenames (every minute from 00:01 to 23:13)
start_time = datetime(2025, 3, 15, 23, 11)
end_time = datetime(2025, 3, 15, 23, 13)
time_format = "%Y%m%d%H%M"
expected_files = []
current_time = start_time

while current_time <= end_time:
    expected_files.append(f"llis_{current_time.strftime(time_format)}")
    current_time += timedelta(minutes=1)

# Check each file in the folder
print(f"Checking files in {folder_path}...")
file_count = 0
success_count = 0

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) and filename in expected_files:
        file_count += 1
        if check_file_readability(file_path):
            success_count += 1

# Summary
print(f"\nTotal files checked: {file_count}")
print(f"Expected files: {len(expected_files)}")
print(f"Successfully opened files: {success_count}")
if file_count > 0:
    print(f"Success rate: {success_count / file_count * 100:.2f}%")
else:
    print("No files found to check.")