import pandas as pd

# Load your data file
file_path = '../PRCData/challenge_set.csv'  # Replace with your actual file path
data = pd.read_csv(file_path)

# Step 1: Group by 'airline', 'aircraft_type', 'adep', 'ades' and count distinct 'tow' values
grouped_data = data.groupby(['airline', 'aircraft_type', 'adep', 'ades'])['tow'].nunique().reset_index(name='different_tow_count')

# Step 2: Calculate the total count of rows in each group
total_counts = data.groupby(['airline', 'aircraft_type', 'adep', 'ades']).size().reset_index(name='total_count')

# Step 3: Merge the different_tow_count and total_count
merged_data = pd.merge(grouped_data, total_counts, on=['airline', 'aircraft_type', 'adep', 'ades'])

# Step 4: Calculate the percentage of unique 'tow' values
merged_data['tow_percentage'] = ( merged_data['total_count']/merged_data['different_tow_count']) * 100

# Step 5: Check for groups with only one unique 'tow' value and add the 'tow' value to the final output
# Merge with original data to get the actual 'tow' values for groups with only one unique tow
single_tow_data = data.groupby(['airline', 'aircraft_type', 'adep', 'ades']).filter(lambda x: x['tow'].nunique() == 1)
single_tow_values = single_tow_data.drop_duplicates(subset=['airline', 'aircraft_type', 'adep', 'ades'])[['airline', 'aircraft_type', 'adep', 'ades', 'tow']]

# Rename the column for clarity and merge with merged_data
single_tow_values = single_tow_values.rename(columns={'tow': 'single_tow_value'})
final_output = pd.merge(merged_data, single_tow_values, on=['airline', 'aircraft_type', 'adep', 'ades'], how='left')
final_output = final_output.dropna(subset=['single_tow_value'])
final_output = final_output[['airline', 'aircraft_type', 'adep', 'ades','total_count','single_tow_value']]
# Display the final result
final_output.to_csv('../some_standard_tow.csv', index=False)