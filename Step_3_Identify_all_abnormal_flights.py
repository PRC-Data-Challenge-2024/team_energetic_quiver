import pandas as pd
import os
from tqdm import tqdm
def drop_abnormal_flight_info(df):
    for row_i in tqdm(range((len(df)))):
        if df.at[row_i, 'time_to_reach_altitude_mode'] == 0:
            df.at[row_i, 'time_to_reach_altitude_mode'] = None
        if df.at[row_i, 'takeoff_groundspeed'] ==0:
            df.at[row_i, 'takeoff_groundspeed'] = None
        if df.at[row_i, 'cruising_altitude'] < 20000:
            df.at[row_i, 'cruising_altitude'] = None
        if pd.isna(df.at[row_i, 'time_to_reach_altitude_mode']):
            df.at[row_i, 'average_u_component_of_wind'] = None
            df.at[row_i, 'average_v_component_of_wind'] = None
            df.at[row_i, 'average_temperature'] = None
            df.at[row_i, 'from_start_time_to_reach_altitude_mode'] =None
            df.at[row_i, 'cruising_altitude'] = None
        if pd.isna(df.at[row_i, 'cruising_altitude']):
            df.at[row_i, 'average_u_component_of_wind'] = None
            df.at[row_i, 'average_v_component_of_wind'] = None
            df.at[row_i, 'average_temperature'] = None
            df.at[row_i, 'from_start_time_to_reach_altitude_mode'] = None
            df.at[row_i, 'time_to_reach_altitude_mode'] = None
    return df


if __name__ == '__main__':
    if os.path.exists('../combined_flight_info.csv'):
        df = pd.read_csv('../combined_flight_info.csv')
    else:
        # Specify the folder path where your CSV files are located
        folder_path = '../Mid_results_for_combine_flight_info/'

        # Create an empty list to store individual DataFrames
        csv_list = []

        Count = 0
        # Loop through all files in the folder
        for file in os.listdir(folder_path):
            Count = Count + 1
            # Check if the file is a CSV file
            if file.endswith('.csv'):
                print('Number:{} File:name: {}'.format(Count, file))
                # Read the CSV file and append it to the list
                csv_path = os.path.join(folder_path, file)
                df = pd.read_csv(csv_path)
                csv_list.append(df)

        # Concatenate all DataFrames into one
        combined_df = pd.concat(csv_list, ignore_index=True)

        # Optionally, save the combined DataFrame to a new CSV file
        combined_df.to_csv('../combined_flight_info.csv', index=False)
        df = pd.read_csv('../combined_flight_info.csv')

    df_flight_info = drop_abnormal_flight_info(df)
    df_flight_info = df_flight_info[
        ["flight_id", "date", "cruising_altitude", "highest_altitude", "takeoff_groundspeed",
         "takeoff_time_duration", "from_start_time_to_reach_altitude_mode", "time_to_reach_altitude_mode",
         "average_u_component_of_wind", "average_v_component_of_wind", "average_temperature", "duration"]]
    df_flight_info.to_csv('../combined_flight_info_drop_abnormal.csv', index=False)