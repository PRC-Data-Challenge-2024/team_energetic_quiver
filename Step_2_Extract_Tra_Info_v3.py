import pandas as pd
import numpy as np
import time
import os
import math
from tqdm import tqdm
import statistics
def extract_first_10_rows(df, col_name):
    first_10_values = [None]
    for first_i in reversed(range(1, 11)):
        try:
            first_10_values = df[col_name].iloc[:first_i].tolist()
            if len(first_10_values)==first_i:
                break
        except:
            continue
    # Assign each value to a separate variable
    if len(first_10_values) < 10:
        first_10_values += [None] * (10 - len(first_10_values))  # Pad with None if less than 10 values

    # var1, var2, var3, var4, var5, var6, var7, var8, var9, var10 = first_10_values
    # return var1, var2, var3, var4, var5, var6, var7, var8, var9, var10
    return first_10_values


def process_flight(df, file_name):
    df = df.sort_values(by=['timestamp'], ascending=True)
    # 0. Check the groundspeed at the starting timestamp to see if it is a normal trajectory
    try:
        duration = df['timestamp'].max() - df['timestamp'].min()
    except:
        duration = None

    first_10_ground_speed = extract_first_10_rows(df, 'groundspeed')
    first_10_vertical_rate = extract_first_10_rows(df, 'vertical_rate')
    first_10_altitude = extract_first_10_rows(df, 'altitude')
    # try:
        # groundspeed1, groundspeed2, groundspeed3, groundspeed4, groundspeed5, groundspeed6, groundspeed7, groundspeed8, groundspeed9, groundspeed10 = extract_first_10_rows(
        #     df, 'groundspeed')
    # except:
    #     groundspeed1, groundspeed2, groundspeed3, groundspeed4, groundspeed5, groundspeed6, groundspeed7, groundspeed8, groundspeed9, groundspeed10 = [None] * 10
    # try:
    #     vertical_rate1, vertical_rate2, vertical_rate3, vertical_rate4, vertical_rate5, vertical_rate6, vertical_rate7, vertical_rate8, vertical_rate9, vertical_rate10 = extract_first_10_rows(
    #         df, 'vertical_rate')
    # except:
    #     vertical_rate1, vertical_rate2, vertical_rate3, vertical_rate4, vertical_rate5, vertical_rate6, vertical_rate7, vertical_rate8, vertical_rate9, vertical_rate10 = [None] * 10
    # try:
    #     altitude1, altitude2, altitude3, altitude4, altitude5, altitude6, altitude7, altitude8, altitude9, altitude10 = extract_first_10_rows(
    #         df, 'altitude')
    # except:
    #     altitude1, altitude2, altitude3, altitude4, altitude5, altitude6, altitude7, altitude8, altitude9, altitude10 = [None] * 10

    # 1. Cruising altitude
    try:
        # Count the frequency of each altitude and sort by count, then by time_stamp
        half_len = int(len(df)/2)
        df_altitude_half = df.iloc[:half_len].copy()
        df_altitude_half.loc[:, 'altitude_count'] = df_altitude_half.groupby('altitude')['altitude'].transform('count')
        df_sorted = df_altitude_half.sort_values(by=['altitude_count', 'timestamp'], ascending=[False, True])
        for _, row in df_sorted.iterrows():
            vertical_rate = row['vertical_rate']
            if abs(vertical_rate) > 10:
                altitude_mode = None
                continue
            else:
                altitude_mode = row['altitude']
                break
    except:
        altitude_mode = None

    # 2. The highest altitude that the aircraft achieved
    try:
        highest_altitude = df['altitude'].max()
    except:
        highest_altitude = None

    # 3. Ground speed at takeoff (when altitude first increases above the initial altitude)
    try:
        initial_altitude = df['altitude'].iloc[0]  # Assuming the first altitude is the initial altitude
        takeoff_row = df[df['altitude'] > initial_altitude].iloc[
            0]  # First row where altitude is higher than the initial altitude
        takeoff_groundspeed = takeoff_row['groundspeed']
    except:
        takeoff_groundspeed = None
        takeoff_row = None

    # 4. Time taken for takeoff (time from start to when altitude increases beyond initial altitude)
    try:
        takeoff_time_duration = takeoff_row['timestamp'] - df['timestamp'].iloc[0]  # Difference in timestamp
    except:
        takeoff_time_duration = None

    # 5. Time to achieve the cruising level
    try:
        mode_altitude_time_row = df[df['altitude'] == altitude_mode].iloc[
            0]  # First row where altitude equals the mode altitude
        from_start_time_to_reach_altitude_mode = mode_altitude_time_row['timestamp'] - df['timestamp'].iloc[
            0]  # Difference in timestamp
    except:
        mode_altitude_time_row = None
        from_start_time_to_reach_altitude_mode = None


        # Trajectories that contain taxiing phase
    try:
        if statistics.mode(first_10_vertical_rate) == 0:
            df_before_cruising = df[df['timestamp'] < mode_altitude_time_row['timestamp']]
            df_before_cruising = df_before_cruising.sort_values(by=['timestamp'], ascending=[True])
            window_size=10
            for start in range(len(df_before_cruising) - window_size + 1):
                window = df_before_cruising['vertical_rate'].iloc[start:start + window_size]
                # Check if at least 7 values in the window are greater than 0
                if (window > 0).sum() >= 7:
                    # Find the first row in this window where vertical_rate is greater than 0
                    for i, value in enumerate(window):
                        if value > 0:
                            # Select the original row where this value appears
                            start_climbing_row = df_before_cruising.iloc[start + i]
                            break
                    break
            time_to_reach_altitude_mode = mode_altitude_time_row['timestamp'] - start_climbing_row['timestamp']
        else:
            start_climbing_row = df.iloc[0]
            time_to_reach_altitude_mode = from_start_time_to_reach_altitude_mode

    except:
        try:
            start_climbing_row = df.iloc[0]
            time_to_reach_altitude_mode = from_start_time_to_reach_altitude_mode
        except:
            start_climbing_row = None
            time_to_reach_altitude_mode =None

    # Filter data between the start and the time of reaching the mode altitude
    try:
        df_during_mode_altitude = df[
            (df['timestamp'] >= start_climbing_row['timestamp']) & (df['timestamp'] <= mode_altitude_time_row['timestamp'])]
        # 6. The average of u_component_of_wind during the time to achieve the mode altitude
        average_u_wind = df_during_mode_altitude['u_component_of_wind'].mean()
        # 7. The average of v_component_of_wind during the time to achieve the mode altitude
        average_v_wind = df_during_mode_altitude['v_component_of_wind'].mean()
        # 8. The average of temperature during the time to achieve the mode altitude
        average_temperature = df_during_mode_altitude['temperature'].mean()
    except:
        average_u_wind = None
        average_v_wind = None
        average_temperature = None


    return_dict = {
    "flight_id": df['flight_id'].iloc[0],
    "date": file_name,
    "cruising_altitude": altitude_mode,
    "highest_altitude": highest_altitude,
    "takeoff_groundspeed": takeoff_groundspeed,
    "takeoff_time_duration": takeoff_time_duration,
    "from_start_time_to_reach_altitude_mode":from_start_time_to_reach_altitude_mode,
    "time_to_reach_altitude_mode": time_to_reach_altitude_mode,
    "average_u_component_of_wind": average_u_wind,
    "average_v_component_of_wind": average_v_wind,
    "average_temperature": average_temperature,
    **{f"ground_speed_{i+1}": first_10_ground_speed[i] if i < len(first_10_ground_speed) else None for i in range(10)},
    **{f"vertical_speed_{i+1}": first_10_vertical_rate[i] if i < len(first_10_vertical_rate) else None for i in range(10)},
    **{f"altitude_{i+1}": first_10_altitude[i] if i < len(first_10_altitude) else None for i in range(10)},
    "duration":duration
    }
    return return_dict
def process_error_file(df, file_name):
    return {
        "flight_id": df['flight_id'].iloc[0],
        "date": file_name,
        "cruising_altitude": None,
        "highest_altitude": None,
        "takeoff_groundspeed": None,
        "takeoff_time_duration": None,
        "from_start_time_to_reach_altitude_mode": None,
        "time_to_reach_altitude_mode": None,
        "time_to_reach_altitude_mode": None,
        "average_u_component_of_wind": None,
        "average_v_component_of_wind": None,
        "average_temperature": None,
        **{f"ground_speed_{i+1}": None for i in range(10)},
        **{f"vertical_speed_{i+1}": None for i in range(10)},
        **{f"altitude_{i+1}": None for i in range(10)},
        "duration": None
    }

def main():
    # Read data with error
    df_error = pd.read_csv('../Filtered_process_error.csv')
    df_error[['date', 'id']] = df_error['flight_id'].str.split('_', expand=True)
    df_error['date'] = df_error['date'].astype(str)

    # Convert the 'number' column to numeric type (integer or float)
    df_error['id'] = pd.to_numeric(df_error['id'], errors='coerce')
    lst_error_date = df_error['date'].tolist()

    # Read all flight need to be processed
    df_challenge = pd.read_csv('../PRCData/challenge_set.csv')
    df_submission = pd.read_csv('../PRCData/final_submission_set.csv')
    df_all_flights_need = pd.concat([df_challenge, df_submission], ignore_index=True)
    df_all_flights_need = df_all_flights_need[['flight_id', 'date']]
    df_all_flights_need['date'] = df_all_flights_need['date'].astype(str)

    # Define the directory containing all parquet files
    file_directory = '../PRCData_daily_filtered/'

    # List all parquet files in the directory
    parquet_files = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if f.endswith('.parquet')]

    # Completed file
    complete_file_path = '../Mid_results_for_combine_flight_info/'
    # lst_complete_files = [f for f in os.listdir(complete_file_path) if f.endswith('.csv')]
    # complete_files = [file.split('.')[0] for file in lst_complete_files]


    # Initialize an empty list to collect results for all files
    # combined_results = []
    COUNT=0
    # Iterate over each parquet file
    for file in parquet_files:
        file_name = file.split('/')[-1].split('.')[0]
        # if file_name in complete_files:
        #     print('File {} completed'.format(file_name))
        #     continue
        # else:
        if file_name in lst_error_date:
            error_file = True
            df_error_id = df_error[df_error['date']==file_name]
            lst_error_id = df_error_id['id'].tolist()
        else:
            error_file = False


        COUNT = COUNT+1
        print(COUNT)
        print('Number: {}, Filename: {}'.format(COUNT, file_name))
        # Load the current parquet file
        df = pd.read_parquet(file)

        df_need_process = df_all_flights_need[df_all_flights_need['date'] == file_name]

        # Get a list of unique flight_ids in the current file
        raw_flight_ids = df_need_process['flight_id'].unique()
        flight_ids = [x for x in raw_flight_ids if x is not None and not (isinstance(x, float) and math.isnan(x))]

        each_date = []

        # Process each flight within the current file
        for flight_id in tqdm(flight_ids):
            flight_df = df[df['flight_id'] == flight_id]
            if not flight_df.empty:
                if (error_file==True) and (flight_id in lst_error_id):
                    flight_metrics = process_error_file(flight_df, file_name)
                else:
                    flight_metrics = process_flight(flight_df, file_name)
            # combined_results.append(flight_metrics)
            each_date.append(flight_metrics)

        df_each_date = pd.DataFrame(each_date)
        df_each_date['takeoff_time_duration'] = df_each_date['takeoff_time_duration'].apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
        df_each_date['time_to_reach_altitude_mode'] = df_each_date['time_to_reach_altitude_mode'].apply(lambda x: x.total_seconds() if pd.notnull(x) else x)
        df_each_date['from_start_time_to_reach_altitude_mode'] = df_each_date['from_start_time_to_reach_altitude_mode'].apply(
            lambda x: x.total_seconds() if pd.notnull(x) else x)
        df_each_date['duration'] = df_each_date['duration'].apply(
            lambda x: x.total_seconds() if pd.notnull(x) else x)
        df_each_date.to_csv(complete_file_path + file_name +'.csv', index=False)
'''
        combined_df = pd.DataFrame(combined_results)

        # Convert the `takeoff_time_duration` and `time_to_reach_altitude_mode` to seconds if they are in nanoseconds
        combined_df['takeoff_time_duration'] = combined_df['takeoff_time_duration'].apply(
            lambda x: x.total_seconds() if pd.notnull(x) else x)
        combined_df['time_to_reach_altitude_mode'] = combined_df['time_to_reach_altitude_mode'].apply(
            lambda x: x.total_seconds() if pd.notnull(x) else x)

        combined_df.to_csv('../combined_results_pandas.csv', index=False)
'''

if __name__ == '__main__':

    main()


