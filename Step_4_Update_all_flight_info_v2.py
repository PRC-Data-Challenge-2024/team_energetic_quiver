import os
import pandas as pd
from tqdm import tqdm
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from functools import partial
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Function to get latitude and longitude for an airport code using Nominatim
def get_airport_coordinates(airport_code, dict_airport_code):
    if airport_code in dict_airport_code:
        return dict_airport_code[airport_code]
    else:

        geolocator = Nominatim(user_agent="airport_distance_calculator")
        location = geolocator.geocode(f"{airport_code} airport")

        if location:
            return (location.latitude, location.longitude)
        else:
            raise ValueError(f"Coordinates for airport code {airport_code} not found.")

def get_airport_location(df_file_challenge,df_file_submission):
    # For calculate all pairs
    df_file_c_s = pd.concat([df_file_submission, df_file_challenge], ignore_index=True)

    unique_airports = df_file_c_s['ades'].unique().tolist()
    unique_airports_2 = df_file_c_s['adep'].unique().tolist()
    unique_airports_2.extend(unique_airports)
    unique_airports_3 = list(set(unique_airports_2))
    dict_airport_code = {'EGCN':(53.480537, -1.010656), 'OKBK':(29.226917, 47.973026), 'HSSK':(15.589497, 32.553161)}

    df_airport_location = pd.DataFrame(columns=['airport', 'latitude', 'longitude'])
    lst_airport_code =df_airport_location['airport'].tolist()
    lst_latitude = df_airport_location['latitude'].tolist()
    lst_longitude = df_airport_location['longitude'].tolist()

    for airport in tqdm(unique_airports_3):
        if airport in lst_airport_code:
            continue
        else:
            location= get_airport_coordinates(airport, dict_airport_code)
            lst_airport_code.append(airport)
            lst_latitude.append(location[0])
            lst_longitude.append(location[1])

            df_airport_location=pd.DataFrame({'airport':lst_airport_code, 'latitude':lst_latitude, 'longitude':lst_longitude})
            df_airport_location.to_csv('../airport_location.csv', index=False)

def get_pair_dist(df_file_challenge, df_file_submission):
    # Function to get latitude and longitude for an airport code using Nominatim
    def get_airport_coordinates(airport_code, df_airport_loc):
        latitude = df_airport_loc[df_airport_loc['airport'] == airport_code]['latitude'].iloc[0]
        longitude = df_airport_loc[df_airport_loc['airport'] == airport_code]['longitude'].iloc[0]
        return (latitude, longitude)

    # Function to calculate the distance between two airport codes
    def calculate_distance(airport_code1, airport_code2, df_airport_loc):
        coords_1 = get_airport_coordinates(airport_code1, df_airport_loc)
        coords_2 = get_airport_coordinates(airport_code2, df_airport_loc)

        # Calculate the great-circle distance
        distance = geodesic(coords_1, coords_2).kilometers
        return distance

    def cal_pair_dist(df, df_airport_loc):

        ades_adep_pairs = df[['ades', 'adep']].drop_duplicates()
        ades_adep_pairs.loc[:, 'flight_dist'] = ''
        for idx, row in tqdm(ades_adep_pairs.iterrows(), total=len(ades_adep_pairs)):
            airport_code1 = row['adep']
            airport_code2 = row['ades']  # Example airport code for Los Angeles International Airport
            try:
                distance_km = calculate_distance(airport_code1, airport_code2, df_airport_loc)
                ades_adep_pairs.at[idx, 'flight_dist'] = distance_km
                ades_adep_pairs.to_csv('../ades_adep_pairs.csv', index=False)
            except ValueError as e:
                print(e)
                print(airport_code1, airport_code2)

    if os.path.exists('../airport_location.csv'):
        df_airport_loc = pd.read_csv('../airport_location.csv')
    else:
        get_airport_location(df_file_challenge, df_file_submission)
        df_airport_loc = pd.read_csv('../airport_location.csv')

    # For calculate all pairs
    df_file_c_s = pd.concat([df_file_submission, df_file_challenge], ignore_index=True)

    cal_pair_dist(df_file_c_s, df_airport_loc)




def find_average_adep_ades_airline_aircraft_flight(row, reference_df):
    potential_matches = reference_df[(reference_df['flight_id']!=row['flight_id'])&
        (reference_df['adep'] == row['adep']) &
        (reference_df['ades'] == row['ades']) &
        (reference_df['airline'] == row['airline']) &
        (reference_df['aircraft_type'] == row['aircraft_type']) &
        (reference_df['wtc'] == row['wtc'])
        ][['flight_duration', 'taxiout_time', 'flown_distance', 'highest_altitude', 'cruising_altitude',
                          'time_to_reach_altitude_mode','from_start_time_to_reach_altitude_mode',
                          'average_u_component_of_wind', 'average_v_component_of_wind', 'average_temperature',"duration"
                          ]]

    if not potential_matches.empty:
        avg_values = potential_matches.mean(skipna=True, numeric_only=True).to_frame().T
        # Drop NaNs along columns as it's now a DataFrame
        avg_values = avg_values.dropna(axis=1, how='all')

        '''
                if not potential_matches.empty:
                    potential_matches['offblock_diff'] = (
                            pd.to_datetime(potential_matches['actual_offblock_time']) - pd.to_datetime(
                        row['actual_offblock_time'])
                    ).abs()
                    closest_flight = potential_matches.sort_values('offblock_diff').iloc[0]
                    return closest_flight
                '''
        return avg_values
    else:
        return None

def find_average_airline_aircraft(row, reference_df):
    potential_matches = reference_df[(reference_df['flight_id']!=row['flight_id'])&
        (reference_df['airline'] == row['airline']) &
        (reference_df['aircraft_type'] == row['aircraft_type'])
        ][['flight_duration', 'taxiout_time', 'flown_distance', 'highest_altitude', 'cruising_altitude',
                          'time_to_reach_altitude_mode','from_start_time_to_reach_altitude_mode',
                          'average_u_component_of_wind', 'average_v_component_of_wind', 'average_temperature',"duration"
                          ]]
    if not potential_matches.empty:
        avg_values = potential_matches.mean(skipna=True, numeric_only=True).to_frame().T
        # Drop NaNs along columns as it's now a DataFrame
        avg_values = avg_values.dropna(axis=1, how='all')
        return avg_values
    else:
        return None

def find_average_aircraft(row, reference_df):
    potential_matches = reference_df[(reference_df['flight_id']!=row['flight_id'])&
        (reference_df['aircraft_type'] == row['aircraft_type'])
    ][['flight_duration', 'taxiout_time', 'flown_distance', 'highest_altitude', 'cruising_altitude',
                          'time_to_reach_altitude_mode','from_start_time_to_reach_altitude_mode',
                          'average_u_component_of_wind', 'average_v_component_of_wind', 'average_temperature',"duration"
                          ]]
    if not potential_matches.empty:
        avg_values = potential_matches.mean(skipna=True, numeric_only=True).to_frame().T
        # Drop NaNs along columns as it's now a DataFrame
        avg_values = avg_values.dropna(axis=1, how='all')
        return avg_values
    else:
        return None

def find_average_wtc(row, reference_df):
    potential_matches = reference_df[(reference_df['flight_id'] != row['flight_id']) &
                                     (reference_df['wtc'] == row['wtc'])
                                     ][
        ['flight_duration', 'taxiout_time', 'flown_distance', 'highest_altitude', 'cruising_altitude',
         'time_to_reach_altitude_mode', 'from_start_time_to_reach_altitude_mode',
         'average_u_component_of_wind', 'average_v_component_of_wind', 'average_temperature',"duration"
         ]]
    if not potential_matches.empty:
        avg_values = potential_matches.mean(skipna=True, numeric_only=True).to_frame().T
        # Drop NaNs along columns as it's now a DataFrame
        avg_values = avg_values.dropna(axis=1, how='all')
        return avg_values
    else:
        return None

def check_if_continue_filling(data):
    row_has_null_or_empty = data.isna().any(axis=1).iloc[0] or (data.iloc[0] == '').any()
    return row_has_null_or_empty


def fill_with_average_strategy(idx, row, merged_df, func_name):
    closest_flight = func_name(row, merged_df)
    if closest_flight is not None:
        for col in closest_flight.columns:
            if pd.isna(merged_df.at[idx, col]):
                merged_df.at[idx, col] = closest_flight[col].iloc[0]
    return merged_df

#
# def fill_with_closest_flight_info(idx, row, merged_df):
#     closest_flight = find_average_adep_ades_airline_aircraft_flight(row, merged_df)
#     if closest_flight is not None:
#         for col in closest_flight.columns:
#             if pd.isna(merged_df.at[idx, col]):
#                 merged_df.at[idx, col] = closest_flight[col]
#     return merged_df
#
# def fill_with_avg_airline_aircraft_flight_info(idx, row, merged_df):
#     avg_airline_aircraft_values = find_average_airline_aircraft(row, merged_df)
#     if avg_airline_aircraft_values is not None:
#         for col in avg_airline_aircraft_values.columns:
#             if pd.isna(merged_df.at[idx, col]):
#                 merged_df.at[idx, col] = avg_airline_aircraft_values[col]
#     return merged_df
#
# def fill_with_avg_aircraft_flight_info(idx, row, merged_df):
#     avg_aircraft_values = find_average_aircraft(row, merged_df)
#     if avg_aircraft_values is not None:
#         for col in avg_aircraft_values.columns:
#             if pd.isna(merged_df.at[idx, col]):
#                 merged_df.at[idx, col] = avg_aircraft_values[col]
#     return merged_df
#
# def fill_with_avg_wtc_flight_info(idx, row, merged_df):
#     avg_wtc_values = find_average_wtc(row, merged_df)
#     if avg_wtc_values is not None:
#         for col in avg_wtc_values.columns:
#             if pd.isna(merged_df.at[idx, col]):
#                 merged_df.at[idx, col] = avg_wtc_values[col]

def update_flight_info(df_need_update, df_flight_info):
    df_need_update['flight_id'] = df_need_update['flight_id'].astype(int)
    df_flight_info['flight_id'] = df_flight_info['flight_id'].astype(int)

    df_need_update['date'] =  pd.to_datetime(df_need_update['date'],errors='coerce')
    df_flight_info['date'] = pd.to_datetime(df_flight_info['date'], errors='coerce')

    # Step 1: Merging based on flight_id and date (columns shared between datasets)
    merged_df = pd.merge(df_need_update,  df_flight_info, on=['flight_id', 'date'], how='left')

    # Identifying rows where cruising_altitude and related values are missing
    missing_rows = merged_df[merged_df.isnull().any(axis=1)]
    # Apply the method to fill missing values
    # Step 2: Fill missing values based on closest flight with matching adep, ades, airline, aircraft_type, and closest actual_offblock_time
    for idx, row in tqdm(missing_rows.iterrows(), total=len(missing_rows)):
        merged_df = fill_with_average_strategy(idx, row, merged_df, find_average_adep_ades_airline_aircraft_flight)
        if check_if_continue_filling(merged_df.loc[idx,:].to_frame().T):
            # Step 3: Use average values for the same airline and aircraft_type if no close flight found
            merged_df = fill_with_average_strategy(idx, row, merged_df, find_average_airline_aircraft)
        if check_if_continue_filling(merged_df.loc[idx,:].to_frame().T):
            # Step 4: Use average values for same aircraft_type if no same airline and aircraft_type
            merged_df = fill_with_average_strategy(idx, row, merged_df, find_average_aircraft)
        if check_if_continue_filling(merged_df.loc[idx,:].to_frame().T):
            # Step 5: Use average values of same wtc if all failed
            merged_df = fill_with_average_strategy(idx, row, merged_df, find_average_wtc)
    return merged_df

if __name__ == '__main__':
    # Load the CSV files
    file1_path = '../combined_flight_info_drop_abnormal.csv'
    file2_path = '../PRCData/challenge_set.csv'
    file4_path = '../PRCData/final_submission_set.csv'

    df_file_challenge = pd.read_csv(file2_path)


    df_file_submission = pd.read_csv(file4_path)

    # Check if generated airport distance file
    # Load identified distance between airports
    file3_path = '../ades_adep_pairs.csv'
    if os.path.exists(file3_path):
        df_pair_dist = pd.read_csv(file3_path)
    else:
        get_pair_dist(df_file_challenge, df_file_submission)
        df_pair_dist = pd.read_csv(file3_path)

    # Load extracted flight information
    df_flight_info = pd.read_csv(file1_path)
    df_flight_info = df_flight_info.dropna(axis=0, how='all')

    # Add information to challenge and final submission set
    print('starting merge')
    merged_challenge = update_flight_info(df_file_challenge, df_flight_info)
    merged_challenge_all = pd.merge(merged_challenge, df_pair_dist, on=['ades', 'adep'], how='left')
    merged_challenge_all.to_csv('../challenge_set_update.csv', index=False)



    df_file_submission = df_file_submission.drop('tow', axis=1)
    merged_submission = update_flight_info(df_file_submission, df_flight_info)
    merged_submission_all = pd.merge(merged_submission, df_pair_dist, on=['ades', 'adep'], how='left')
    merged_submission['tow'] = ''
    merged_submission.to_csv('../submission_set_update.csv', index=False)

    print('finished merge')