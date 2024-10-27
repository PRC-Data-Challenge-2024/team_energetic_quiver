import warnings
import os
from traffic.core import Traffic, Flight
import pandas as pd
import time
from tqdm import tqdm
def smooth_traj(parquet_file):
    warning_messages = []
    # Catch UserWarnings and exit the program if they occur
    with warnings.catch_warnings(record=True) as w:
        # Make sure all warnings are turned into recorded warnings
        warnings.simplefilter("always", UserWarning)
        try:
            # Process the Traffic data
            t = (Traffic.from_file(parquet_file)
                 # smooth vertical glitches
                 .filter()
                 # resample at 1s
                 .resample('1s')
                 # execute all
                 .eval()
                 )

            # Check if any UserWarnings were raised
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    warning_messages.append(str(warning.message))

            if len(warning_messages) > 0:
                uni_warning_messages =  list(set(warning_messages))
                store_error_flight = parquet_file.split('/')[-1].split('.')[0]
                lst_store_error_flight = [store_error_flight] * len(uni_warning_messages)
                df_store_error = pd.DataFrame({'flight_id': lst_store_error_flight, 'warnings': uni_warning_messages})
            else:
                df_store_error = None
        except Exception as e:
            # If any exception occurs, print it
            print(f"An error occurred: {e}")
            exit(1)


    return t, df_store_error

def main():
    # Define the directory containing all parquet files
    file_directory = '../PRCData/'

    # Raw file path
    raw_file_path = '../PRCData_raw/'
    # Filtered each flight path
    filtered_file_path = '../PRCData_filtered/'
    # Filtered each day path
    filtered_daily_file_path = '../PRCData_daily_filtered/'

    # List all parquet files in the directory
    parquet_files = [os.path.join(file_directory, f) for f in os.listdir(file_directory) if
                     f.endswith('.parquet')]

    COUNT = 0
    df_store_error_all = pd.DataFrame()
    df_record = pd.DataFrame(columns=['file_name'])

    # Iterate over each parquet file
    for file in parquet_files:
        file_name = file.split('/')[-1].split('.')[0]
        COUNT = COUNT + 1
        print('Number: {}, Filename: {}'.format(COUNT, file_name))
        df_reord_i = pd.DataFrame({'file_name':[file_name]})


        df_each_date = pd.read_parquet(file, engine='pyarrow')
        # Iterate over each flight file
        flight_id_lst = df_each_date['flight_id'].unique()

        for flight_id in tqdm(flight_id_lst):
            # Select specific flight
            filtered_df = df_each_date[df_each_date['flight_id'] == flight_id]
            # Save the specific flight data to a new parquet file
            output_raw_file_path = raw_file_path + str(file_name) + '_' + str(flight_id) + '.parquet'
            filtered_df.to_parquet(output_raw_file_path, engine='pyarrow')

            # Smooth the specific flight and store if UserWarning happens
            filtered_flight_file, df_store_error_i = smooth_traj(output_raw_file_path)
            output_filtered_file_path = filtered_file_path + str(file_name) + '_' + str(flight_id) + '.parquet'

            # Store the filtered flight
            filtered_flight_file.to_parquet(output_filtered_file_path, index=False)

            # Delete the mid file
            os.remove(output_raw_file_path)

            # Append the error dataframe
            df_store_error_all = pd.concat([df_store_error_all, df_store_error_i], ignore_index=True)

        # Combine all filtered into a single dataframe
        parquet_files = [f for f in os.listdir(filtered_file_path) if f.endswith('.parquet')]

        # List to store individual dataframes
        dataframes = []

        # Read and combine all parquet files
        for parquet_file in parquet_files:
            file_path = os.path.join(filtered_file_path, parquet_file)
            df = pd.read_parquet(file_path)
            dataframes.append(df)

        # Combine all dataframes into a single dataframe
        combined_df = pd.concat(dataframes, ignore_index=True)

        output_file = filtered_daily_file_path + str(file_name) + '.parquet'
        # Save the combined dataframe to a new parquet file
        combined_df.to_parquet(output_file, index=False)

        # Delete the old parquet files
        for parquet_file in parquet_files:
            file_path = os.path.join(filtered_file_path, parquet_file)
            try:
                os.remove(file_path)
            except PermissionError:
                print('PermissionError for {}'.format(file_path))
                time.sleep(5)
                os.remove(file_path)


        df_record = pd.concat([df_record, df_reord_i], ignore_index=True)
        df_record.to_csv('../Recorded_completed_files.csv', index=False)
        df_store_error_all.to_csv('../Filtered_process_error.csv', index=False)

if __name__ == '__main__':
    main()