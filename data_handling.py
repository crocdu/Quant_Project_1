# Jing
import os
import re
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from data.client import get_client

# Set up API keys (if needed)
os.environ['X_KEY'] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ['X_SECRET'] = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
cli = get_client()


# Function to read Parquet files
def read_parquet(file_path):
    try:
        df = pd.read_parquet(file_path)
        print(f"Contents of {file_path}:")
        print(df.head())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")


# Function to convert Parquet to CSV
def convert_parquet_to_csv(file_path, output_path):
    df = pd.read_parquet(file_path)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to: {output_path}")


# Function to merge option tick Parquet files
def merge_option_tick_files(input_path, output_csv):
    files = glob(os.path.join(input_path, "*.parquet"))
    dataframes = [pd.read_parquet(file) for file in files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged option ticks saved to {output_csv}")


# Function to filter and download option tick data
def download_filtered_option_tick(contract_letter, contract_number, save_path, start_time, end_time):
    try:
        folder_path = f'option_tick/{contract_letter}/{contract_letter}'
        df = cli.read(folder_path, start_time=start_time, end_time=end_time,
                      columns=['TradeDay', 'Code', 'BidiPrice', 'AskiPrice', 'date'])
        df = df.reset_index()

        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        df_filtered = df[(df['date'].dt.time >= pd.to_datetime("14:56:00").time()) &
                         (df['date'].dt.time <= pd.to_datetime("14:57:00").time())]

        if df_filtered.empty:
            print(f"No matching times in the range {start_time} to {end_time} for {contract_letter}. Skipping.")
            return

        df_contract = df_filtered[df_filtered['Code'].str.contains(contract_number, na=False)]
        if df_contract.empty:
            print(f"No matching contract numbers for {contract_number} on {start_time}. Skipping.")
            return

        save_file = f"{save_path}/{contract_number}_{start_time[:10]}.parquet"
        os.makedirs(save_path, exist_ok=True)
        df_contract.to_parquet(save_file, index=False)
        print(f"Filtered option tick data for {contract_number} saved to {save_file}.")
    except Exception as e:
        print(f"Error downloading filtered option tick data for {contract_letter}-{contract_number}: {e}")


# Function to process contract mapping per date
def process_single_date(contract_mapping, current_date, save_path):
    start_time = f'{current_date.strftime("%Y-%m-%d")}T14:56:00.000000000'
    end_time = f'{current_date.strftime("%Y-%m-%d")}T14:57:00.000000000'
    tasks = []
    for contract_letter, contract_numbers in contract_mapping.items():
        for contract_number in contract_numbers:
            tasks.append((contract_letter, contract_number, save_path, start_time, end_time))
    return tasks


# Parallelized execution of downloads
def main_parallelized(contract_mapping, save_path, start_date, end_date):
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    tasks = []
    while current_date <= end_date:
        tasks.extend(process_single_date(contract_mapping, current_date, save_path))
        current_date += timedelta(days=1)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_filtered_option_tick, *task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    print("All tasks completed successfully.")


# Rank options by maturity
def rank_options_by_maturity(option_rank_df, rank_by_maturity=True):
    if rank_by_maturity:
        option_rank_df.sort_values(['date', 'exercise_date'], ascending=True, inplace=True)
        for group in option_rank_df['group_tag'].unique():
            for date in option_rank_df['date'].unique():
                group_date_filter = (option_rank_df['group_tag'] == group) & (option_rank_df['date'] == date)
                option_rank_df.loc[group_date_filter, 'mapping'] = range(1, len(option_rank_df[group_date_filter]) + 1)
    return option_rank_df


# Load contract mapping from Excel
def load_contract_mapping(option_rank_path):
    option_rank_df = pd.read_excel(option_rank_path)
    option_rank_df['group_tag'] = option_rank_df['group_tag'].str.upper()
    option_rank_df['underlying_code'] = option_rank_df['underlying_code'].str.split('.').str[0].str.upper()
    return option_rank_df.groupby("group_tag")["underlying_code"].apply(list).to_dict()


# Main execution function
def main():
    try:
        option_rank_path = "C:/Users/jing.yu/Desktop/option_rank_i.xlsx"
        save_path = "i_data/i_options_tick"
        start_date = "2022-11-29"
        end_date = "2024-11-29"

        contract_mapping = load_contract_mapping(option_rank_path)
        main_parallelized(contract_mapping, save_path, start_date, end_date)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()