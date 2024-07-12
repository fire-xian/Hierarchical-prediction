import os
import pandas as pd

def fill_missing_values(source_folder, target_folder):
    """
    Fill missing values in each Excel file in the source folder using rolling window mean and save to the target folder.

    Args:
        source_folder (str): Path to the source folder containing input Excel files.
        target_folder (str): Path to the target folder to save output Excel files.
    """
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all xlsx files in the source folder
    all_files = os.listdir(source_folder)
    xlsx_files = [file for file in all_files if file.endswith('.xlsx')]

    # Process each xlsx file
    for file in xlsx_files:
        source_file_path = os.path.join(source_folder, file)
        target_file_path = os.path.join(target_folder, file)

        # Read the xlsx file
        df = pd.read_excel(source_file_path)

        # Convert the first column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Generate complete date range
        date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')

        # Set date column as index
        df.set_index('date', inplace=True)

        # Reindex the DataFrame, only filling missing values
        df = df.reindex(date_range)

        # Rename columns to original names
        df.columns = ['sales', 'prices', 'wholesale_price']

        # Initialize the DataFrame after the last filling
        last_filled_df = df.copy()

        # Fill missing values using rolling window mean
        while True:
            filled_df = last_filled_df.fillna(last_filled_df.rolling(window=7, min_periods=1).mean())

            # Check if the filled DataFrame is the same as the last filled DataFrame
            if filled_df.equals(last_filled_df):
                break

            # Update the last filled DataFrame
            last_filled_df = filled_df

        # Rename index column back to "date"
        filled_df.index.name = 'date'

        # Save the filled DataFrame to the target folder
        filled_df.to_excel(target_file_path, index=True)

        print(f"Filled data with rolling window method saved to {target_file_path}")

    print("Processing completed!")

# Define source and target folder paths
source_folder = r"source_folder"
target_folder = r"target_folder"

# Fill missing values in Excel files in the source folder and save results to the target folder
fill_missing_values(source_folder, target_folder)
