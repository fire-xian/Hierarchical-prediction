import os
import pandas as pd


def fill_missing_dates(source_folder, target_folder):
    """
    Fill missing dates in Excel files from the source folder and save the updated files to the target folder.

    Args:
        source_folder (str): Path to the source folder containing the Excel files.
        target_folder (str): Path to the target folder to save the updated files.
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Get all xlsx files from the source folder
    all_files = os.listdir(source_folder)
    xlsx_files = [file for file in all_files if file.endswith('.xlsx')]

    # Process each xlsx file
    for file in xlsx_files:
        file_path = os.path.join(source_folder, file)
        target_file_path = os.path.join(target_folder, file)

        # Read the xlsx file
        df = pd.read_excel(file_path)

        # Convert the first column to datetime format
        df['销售日期'] = pd.to_datetime(df['销售日期'])

        # Generate a complete date range
        date_range = pd.date_range(start=df['销售日期'].min(), end=df['销售日期'].max(), freq='D')

        # Reindex the DataFrame, filling in missing dates and setting missing values to 0
        df = df.set_index('销售日期').reindex(date_range).fillna(0).reset_index()

        # Rename columns appropriately
        df.columns = ['date', 'sales', 'prices', 'wholesale_price']

        # Save the updated DataFrame to the target folder
        df.to_excel(target_file_path, index=False)

        print("Data filled and saved to", target_file_path)

    print("Processing completed!")


# Define source and target folder paths
source_folder = r"F:\Seconddata\splitdata_2"
target_folder = r"F:\Seconddata\splitdata_3"

# Call the function to fill missing dates and save files
fill_missing_dates(source_folder, target_folder)
