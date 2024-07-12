import os
import pandas as pd

def pre_processing_2(source_folder, target_folder):
    """
    Process Excel files from the source folder and save the processed files to the target folder.

    Args:
        source_folder (str): Path to the source folder containing the Excel files.
        target_folder (str): Path to the target folder to save the processed files.
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Get all xlsx files from the source folder
    all_files = os.listdir(source_folder)
    xlsx_files = [file for file in all_files if file.endswith('.xlsx')]

    # Process each xlsx file
    for file in xlsx_files:
        file_path = os.path.join(source_folder, file)
        # Read the xlsx file
        df = pd.read_excel(file_path)

        # Group by sales date, sum the sales volume, and calculate the mean of sales price and wholesale price
        grouped_df = df.groupby('销售日期').agg({
            '销量(千克)': 'sum',
            '销售单价(元/千克)': 'mean',
            '批发价格(元/千克)': 'mean'
        }).reset_index()

        # Construct the output file path
        output_file = os.path.join(target_folder, f"processed_{file}")

        # Save the processed data to the target folder
        grouped_df.to_excel(output_file, index=False)

    print("Processing completed!")

# Define source and target folder paths
source_folder = r"source_folder"
target_folder = r"target_folder"

# Call the function to process files
pre_processing_2(source_folder, target_folder)
