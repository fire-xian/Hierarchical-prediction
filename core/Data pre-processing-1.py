import os
import pandas as pd

# Read the original Excel file
df = pd.read_excel(r"Dataset.xlsx")

# Group by category name and product name
grouped = df.groupby(['分类名称', '单品名称'])

# Define the output folder path
output_folder = r"F:\Seconddata\splitdata_1"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Save each group as a separate Excel file
for (category_name, product_name), group_df in grouped:
    # Construct the file name
    file_name = f"{category_name}_{product_name}.xlsx"
    # Construct the file path
    file_path = os.path.join(output_folder, file_name)
    # Save the grouped data to an Excel file
    group_df.to_excel(file_path, index=False)

print("Splitting completed!")
