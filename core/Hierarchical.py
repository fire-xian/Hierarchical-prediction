import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

def Hierarchical(file_path, output_folder):
    """
    Process a single Excel file: clean data, standardize features, perform hierarchical clustering,
    and save the results to an output folder.

    Args:
        file_path (str): Path to the input Excel file.
        output_folder (str): Path to the output folder.
    """
    # Read the data
    data = pd.read_excel(file_path)

    # Drop rows with missing values
    data = data.dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[['needsales']])

    # Create dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(scaled_features, method='ward'))
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Euclidean distances')
    plt.show()

    # Determine the number of clusters from the dendrogram
    n_clusters = 3  # Adjust the number of clusters based on the dendrogram

    # Perform hierarchical clustering
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    data['flag'] = hc.fit_predict(scaled_features)

    # Display the number of samples in each cluster
    print(data['flag'].value_counts())

    # Save the clustered data to an Excel file
    output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.xlsx")
    data.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

def process_folder(input_folder, output_folder):
    """
    Process all Excel files in the input folder and save the results to the output folder.

    Args:
        input_folder (str): Path to the folder containing input Excel files.
        output_folder (str): Path to the folder to save output Excel files.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            file_path = os.path.join(input_folder, file)
            Hierarchical(file_path, output_folder)

# Define input and output folder paths
input_folder = r"input_folder"
output_folder = r"output_folder"

# Process all files in the input folder
process_folder(input_folder, output_folder)
