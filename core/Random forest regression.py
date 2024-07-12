import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os

def random_forest_regression(file_path, output_folder):
    """
    Process a single Excel file: compute lagged features, train RandomForest model,
    calculate residuals, and save the results.

    Args:
        file_path (str): Path to the input Excel file.
        output_folder (str): Path to the output folder.
    """
    # Read data
    data = pd.read_excel(file_path)

    # Compute lagged features
    for i in range(1, 5):
        data[f'lagged_sales_{i}'] = data['sales'].shift(i)
        data[f'lagged_prices_{i}'] = data['prices'].shift(i)

    # Drop rows with NaN values
    data = data.dropna()

    # Extract features and target variable
    X = data.drop(columns=['sales', 'prices', 'wholesale'])
    y = data['sales']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    prin01 = pd.DataFrame(X_test)
    sales = y_test.copy()

    # Train RandomForest model
    rf_regressor = RandomForestRegressor(random_state=42)
    rf_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_regressor.predict(X_test)

    # Calculate residuals
    prin01['needsales'] = sales - y_pred
    prin01['lastsales'] = y_pred

    # Save results to Excel file
    results_name = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}.xlsx")
    prin01.to_excel(results_name, index=False)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R^2 Score:", r2)

    # Append results to the results list
    results_list.append({'File': os.path.basename(file_path), 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2 Score': r2})

    # Visualize true vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Sales', color='blue', marker='o')
    plt.plot(y_pred, label='Predicted Sales', color='red', marker='x')
    plt.title('True vs Predicted Sales')
    plt.xlabel('Sample')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

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
            random_forest_regression(file_path, output_folder)

# Define input and output folder paths
input_folder = r"input_folder"
output_folder = r"output_folder"

# Create an empty list to store evaluation metrics
results_list = []

# Process all files in the input folder
process_folder(input_folder, output_folder)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Save evaluation metrics to an Excel file
results_file = os.path.join(output_folder, 'RF_Evaluation_Metrics.xlsx')
results_df.to_excel(results_file, index=False)

print("Evaluation metrics saved to:", results_file)
