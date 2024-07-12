import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
import time

def train_rf_classifier(X_train, X_test, y_train, y_test, file):
    """
    Train a Random Forest classifier and evaluate accuracy.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training labels.
        y_test (Series): Testing labels.
        file (str): File name for logging.

    Returns:
        float: Accuracy of the classifier on the test set.
    """
    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Log predictions to DataFrame
    X_test['flag'] = y_pred

    return accuracy

def train_xgb_regressor(X_train, X_test, y_train, y_test, prin01, file):
    """
    Train an XGBoost regressor, make predictions, and evaluate performance metrics.

    Args:
        X_train (DataFrame): Training features.
        X_test (DataFrame): Testing features.
        y_train (Series): Training target.
        y_test (Series): Testing target.
        prin01 (DataFrame): DataFrame for logging predictions.
        file (str): File name for logging.

    Returns:
        DataFrame: Results DataFrame containing evaluation metrics.
    """
    # Train XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=100)
    xgb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regressor.predict(X_test)

    # Calculate combined prediction
    y_pred_result = y_pred * 0.5 + prin01['lastsales']
    y_pred_real = prin01['sales']

    # Calculate evaluation metrics
    mse = mean_squared_error(y_pred_real, y_pred_result)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_pred_real, y_pred_result)
    r2 = r2_score(y_pred_real, y_pred_result)
    mape = np.mean(np.abs((y_pred_real - y_pred_result) / y_pred_real)) * 100

    # Print evaluation metrics
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("R-squared (R^2):", r2)

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='True Sales', color='blue', marker='o')
    plt.plot(y_pred, label='Predicted Sales', color='red', marker='x')
    plt.title('True vs Predicted Sales')
    plt.xlabel('Sample')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Log evaluation metrics to results list
    return {'文件': file, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2 Score': r2}

def process_files(input_folder, output_folder):
    """
    Process each Excel file in the input folder, train models, and save evaluation metrics.

    Args:
        input_folder (str): Path to the folder containing input Excel files.
        output_folder (str): Path to the folder to save output Excel files.
    """
    results_list = []
    accuracy_list = []
    start_time = time.time()

    # Iterate through files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            # Read data from Excel file
            data = pd.read_excel(os.path.join(input_folder, file))

            # Extract features and target variable
            X = data
            y = data['needsales']

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

            prin01 = pd.DataFrame(X_test)

            # Train and evaluate Random Forest classifier
            accuracy = train_rf_classifier(X_train.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale']),
                                           X_test.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale']),
                                           X_train['flag'], X_test['flag'], file)
            accuracy_list.append({'文件': file, 'accuracy': accuracy})

            # Train and evaluate XGBoost regressor
            results = train_xgb_regressor(X_train.drop(columns=['sales', 'prices', 'needsales', 'wholesale']),
                                          X_test.drop(columns=['sales', 'prices', 'needsales', 'wholesale']),
                                          y_train, y_test, prin01, file)
            results_list.append(results)

    # Save evaluation metrics to Excel files
    results_df = pd.DataFrame(results_list)
    accuracy_df = pd.DataFrame(accuracy_list)
    results_file = os.path.join(output_folder, 'RF_Test_Results.xlsx')
    accuracy_file = os.path.join(output_folder, 'RF_Accuracy.xlsx')
    results_df.to_excel(results_file, index=False)
    accuracy_df.to_excel(accuracy_file, index=False)

    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    print("Evaluation metrics saved to:", results_file)
    print("Accuracy metrics saved to:", accuracy_file)

# Define input and output folder paths
input_folder = r"input_folder"
output_folder = r"output_folder"

# Process files in input folder, train models, and save results
process_files(input_folder, output_folder)
