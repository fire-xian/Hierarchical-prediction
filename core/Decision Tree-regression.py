import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import time

def decision_regression(input_folder, output_folder):
    """
    Process Excel files in the input folder, train models, evaluate them, and save results.

    Args:
        input_folder (str): Path to the input folder containing Excel files.
        output_folder (str): Path to the output folder to save the results.
    """
    # Create lists to store evaluation metrics and accuracy
    results_list = []
    accuracy_list = []

    # Record the start time
    start_time = time.time()

    # Iterate through all files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith('.xlsx'):
            # Read the data
            data = pd.read_excel(os.path.join(input_folder, file))

            # Extract features and target variable
            X = data
            y = data['needsales']  # Target variable

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

            prin01 = pd.DataFrame(X_test)

            # Prepare training and testing features and labels for the classifier
            features_x_train = X_train.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])  # Features
            features_X_test = X_test.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])  # Features

            labels_y_train = X_train['flag']  # Labels
            labels_y_test = X_test['flag']  # Labels

            # Train a Random Forest classifier
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(features_x_train, labels_y_train)

            # Evaluate the classifier
            features_y_pred = rf_classifier.predict(features_X_test)
            accuracy = accuracy_score(labels_y_test, features_y_pred)
            accuracy_list.append({'File': file, 'Accuracy': accuracy})

            print(f"Model Accuracy: {accuracy:.2f}")

            X_test['flag'] = features_y_pred

            # Prepare training and testing features for the regressor
            X_train = X_train.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])  # Features
            X_test = X_test.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])  # Features

            # Train a Decision Tree Regressor
            xgb_regressor = DecisionTreeRegressor(random_state=42)
            xgb_regressor.fit(X_train, y_train)

            # Predict on the test set
            y_pred = xgb_regressor.predict(X_test)

            # Combine predictions from the first model with the new predictions
            y_pred_result = y_pred * 0.5 + prin01['lastsales']
            y_pred_real = prin01['sales']

            # Calculate evaluation metrics
            mse = mean_squared_error(y_pred_real, y_pred_result)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_pred_real, y_pred_result)
            r2 = r2_score(y_pred_real, y_pred_result)
            mape = np.mean(np.abs((y_pred_real - y_pred_result) / y_pred_real)) * 100

            print("Mean Squared Error (MSE):", mse)
            print("Root Mean Squared Error (RMSE):", rmse)
            print("Mean Absolute Error (MAE):", mae)
            print("Mean Absolute Percentage Error (MAPE):", mape)
            print("R^2 Score:", r2)

            # Visualize the prediction results
            plt.figure(figsize=(10, 6))
            plt.plot(y_test.values, label='True Sales', color='blue', marker='o')
            plt.plot(y_pred, label='Predicted Sales', color='red', marker='x')
            plt.title('True vs Predicted Sales')
            plt.xlabel('Sample')
            plt.ylabel('Sales')
            plt.legend()
            plt.show()

            # Store evaluation metrics
            results_list.append({'File': file, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2 Score': r2})

    # Convert lists to DataFrames
    results_df = pd.DataFrame(results_list)
    accuracy_df = pd.DataFrame(accuracy_list)

    # Save evaluation metrics to Excel files
    results_file = os.path.join(output_folder, 'Decision_Tree_Results.xlsx')
    accuracy_file = os.path.join(output_folder, 'Accuracy.xlsx')
    results_df.to_excel(results_file, index=False)
    accuracy_df.to_excel(accuracy_file, index=False)

    print("Evaluation metrics saved to:", results_file)

    # Record the end time
    end_time = time.time()

    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

# Define input and output folder paths
input_folder = r"input_folder"
output_folder = r"output_folder"

# Call the function to process files
decision_regression(input_folder, output_folder)
