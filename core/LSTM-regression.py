import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import time

def LSTM_regression(file_path, output_folder):
    """
    Process a single Excel file: clean data, standardize features, train models, and save the results.

    Args:
        file_path (str): Path to the input Excel file.
        output_folder (str): Path to the output folder.
    """
    # Read data
    data = pd.read_excel(file_path)

    # Extract features and target variable
    X = data
    y = data['needsales']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    prin01 = pd.DataFrame(X_test)

    features_x_train = X_train.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])
    features_X_test = X_test.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])

    labels_y_train = X_train['flag']
    labels_y_test = X_test['flag']

    # Train RandomForest model for classification
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(features_x_train, labels_y_train)

    # Evaluate the classifier model
    features_y_pred = rf_classifier.predict(features_X_test)
    accuracy = accuracy_score(labels_y_test, features_y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    X_test['flag'] = features_y_pred

    X_train = X_train.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])
    X_test = X_test.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input data for LSTM model
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(1, X_train_scaled.shape[1])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the LSTM model
    model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2,
              callbacks=[early_stopping], verbose=2)

    # Predict on the test set
    y_pred = model.predict(X_test_reshaped).flatten()

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

    # Append results to the results list
    results_list.append({'File': file_path, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2 Score': r2})

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
            LSTM_regression(file_path, output_folder)

# Define input and output folder paths
input_folder = r"input_folder"
output_folder = r"output_folder"

# Create an empty list to store evaluation metrics
results_list = []

# Record the start time
start_time = time.time()

# Process all files in the input folder
process_folder(input_folder, output_folder)

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)

# Save evaluation metrics to an Excel file
results_file = os.path.join(output_folder, 'LSTM_Evaluation_Metrics.xlsx')
results_df.to_excel(results_file, index=False)

print("Evaluation metrics saved to:", results_file)

# Record the end time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
