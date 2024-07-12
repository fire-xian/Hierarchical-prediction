import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import time
from sklearn.ensemble import RandomForestClassifier


# 设置文件夹路径
input_folder = r"F:\Thirddata\01筛选数据\03添加标签"
output_folder = r"F:\Thirddata\01筛选数据"

# 创建一个空的列表来存储评价指标数据
results_list = []

# 记录开始时间
start_time = time.time()


# 遍历文件夹中的所有文件
for file in os.listdir(input_folder):
    if file.endswith('.xlsx'):
        # 读取数据
        data = pd.read_excel(os.path.join(input_folder, file))

        # 提取特征和目标变量
        X = data
        y = data['needsales']  # 目标变量

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        prin01 = pd.DataFrame(X_test)

        features_x_train = X_train.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])  # 特征
        features_X_test = X_test.drop(columns=['sales', 'prices', 'flag', 'needsales', 'wholesale'])  # 特征

        labels_y_train = X_train['flag']  # 标签
        labels_y_test = X_test['flag']  # 标签

        # 步骤5: 训练随机森林模型

        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(features_x_train, labels_y_train)

        # 步骤6: 评估模型
        features_y_pred = rf_classifier.predict(features_X_test)
        accuracy = accuracy_score(labels_y_test, features_y_pred)
        print(f"模型准确率: {accuracy:.2f}")

        X_test['flag'] = features_y_pred

        X_train = X_train.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])  # 特征
        X_test = X_test.drop(columns=['sales', 'prices', 'needsales', 'wholesale'])  # 特征

        # 标准化数据
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建BP神经网络模型
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(), loss='mse')

        # 训练模型
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

        # 在测试集上进行预测
        y_pred = model.predict(X_test_scaled).flatten()

        y_pred_result = y_pred * 0.5 + prin01['lastsales']
        y_pred_real = prin01['sales']

        # 计算评价指标
        mse = mean_squared_error(y_pred_real, y_pred_result)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_pred_real, y_pred_result)
        r2 = r2_score(y_pred_real, y_pred_result)
        mape = np.mean(np.abs((y_pred_real - y_pred_result) / y_pred_real)) * 100

        print("均方误差（MSE）:", mse)
        print("均方根误差（RMSE）:", rmse)
        print("平均绝对误差（MAE）:", mae)
        print("平均绝对百分比误差（MAPE）:", mape)
        print("决定系数（R^2）:", r2)

        # 将评价指标数据存储到列表中
        results_list.append({'文件': file, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R^2 Score': r2})

# 将列表转换为 DataFrame
results_df = pd.DataFrame(results_list)

# 保存评价指标数据到 Excel 文件
results_file = os.path.join(output_folder, 'BPNN_评价指标.xlsx')
results_df.to_excel(results_file, index=False)

print("评价指标数据已保存到:", results_file)

# 记录结束时间
end_time = time.time()

# 计算运行时间
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
