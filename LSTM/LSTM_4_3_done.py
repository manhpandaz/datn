import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Đọc dữ liệu từ nguồn
df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')

# Chọn các trường dữ liệu cần thiết
selected_columns = ['DATE', 'USD_W', 'DT_W', 'V_W']
df_selected = df[selected_columns]

# Chuyển đổi cột 'DATE' thành kiểu dữ liệu datetime
df_selected['DATE'] = pd.to_datetime(df_selected['DATE'])

# Đặt 'DATE' làm chỉ số của DataFrame
df_selected.set_index('DATE', inplace=True)

# Chuẩn hóa dữ liệu sử dụng Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_selected)

# Chia thành tập huấn luyện và tập kiểm tra (tỷ lệ 80-20)
train_size = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

# Chuẩn Bị Dữ Liệu cho LSTM


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10  # Số lượng bước thời gian quan sát trước đó
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Xây dựng mô hình LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dense(units=3))  # 3 units cho 3 trường dữ liệu
model_lstm.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình LSTM
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32,
               validation_data=(X_test, y_test), shuffle=False)

# Đánh giá mô hình LSTM trên tập kiểm tra
mse_lstm = model_lstm.evaluate(X_test, y_test)
print(f"Mean Squared Error (LSTM) on Test Data: {mse_lstm}")

# Dự báo trên tập kiểm tra cho LSTM
y_pred_lstm = model_lstm.predict(X_test)

# Xây dựng mô hình Decision Tree
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

# Dự báo trên tập kiểm tra cho Decision Tree
y_pred_dt = model_dt.predict(X_test.reshape((X_test.shape[0], -1)))

# Tính Mean Squared Error cho Decision Tree
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Mean Squared Error (Decision Tree) on Test Data: {mse_dt}")

# Xây dựng mô hình Random Forest
model_rf = RandomForestRegressor()
model_rf.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

# Dự báo trên tập kiểm tra cho Random Forest
y_pred_rf = model_rf.predict(X_test.reshape((X_test.shape[0], -1)))

# Tính Mean Squared Error cho Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (Random Forest) on Test Data: {mse_rf}")

# Trực quan hóa kết quả cho cột USD_W của LSTM
y_test_inverse_lstm = scaler.inverse_transform(y_test)
y_pred_inverse_lstm = scaler.inverse_transform(y_pred_lstm)

time_steps_test = df_selected.index[train_size + time_steps:]

plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 0], label='Actual (USD_W)', marker='o')
plt.plot(time_steps_test,
         y_pred_inverse_lstm[:, 0], label='Predicted (USD_W)', marker='o')
plt.title('USD_W - Actual vs. Predicted (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột DT_W của LSTM
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 1], label='Actual (DT_W)', marker='o')
plt.plot(time_steps_test,
         y_pred_inverse_lstm[:, 1], label='Predicted (DT_W)', marker='o')
plt.title('DT_W - Actual vs. Predicted (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột V_W của LSTM
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 2], label='Actual (V_W)', marker='o')
plt.plot(time_steps_test,
         y_pred_inverse_lstm[:, 2], label='Predicted (V_W)', marker='o')
plt.title('V_W - Actual vs. Predicted (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột USD_W của Decision Tree
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 0], label='Actual (USD_W)', marker='o')
plt.plot(time_steps_test, y_pred_dt[:, 0],
         label='Predicted (USD_W)', marker='o')
plt.title('USD_W - Actual vs. Predicted (Decision Tree)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột DT_W của Decision Tree
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 1], label='Actual (DT_W)', marker='o')
plt.plot(time_steps_test, y_pred_dt[:, 1],
         label='Predicted (DT_W)', marker='o')
plt.title('DT_W - Actual vs. Predicted (Decision Tree)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột V_W của Decision Tree
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 2], label='Actual (V_W)', marker='o')
plt.plot(time_steps_test, y_pred_dt[:, 2], label='Predicted (V_W)', marker='o')
plt.title('V_W - Actual vs. Predicted (Decision Tree)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột USD_W của Random Forest
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 0], label='Actual (USD_W)', marker='o')
plt.plot(time_steps_test, y_pred_rf[:, 0],
         label='Predicted (USD_W)', marker='o')
plt.title('USD_W - Actual vs. Predicted (Random Forest)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột DT_W của Random Forest
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 1], label='Actual (DT_W)', marker='o')
plt.plot(time_steps_test, y_pred_rf[:, 1],
         label='Predicted (DT_W)', marker='o')
plt.title('DT_W - Actual vs. Predicted (Random Forest)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột V_W của Random Forest
plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 2], label='Actual (V_W)', marker='o')
plt.plot(time_steps_test, y_pred_rf[:, 2], label='Predicted (V_W)', marker='o')
plt.title('V_W - Actual vs. Predicted (Random Forest)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()
