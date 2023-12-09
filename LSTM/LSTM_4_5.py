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

print("Một vài dữ liệu đầu tiên:")
print(df.head())

print(f"Tổng số lượng dữ liệu: {len(df)}")
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
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

# Chuẩn bị dữ liệu cho LSTM


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Tách dữ liệu cho từng cột
y_train_usd, y_train_dt, y_train_v = y_train[:,
                                             0], y_train[:, 1], y_train[:, 2]
y_test_usd, y_test_dt, y_test_v = y_test[:, 0], y_test[:, 1], y_test[:, 2]

# Hàm xây dựng mô hình LSTM cho từng cột dữ liệu


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')
    return model


# Xây dựng mô hình LSTM cho từng cột dữ liệu
model_lstm_usd = build_lstm_model((X_train.shape[1], X_train.shape[2]))
model_lstm_dt = build_lstm_model((X_train.shape[1], X_train.shape[2]))
model_lstm_v = build_lstm_model((X_train.shape[1], X_train.shape[2]))

# Huấn luyện mô hình LSTM cho từng cột dữ liệu
model_lstm_usd.fit(X_train, y_train_usd, epochs=50, batch_size=32,
                   validation_data=(X_test, y_test_usd), shuffle=False)

model_lstm_dt.fit(X_train, y_train_dt, epochs=50, batch_size=32,
                  validation_data=(X_test, y_test_dt), shuffle=False)

model_lstm_v.fit(X_train, y_train_v, epochs=50, batch_size=32,
                 validation_data=(X_test, y_test_v), shuffle=False)

# Hàm xây dựng mô hình Decision Tree cho từng cột dữ liệu


def build_decision_tree_model():
    return DecisionTreeRegressor()

# Hàm xây dựng mô hình Random Forest cho từng cột dữ liệu


def build_random_forest_model():
    return RandomForestRegressor()


# Xây dựng mô hình Decision Tree và Random Forest cho từng cột dữ liệu
model_dt_usd = build_decision_tree_model()
model_dt_dt = build_decision_tree_model()
model_dt_v = build_decision_tree_model()

model_rf_usd = build_random_forest_model()
model_rf_dt = build_random_forest_model()
model_rf_v = build_random_forest_model()

# Huấn luyện mô hình Decision Tree cho từng cột dữ liệu
model_dt_usd.fit(X_train.reshape((X_train.shape[0], -1)), y_train_usd)
model_dt_dt.fit(X_train.reshape((X_train.shape[0], -1)), y_train_dt)
model_dt_v.fit(X_train.reshape((X_train.shape[0], -1)), y_train_v)

# Huấn luyện mô hình Random Forest cho từng cột dữ liệu
model_rf_usd.fit(X_train.reshape((X_train.shape[0], -1)), y_train_usd)
model_rf_dt.fit(X_train.reshape((X_train.shape[0], -1)), y_train_dt)
model_rf_v.fit(X_train.reshape((X_train.shape[0], -1)), y_train_v)

# Dự đoán trên tập kiểm tra cho từng mô hình
y_pred_lstm_usd = model_lstm_usd.predict(X_test)
y_pred_lstm_dt = model_lstm_dt.predict(X_test)
y_pred_lstm_v = model_lstm_v.predict(X_test)

y_pred_dt_usd = model_dt_usd.predict(X_test.reshape((X_test.shape[0], -1)))
y_pred_dt_dt = model_dt_dt.predict(X_test.reshape((X_test.shape[0], -1)))
y_pred_dt_v = model_dt_v.predict(X_test.reshape((X_test.shape[0], -1)))

y_pred_rf_usd = model_rf_usd.predict(X_test.reshape((X_test.shape[0], -1)))
y_pred_rf_dt = model_rf_dt.predict(X_test.reshape((X_test.shape[0], -1)))
y_pred_rf_v = model_rf_v.predict(X_test.reshape((X_test.shape[0], -1)))

# Đánh giá mô hình LSTM trên tập kiểm tra
mse_lstm_usd = mean_squared_error(y_test_usd, y_pred_lstm_usd)
mse_lstm_dt = mean_squared_error(y_test_dt, y_pred_lstm_dt)
mse_lstm_v = mean_squared_error(y_test_v, y_pred_lstm_v)

# Đánh giá mô hình Decision Tree trên tập kiểm tra
mse_dt_usd = mean_squared_error(y_test_usd, y_pred_dt_usd)
mse_dt_dt = mean_squared_error(y_test_dt, y_pred_dt_dt)
mse_dt_v = mean_squared_error(y_test_v, y_pred_dt_v)

# Đánh giá mô hình Random Forest trên tập kiểm tra
mse_rf_usd = mean_squared_error(y_test_usd, y_pred_rf_usd)
mse_rf_dt = mean_squared_error(y_test_dt, y_pred_rf_dt)
mse_rf_v = mean_squared_error(y_test_v, y_pred_rf_v)

# In kết quả
print(f"Mean Squared Error (LSTM USD_W) on Test Data: {mse_lstm_usd}")
print(f"Mean Squared Error (Decision Tree USD_W) on Test Data: {mse_dt_usd}")
print(f"Mean Squared Error (Random Forest USD_W) on Test Data: {mse_rf_usd}")

print(f"Mean Squared Error (LSTM DT_W) on Test Data: {mse_lstm_dt}")
print(f"Mean Squared Error (Decision Tree DT_W) on Test Data: {mse_dt_dt}")
print(f"Mean Squared Error (Random Forest DT_W) on Test Data: {mse_rf_dt}")

print(f"Mean Squared Error (LSTM V_W) on Test Data: {mse_lstm_v}")
print(f"Mean Squared Error (Decision Tree V_W) on Test Data: {mse_dt_v}")
print(f"Mean Squared Error (Random Forest V_W) on Test Data: {mse_rf_v}")

# Trực quan hóa kết quả cho cột USD_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test_usd):],
         scaler.inverse_transform(y_test_usd.reshape(-1, 1)), label='Actual (USD_W)', marker='o')
plt.plot(df_selected.index[-len(y_test_usd):],
         scaler.inverse_transform(y_pred_lstm_usd.reshape(-1, 1)), label='Predicted (USD_W) LSTM', marker='o')
plt.plot(df_selected.index[-len(y_test_usd):],
         scaler.inverse_transform(y_pred_dt_usd.reshape(-1, 1)), label='Predicted (USD_W) Decision Tree', marker='o')
plt.plot(df_selected.index[-len(y_test_usd):],
         scaler.inverse_transform(y_pred_rf_usd.reshape(-1, 1)), label='Predicted (USD_W) Random Forest', marker='o')
plt.title('USD_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột DT_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test_dt):],
         scaler.inverse_transform(y_test_dt.reshape(-1, 1)), label='Actual (DT_W)', marker='o')
plt.plot(df_selected.index[-len(y_test_dt):],
         scaler.inverse_transform(y_pred_lstm_dt.reshape(-1, 1)), label='Predicted (DT_W) LSTM', marker='o')
plt.plot(df_selected.index[-len(y_test_dt):],
         scaler.inverse_transform(y_pred_dt_dt.reshape(-1, 1)), label='Predicted (DT_W) Decision Tree', marker='o')
plt.plot(df_selected.index[-len(y_test_dt):],
         scaler.inverse_transform(y_pred_rf_dt.reshape(-1, 1)), label='Predicted (DT_W) Random Forest', marker='o')
plt.title('DT_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Trực quan hóa kết quả cho cột V_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test_v):],
         scaler.inverse_transform(y_test_v.reshape(-1, 1)), label='Actual (V_W)', marker='o')
plt.plot(df_selected.index[-len(y_test_v):],
         scaler.inverse_transform(y_pred_lstm_v.reshape(-1, 1)), label='Predicted (V_W) LSTM', marker='o')
plt.plot(df_selected.index[-len(y_test_v):],
         scaler.inverse_transform(y_pred_dt_v.reshape(-1, 1)), label='Predicted (V_W) Decision Tree', marker='o')
plt.plot(df_selected.index[-len(y_test_v):],
         scaler.inverse_transform(y_pred_rf_v.reshape(-1, 1)), label='Predicted (V_W) Random Forest', marker='o')
plt.title('V_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()
