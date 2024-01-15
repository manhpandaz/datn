import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
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

# Dự báo trên tập kiểm tra bằng LSTM
y_pred_lstm = model_lstm.predict(X_test)

# Xây dựng mô hình ARIMA
model_arima = ARIMA(train_data[:, 0], order=(5, 1, 0))
model_arima_fit = model_arima.fit()

# Dự báo trên tập kiểm tra bằng ARIMA
arima_predictions = model_arima_fit.forecast(steps=len(test_data))
mse_arima = np.mean((arima_predictions - test_data[:, 0])**2)
print(f"Mean Squared Error (ARIMA) on Test Data: {mse_arima}")

# Xây dựng mô hình Decision Tree
model_dt = RandomForestRegressor(n_estimators=100, random_state=42)
model_dt.fit(X_train.reshape((X_train.shape[0], -1)), y_train[:, 0])

# Dự báo trên tập kiểm tra bằng Decision Tree
y_pred_dt = model_dt.predict(X_test.reshape((X_test.shape[0], -1)))
mse_dt = np.mean((y_pred_dt - y_test[:, 0])**2)
print(f"Mean Squared Error (Decision Tree) on Test Data: {mse_dt}")

# -------- Trực quan hóa mô hình
# Đảo ngược chuẩn hóa để có giá trị gốc
y_test_inverse = scaler.inverse_transform(y_test)

y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)
print("y_pred_lstm_inverse", y_pred_lstm_inverse.shape)
y_pred_dt_inverse = scaler.inverse_transform(y_pred_dt.reshape(-1, 3))
print("_pred_dt:", y_pred_dt.shape)
arima_predictions_inverse = scaler.inverse_transform(
    arima_predictions.reshape(-1, 3))
print("arima_predictions_inverse:", arima_predictions_inverse.shape)
# Tạo mảng các giá trị thời gian
time_steps_test = df_selected.index[train_size + time_steps:]

# Trực quan hóa kết quả
plt.figure(figsize=(15, 8))
plt.plot(time_steps_test, y_test_inverse[:, 0], label='Actual', marker='o')
plt.plot(time_steps_test,
         y_pred_lstm_inverse[:, 0], label='LSTM Predicted', marker='o')
plt.plot(time_steps_test, y_pred_dt_inverse,
         label='Decision Tree Predicted', marker='o')
plt.plot(time_steps_test, arima_predictions_inverse,
         label='ARIMA Predicted', marker='o')
plt.title('Gold Price Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Gold Price (USD/oz)')
plt.legend()
plt.show()
