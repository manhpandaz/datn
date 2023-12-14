import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Đọc dữ liệu từ nguồn
df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')

print("Một số dữ liệu đầu tiên:")
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

# Chia thành tập huấn luyện và tập kiểm tra (80-20)
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

# Chuẩn Bị Dữ Liệu cho LSTM


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Xây dựng mô hình LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])))
# Output layer có 3 units tương ứng với 3 cột dữ liệu
model_lstm.add(Dense(units=3))
model_lstm.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình LSTM
model_lstm.fit(X_train, y_train, epochs=1000, batch_size=32,
               validation_data=(X_test, y_test), shuffle=False)

# Dự báo trên tập kiểm tra cho LSTM
y_pred_lstm = model_lstm.predict(X_test)

#  ARIMA
for i in range(1, 4):
    model_arima = ARIMA(train_data[:, i-1], order=(5, 1, 0))
    model_arima_fit = model_arima.fit()
    y_pred_arima = model_arima_fit.forecast(steps=len(test_data))
    mse_arima = mean_squared_error(test_data[:, i-1], y_pred_arima)
    print(f"MSE (ARIMA - {selected_columns[i]}): {mse_arima:.4f}")

# Prophet
for i in range(1, 4):
    prophet_data = pd.DataFrame(
        {'ds': df_selected.index, 'y': df_selected[selected_columns[i]].values})
    prophet_data.columns = ['ds', 'y']

    model_prophet = Prophet()
    model_prophet.fit(prophet_data)

    future = model_prophet.make_future_dataframe(periods=len(test_data))
    forecast = model_prophet.predict(future)
    y_pred_prophet = forecast.tail(len(test_data))['yhat'].values
    mse_prophet = mean_squared_error(test_data[:, i-1], y_pred_prophet)
    print(f"MSE (Prophet - {selected_columns[i]}): {mse_prophet:.4f}")

# Trực quan hóa kết quả cho cột USD_W của LSTM
# (Code vẽ biểu đồ cho LSTM ở đây)

# Trực quan hóa kết quả cho cột USD_W của ARIMA
# plt.figure(figsize=(12, 6))
# plt.plot(df_selected.index[-len(test_data):], test_data[:, 0], label='Actual (USD_W)', marker='o')
# plt.plot(df_selected.index[-len(test_data):], y_pred_arima, label='Predicted (ARIMA - USD_W)', marker='o')
# plt.title('USD_W - Actual vs. Predicted (ARIMA)')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# Trực quan hóa kết quả cho cột USD_W của Prophet
# (Code vẽ biểu đồ cho Prophet ở đây)
