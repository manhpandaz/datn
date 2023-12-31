import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
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

# Trực quan hóa dữ liệu gốc
# plt.figure(figsize=(12, 6))

# plt.plot(df_selected.index, df_selected['USD_W'], label='USD_W', marker='o')
# plt.plot(df_selected.index, df_selected['DT_W'], label='DT_W', marker='o')
# plt.plot(df_selected.index, df_selected['V_W'], label='V_W', marker='o')

# plt.title('Original Data - USD_W, DT_W, V_W')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# Chuẩn hóa dữ liệu sử dụng Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_selected)

# Chia thành tập huấn luyện và tập kiểm tra (tỷ lệ 80-20)
# train_size = int(len(df_scaled) * 0.8)
# train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

# Chuẩn Bị Dữ Liệu cho LSTM


def prepare_data(data, time_steps):
    if len(data) == 0:
        raise ValueError("Dữ liệu đầu vào không được rỗng.")
    if time_steps >= len(data):
        raise ValueError("time_steps phải nhỏ hơn chiều dài của dữ liệu.")
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10  # Số lượng bước thời gian quan sát trước đó
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)


lstm_forget = LSTM(units=50, activation='sigmoid', recurrent_activation='sigmoid', return_sequences=True)
lstm_input = LSTM(units=50, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)
lstm_output = LSTM(units=50, activation='tanh', recurrent_activation='sigmoid')
# Xây dựng mô hình LSTM
model = Sequential()
model.add(lstm_forget)
model.add(lstm_input)
model.add(lstm_output)
model.add(Dropout(0.2))
model.add(Dense(units=3))
model.compile(optimizer='adam', loss='mse')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, y_test), shuffle=False)

# Đánh giá mô hình trên tập kiểm tra
# mse = model.evaluate(X_test, y_test)
# print(f"Mean Squared Error on Test Data: {mse}")

# Dự báo trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính MSE cho từng cột dữ liệu
for i, column in enumerate(selected_columns[1:]):
    mse_column = mean_squared_error(y_test[:, i], y_pred[:, i])
    print(f"Mean Squared Error for Column {column}: {mse_column}")

# Đảo ngược chuẩn hóa để có giá trị gốc
y_test_inverse = scaler.inverse_transform(y_test)
y_pred_inverse = scaler.inverse_transform(y_pred)


# Trực quan hóa kết quả cho cột USD_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test):],
         y_test_inverse[:, 0], label='Actual', marker='o')
plt.plot(df_selected.index[-len(y_test):],
         y_pred_inverse[:, 0], label='Predicted', marker='o')
plt.title('USD_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

#  DT_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test):],
         y_test_inverse[:, 1], label='Actual', marker='o')
plt.plot(df_selected.index[-len(y_test):],
         y_pred_inverse[:, 1], label='Predicted', marker='o')
plt.title('DT_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

#  V_W
plt.figure(figsize=(12, 6))
plt.plot(df_selected.index[-len(y_test):],
         y_test_inverse[:, 2], label='Actual', marker='o')
plt.plot(df_selected.index[-len(y_test):],
         y_pred_inverse[:, 2], label='Predicted', marker='o')
plt.title('V_W - Actual vs. Predicted')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# plt.subplot(1, 3, 1)  # Lưới 1x3, ô ở vị trí 1
# plt.plot(df_selected.index[-len(y_test):], y_test_inverse[:, 0], label='Actual', marker='o')
# plt.plot(df_selected.index[-len(y_test):], y_pred_inverse[:, 0], label='Predicted', marker='o')
# plt.title('USD_W - Actual vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# # Trực quan hóa kết quả cho cột DT_W
# plt.subplot(1, 3, 2)  # Lưới 1x3, ô ở vị trí 2
# plt.plot(df_selected.index[-len(y_test):], y_test_inverse[:, 1], label='Actual', marker='o')
# plt.plot(df_selected.index[-len(y_test):], y_pred_inverse[:, 1], label='Predicted', marker='o')
# plt.title('DT_W - Actual vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# # Trực quan hóa kết quả cho cột V_W
# plt.subplot(1, 3, 3)  # Lưới 1x3, ô ở vị trí 3
# plt.plot(df_selected.index[-len(y_test):], y_test_inverse[:, 2], label='Actual', marker='o')
# plt.plot(df_selected.index[-len(y_test):], y_pred_inverse[:, 2], label='Predicted', marker='o')
# plt.title('V_W - Actual vs. Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# plt.show()
