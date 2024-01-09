import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import History
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

# Xử lý giá trị NaN
df_selected = df_selected.fillna(method="ffill", inplace=False)
# Xóa dữ liệu trùng lặp
df_selected = df_selected.drop_duplicates(inplace=False)

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
history = History()
# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(units=50, activation='tanh', input_shape=(
    X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=3))
model.compile(optimizer='adam', loss='mse')


# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=1000, batch_size=32,
          validation_data=(X_test, y_test), shuffle=False, callbacks=[history])


loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
# print(loss_values)
# print("val_loss_values:", val_loss_values)

# Lấy giá trị mất mát trên tập huấn luyện và tập validation từ callback history

# biểu đồ biểu thị loss vaidation value
epochs = range(1, len(loss_values) + 1)

# plt.plot(epochs, loss_values, label='Training Loss', marker='o')
plt.plot(epochs, val_loss_values, label='Validation Loss', marker='o')

plt.title('Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss Values')
plt.legend()

plt.show()

# Đánh giá mô hình trên tập kiểm tra
# mse = model.evaluate(X_test, y_test)
# print(f"Mean Squared Error on Test Data: {mse}")

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
