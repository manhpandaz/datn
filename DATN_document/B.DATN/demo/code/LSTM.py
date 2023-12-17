import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Bước 1: Đọc và tiền xử lý dữ liệu
data = pd.read_excel('../data.xlsx')
data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)

# Bước 2: Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data['USD_W'] = scaler.fit_transform(data['USD_W'].values.reshape(-1, 1))
data['DT_W'] = scaler.fit_transform(data['DT_W'].values.reshape(-1, 1))
data['V_W'] = scaler.fit_transform(data['V_W'].values.reshape(-1, 1))

# Bước 3: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Bước 4: Xây dựng mô hình LSTM
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

model = create_lstm_model()

# Bước 5: Đào tạo mô hình
X_train, y_train = train_data.drop('USD_W', axis=1).values, train_data['USD_W'].values
X_test, y_test = test_data.drop('USD_W', axis=1).values, test_data['USD_W'].values

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model.fit(X_train, y_train, epochs=100, batch_size=64)

# Bước 6: Dự đoán và đánh giá
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Bước 7: Trực quan hóa kết quả
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size:], y_test, label='Thực tế')
plt.plot(data.index[train_size:], y_pred, label='Dự đoán')
plt.legend()
plt.show()
