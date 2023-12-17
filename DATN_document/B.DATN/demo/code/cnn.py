import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv(os.path.join("../../main/data/data2.csv"))

# Lựa chọn các trường dữ liệu quan trọng
selected_columns = ["USD_W", "DT_W", "V_W"]
data = data["USD_W", "DT_W", "V_W"]

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Chia dữ liệu thành dữ liệu đầu vào và đầu ra
X = data[:-1]  # Dữ liệu đầu vào là các dòng trừ dòng cuối
y = data[1:]   # Dữ liệu đầu ra là các dòng trừ dòng đầu

# Chia thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Xây dựng mô hình CNN
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(3, activation='linear'))  # 3 là số lượng trường dữ liệu đầu ra

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=100, batch_size=32, validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test))

# Biểu đồ hóa kết quả huấn luyện
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Dự đoán dữ liệu mới
new_data_point = np.array([[0.8, 0.9, 1.0]])  # Thay thế giá trị này bằng dữ liệu mới
predicted_value = model.predict(new_data_point.reshape(1, new_data_point.shape[1], 1))
print(f"Predicted Values: {predicted_value}")
