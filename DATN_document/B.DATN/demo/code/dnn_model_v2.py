import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ tệp CSV hoặc nguồn dữ liệu khác
# Ở đây, chúng ta giả định rằng bạn đã có dữ liệu CSV với các cột 'USD_W', 'DT_W', 'V_W', và 'Xuat_khau_dau_VN'
# Thay đổi tên tệp CSV và cột tương ứng cho dữ liệu của bạn.
data = pd.read_csv('../data/Dulieu_vang_dau_tygia.csv')

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = data[['USD_W', 'DT_W', 'V_W']]
y = data['Xuat_khau_dau_VN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình DNN
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Đầu ra là một giá trị liên tục
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Đánh giá mô hình trên tập kiểm tra
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Data: {mse}")
