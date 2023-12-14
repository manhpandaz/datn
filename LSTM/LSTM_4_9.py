import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from scikeras.wrappers import KerasRegressor

# Đọc dữ liệu từ nguồn
df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')

print("Mỗi vài dữ liệu đầu tiên:")
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


time_steps = 100
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Hàm tạo mô hình LSTM


def create_lstm_model(units=50, activation='relu', optimizer='adam', dropout=0.2, recurrent_dropout=0.2, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation, input_shape=(
        X_train.shape[1], X_train.shape[2]), dropout=dropout, recurrent_dropout=recurrent_dropout))
    model.add(Dense(units=3))
    model.compile(optimizer=optimizer, loss='mse')
    return model


# Đóng gói mô hình Keras để sử dụng với RandomizedSearchCV
keras_lstm_model = KerasRegressor(
    build_fn=create_lstm_model, epochs=100, batch_size=32, verbose=0)

# Định nghĩa không gian siêu tham số
param_dist = {
    'units': [50, 100, 150],
    'activation': ['relu', 'tanh'],
    'optimizer': ['adam', 'sgd'],
    'dropout': [0.2, 0.3, 0.4],
    'recurrent_dropout': [0.2, 0.3, 0.4],
    'batch_size': [16, 32, 64]
}

# Thực hiện Randomized Search
random_search = RandomizedSearchCV(
    estimator=keras_lstm_model, param_distributions=param_dist, n_iter=10, cv=3, verbose=2, n_jobs=-1)
random_search_result = random_search.fit(X_train, y_train)

# In ra kết quả tìm kiếm
print("Best Parameters: ", random_search_result.best_params_)

# Sử dụng siêu tham số tối ưu để xây dựng mô hình LSTM
best_units = random_search_result.best_params_['units']
best_activation = random_search_result.best_params_['activation']
best_optimizer = random_search_result.best_params_['optimizer']
best_dropout = random_search_result.best_params_['dropout']
best_recurrent_dropout = random_search_result.best_params_['recurrent_dropout']
best_batch_size = random_search_result.best_params_['batch_size']

model_lstm = create_lstm_model(units=best_units, activation=best_activation, optimizer=best_optimizer,
                               dropout=best_dropout, recurrent_dropout=best_recurrent_dropout, batch_size=best_batch_size)

# Huấn luyện mô hình LSTM
model_lstm.fit(X_train, y_train, epochs=100, batch_size=best_batch_size,
               validation_data=(X_test, y_test), shuffle=False)

# Dự báo trên tập kiểm tra cho LSTM
y_pred_lstm = model_lstm.predict(X_test)


mse_lstm = mean_squared_error(y_test, y_pred_lstm, multioutput='raw_values')
print("MSE (LSTM) for each column:")
for i, column_name in enumerate(selected_columns[1:]):
    print(f"{column_name}: {mse_lstm[i]:.4f}")
