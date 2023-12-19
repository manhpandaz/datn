import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
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

# Chia thành tập huấn luyện và tập kiểm tra (80-20)
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

# Chuẩn Bị Dữ Liệu cho LSTM, GRU, và RNN


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Tạo mô hình cho LSTM với các tham số cần tối ưu


def create_lstm_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation,
                   input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Tạo mô hình cho GRU với các tham số cần tối ưu


def create_gru_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001, batch_size=32):
    model = Sequential()
    model.add(GRU(units=units, activation=activation,
                  input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Tạo mô hình cho RNN với các tham số cần tối ưu


def create_rnn_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001, batch_size=32):
    model = Sequential()
    model.add(SimpleRNN(units=units, activation=activation,
                        input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# Wrap mô hình vào KerasRegressor để sử dụng với GridSearchCV
lstm_model = KerasRegressor(
    build_fn=create_lstm_model, epochs=50, batch_size=32, verbose=0)
gru_model = KerasRegressor(build_fn=create_gru_model,
                           epochs=50, batch_size=32, verbose=0)
rnn_model = KerasRegressor(build_fn=create_rnn_model,
                           epochs=50, batch_size=32, verbose=0)

# Định nghĩa các giá trị thử nghiệm cho các siêu tham số
param_grid = {
    'units': [16, 32, 64, 128, 256],
    'activation': ['tanh', 'relu'],
    'dropout_rate': [0.1, 0.2, 0.25, 0.5],
    'learning_rate': [0.001, 0.005, 0.01],
    'batch_size': [16, 32, 64, 128, 256],
}

# Tìm kiếm siêu tham số bằng GridSearchCV cho LSTM
grid_search_lstm = GridSearchCV(estimator=lstm_model, param_grid=param_grid,
                                scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search_lstm_result = grid_search_lstm.fit(X_train, y_train)

# In kết quả tìm kiếm siêu tham số cho LSTM
print("Best LSTM: %f using %s" % (grid_search_lstm_result.best_score_,
      grid_search_lstm_result.best_params_))
best_lstm_model = grid_search_lstm.best_estimator_.model

# Tìm kiếm siêu tham số bằng GridSearchCV cho GRU
grid_search_gru = GridSearchCV(estimator=gru_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search_gru_result = grid_search_gru.fit(X_train, y_train)

# In kết quả tìm kiếm siêu tham số cho GRU
print("Best GRU: %f using %s" % (grid_search_gru_result.best_score_,
      grid_search_gru_result.best_params_))
best_gru_model = grid_search_gru.best_estimator_.model

# Tìm kiếm siêu tham số bằng GridSearchCV cho RNN
grid_search_rnn = GridSearchCV(estimator=rnn_model, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=3, verbose=1)
grid_search_rnn_result = grid_search_rnn.fit(X_train, y_train)

# In kết quả tìm kiếm siêu tham số cho RNN
print("Best RNN: %f using %s" % (grid_search_rnn_result.best_score_,
      grid_search_rnn_result.best_params_))
best_rnn_model = grid_search_rnn.best_estimator_.model

# Huấn luyện mô hình LSTM với siêu tham số tốt nhất
best_lstm_model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

# Huấn luyện mô hình GRU với siêu tham số tốt nhất
best_gru_model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

# Huấn luyện mô hình RNN với siêu tham số tốt nhất
best_rnn_model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=0)

# Dự báo trên tập kiểm tra cho LSTM, GRU, và RNN
y_pred_lstm = best_lstm_model.predict(X_test)
y_pred_gru = best_gru_model.predict(X_test)
y_pred_rnn = best_rnn_model.predict(X_test)

# Đánh giá mô hình và tính MSE cho từng cột dữ liệu
mse_lstm = mean_squared_error(y_test, y_pred_lstm, multioutput='raw_values')
mse_gru = mean_squared_error(y_test, y_pred_gru, multioutput='raw_values')
mse_rnn = mean_squared_error(y_test, y_pred_rnn, multioutput='raw_values')

# In kết quả MSE cho từng cột dữ liệu
print("\nMSE (LSTM) for each column:")
for i, column_name in enumerate(selected_columns[1:]):
    print(f"{column_name}: {mse_lstm[i]}")

print("\nMSE (GRU) for each column:")
for i, column_name in enumerate(selected_columns[1:]):
    print(f"{column_name}: {mse_gru[i]}")

print("\nMSE (RNN) for each column:")
for i, column_name in enumerate(selected_columns[1:]):
    print(f"{column_name}: {mse_rnn[i]}")

# Trực quan hóa kết quả cho mỗi cột dữ liệu từ cả 3 mô hình
for i, column_name in enumerate(selected_columns[1:]):
    plt.figure(figsize=(15, 8))

    # Vẽ dữ liệu thực tế
    plt.plot(df_selected.index[-len(y_test):],
             y_test[:, i], label='Actual', color='black')

    # Vẽ dự báo từ mô hình LSTM
    plt.plot(df_selected.index[-len(y_test):], y_pred_lstm[:, i],
             label='LSTM Prediction', linestyle='dashed', color='blue')

    # Vẽ dự báo từ mô hình GRU
    plt.plot(df_selected.index[-len(y_test):], y_pred_gru[:, i],
             label='GRU Prediction', linestyle='dashed', color='green')

    # Vẽ dự báo từ mô hình RNN
    plt.plot(df_selected.index[-len(y_test):], y_pred_rnn[:, i],
             label='RNN Prediction', linestyle='dashed', color='red')

    # Thiết lập các thuộc tính đồ thị
    plt.title(f'Comparison of Predictions for {column_name}')
    plt.xlabel('Date')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.show()
