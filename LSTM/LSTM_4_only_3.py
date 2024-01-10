import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, History
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
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


print("Một vài dữ liệu đầu tiên:")
print(df_selected.head())
print(f"Tổng số lượng dữ gốc: {len(df_selected)}")


# Chuẩn hóa
# Xử lý giá trị NaN
df_selected = df_selected.fillna(method="ffill", inplace=False)
# Xóa dữ liệu trùng lặp
df_selected = df_selected.drop_duplicates()

print(df_selected.head())
print(f"Tổng số lượng dữ liệu sau khi xử lý: {len(df_selected)}")

# chuẩn hóa Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_selected)

# Chia thành tập huấn luyện và tập kiểm tra (80-20)
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)


# Biểu đồ phân tích xu hướng dữ liệu gốc cho cột 'USD_W'
# plt.figure(figsize=(12, 6))
# plt.plot(df_selected['USD_W'], label='USD_W', color='blue')
# plt.title('Trend Analysis for USD_W')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# #  'DT_W'
# plt.figure(figsize=(12, 6))
# plt.plot(df_selected['DT_W'], label='DT_W', color='green')
# plt.title('Trend Analysis for DT_W')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# #  'V_W'
# plt.figure(figsize=(12, 6))
# plt.plot(df_selected['V_W'], label='V_W', color='red')
# plt.title('Trend Analysis for V_W')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.ylabel('Value')
# plt.legend()
# plt.show()


# Chuẩn Bị Dữ Liệu cho LSTM, GRU, và RNN
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


time_steps = 10
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

#  tạo mô hình cho LSTM


def create_lstm_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# Wrap mô hình vào KerasRegressor để sử dụng với RandomizedSearchCV
lstm_model = KerasRegressor(
    build_fn=create_lstm_model, epochs=300, batch_size=32, verbose=0)


# Định nghĩa các giá trị thử nghiệm cho các siêu tham số
param_dist = {
    # 16, 32, 64, 128, 256
    'units': [64],
    # 'sigmoid', 'tanh', 'relu'
    'activation': ['tanh'],
    # 0.1, 0.2, 0.25, 0.5
    'dropout_rate': [0.2],
    # 0.001, 0.005, 0.01
    'learning_rate': [0.01],
}

# Tìm kiếm siêu tham số bằng RandomizedSearchCV cho LSTM
random_search_lstm = RandomizedSearchCV(estimator=lstm_model, param_distributions=param_dist,
                                        scoring='neg_mean_squared_error', n_iter=10, cv=3, verbose=1, random_state=42)
random_search_lstm_result = random_search_lstm.fit(X_train, y_train)

# In kết quả tìm kiếm siêu tham số cho LSTM
print("Best LSTM: %f using %s" % (random_search_lstm_result.best_score_,
      random_search_lstm_result.best_params_))
best_lstm_model = random_search_lstm.best_estimator_.model

history_lstm = History()

# Huấn luyện mô hình LSTM với siêu tham số tốt nhất
best_lstm_model.fit(X_train, y_train, epochs=1000, batch_size=32,
                    validation_data=(X_test, y_test), shuffle=False, callbacks=[history_lstm], verbose=0)


# Dự báo trên tập kiểm tra cho LSTM
y_pred_lstm = best_lstm_model.predict(X_test)

# Đánh giá mô hình và tính MSE cho từng cột dữ liệu
mse_lstm = mean_squared_error(y_test, y_pred_lstm, multioutput='raw_values')

# In kết quả MSE cho từng cột dữ liệu
print("\nMSE (LSTM) for each column:")
for i, column_name in enumerate(selected_columns[1:]):
    print(f"{column_name}: {mse_lstm[i]}")


# Vẽ đồ thị cho mô hình LSTM
plt.figure(figsize=(15, 8))
plt.plot(y_test[:, 0], label='Actual')
plt.plot(y_pred_lstm[:, 0], label='LSTM Prediction')
plt.title('USD_W - LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()

plt.figure(figsize=(15, 8))
plt.plot(y_test[:, 1], label='Actual')
plt.plot(y_pred_lstm[:, 1], label='LSTM Prediction')
plt.title('USD_W - LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()

plt.figure(figsize=(15, 8))
plt.plot(y_test[:, 2], label='Actual')
plt.plot(y_pred_lstm[:, 2], label='LSTM Prediction')
plt.title('USD_W - LSTM')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()


plt.tight_layout()
plt.show()


# for i, column_name in enumerate(selected_columns[1:]):
#     plt.figure(figsize=(15, 8))

#     # Vẽ dữ liệu thực tế
#     plt.plot(df_selected.index[-len(y_test):],
#              y_test[:, i], label='Actual', color='black')

#     # Vẽ dự báo từ mô hình LSTM
#     plt.plot(df_selected.index[-len(y_test):], y_pred_lstm[:, i],
#              label='LSTM Prediction', linestyle='dashed', color='blue')

#     # Vẽ dự báo từ mô hình GRU
#     plt.plot(df_selected.index[-len(y_test):], y_pred_gru[:, i],
#              label='GRU Prediction', linestyle='dashed', color='green')

#     # Vẽ dự báo từ mô hình RNN
#     plt.plot(df_selected.index[-len(y_test):], y_pred_rnn[:, i],
#              label='RNN Prediction', linestyle='dashed', color='red')

#     # Thiết lập các thuộc tính đồ thị
#     plt.title(f'Comparison of Predictions for {column_name}')
#     plt.xlabel('Time Steps')
# plt.ylabel('Value')
#     plt.ylabel('Scaled Value')
#     plt.legend()
#     plt.show()

# Dự báo và thực tế cho 'USD_W'
# plt.figure(figsize=(12, 6))
# plt.plot(y_test[:, 0], label='Actual', color='blue')
# plt.plot(y_pred_lstm[:, 0], label='LSTM', linestyle='dashed', color='orange')
# plt.plot(y_pred_gru[:, 0], label='GRU', linestyle='dashed', color='green')
# plt.plot(y_pred_rnn[:, 0], label='RNN', linestyle='dashed', color='red')
# plt.title('USD_W - Actual vs Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# # Dự báo và thực tế cho 'DT_W'
# plt.figure(figsize=(12, 6))
# plt.plot(y_test[:, 1], label='Actual', color='blue')
# plt.plot(y_pred_lstm[:, 1], label='LSTM', linestyle='dashed', color='orange')
# plt.plot(y_pred_gru[:, 1], label='GRU', linestyle='dashed', color='green')
# plt.plot(y_pred_rnn[:, 1], label='RNN', linestyle='dashed', color='red')
# plt.title('DT_W - Actual vs Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# # Dự báo và thực tế cho 'V_W'
# plt.figure(figsize=(12, 6))
# plt.plot(y_test[:, 2], label='Actual', color='blue')
# plt.plot(y_pred_lstm[:, 2], label='LSTM', linestyle='dashed', color='orange')
# plt.plot(y_pred_gru[:, 2], label='GRU', linestyle='dashed', color='green')
# plt.plot(y_pred_rnn[:, 2], label='RNN', linestyle='dashed', color='red')
# plt.title('V_W - Actual vs Predicted')
# plt.xlabel('Time Steps')
# plt.ylabel('Value')
# plt.legend()

# plt.tight_layout()
# plt.show()
