import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, History
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

# Xử lý giá trị NaN
df_selected = df_selected.fillna(method="ffill", inplace=False)
# Xóa dữ liệu trùng lặp
df_selected = df_selected.drop_duplicates()

# Chuẩn hóa Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_selected)

# Chia thành tập huấn luyện và tập kiểm tra (80-20)
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

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

# Định nghĩa callback để lưu trữ trạng thái tốt nhất của mô hình dựa trên val_loss
checkpoint_lstm = ModelCheckpoint('best_lstm_model.h5', save_best_only=True)
checkpoint_gru = ModelCheckpoint('best_gru_model.h5', save_best_only=True)
checkpoint_rnn = ModelCheckpoint('best_rnn_model.h5', save_best_only=True)

# Định nghĩa callback để lưu lại giá trị loss và val_loss
history_lstm = History()
history_gru = History()
history_rnn = History()

# Sử dụng callbacks trong quá trình huấn luyện


def create_lstm_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def create_gru_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(GRU(units=units, activation=activation,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


def create_rnn_model(units=50, activation='relu', dropout_rate=0.0, learning_rate=0.001):
    model = Sequential()
    model.add(SimpleRNN(units=units, activation=activation,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=3))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model


# Wrap mô hình vào KerasRegressor để sử dụng với RandomizedSearchCV
lstm_model = KerasRegressor(
    build_fn=create_lstm_model, epochs=300, batch_size=32, verbose=0)
gru_model = KerasRegressor(build_fn=create_gru_model,
                           epochs=300, batch_size=32, verbose=0)
rnn_model = KerasRegressor(build_fn=create_rnn_model,
                           epochs=300, batch_size=32, verbose=0)

param_dist = {
    # 16, 32, 64, 128, 256
    'units': [16, 32, 64, 128, 256],
    # 'sigmoid', 'tanh', 'relu'
    'activation': ['sigmoid', 'tanh', 'relu'],
    # 0.1, 0.2, 0.25, 0.5
    'dropout_rate': [0.0, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6],
    # 0.001, 0.005, 0.01
    'learning_rate': [0.01, 0.001, 0.005],
}

# Tìm kiếm siêu tham số bằng RandomizedSearchCV cho LSTM
random_search_lstm = RandomizedSearchCV(estimator=lstm_model, param_distributions=param_dist,
                                        scoring='neg_mean_squared_error', n_iter=10, cv=3, verbose=1, random_state=42)
random_search_lstm_result = random_search_lstm.fit(X_train, y_train)

# Huấn luyện mô hình LSTM với siêu tham số tốt nhất
best_lstm_model = random_search_lstm.best_estimator_.model
best_lstm_model.fit(X_train, y_train, epochs=1000, batch_size=32,
                    validation_data=(X_test, y_test), shuffle=False, verbose=0,
                    callbacks=[checkpoint_lstm, history_lstm])

# Tìm kiếm siêu tham số bằng RandomizedSearchCV cho GRU
random_search_gru = RandomizedSearchCV(estimator=gru_model, param_distributions=param_dist,
                                       scoring='neg_mean_squared_error', n_iter=10, cv=3, verbose=1, random_state=42)
random_search_gru_result = random_search_gru.fit(X_train, y_train)

# Huấn luyện mô hình GRU với siêu tham số tốt nhất
best_gru_model = random_search_gru.best_estimator_.model
best_gru_model.fit(X_train, y_train, epochs=1000, batch_size=32,
                   validation_data=(X_test, y_test), shuffle=False, verbose=0,
                   callbacks=[checkpoint_gru, history_gru])

# Tìm kiếm siêu tham số bằng RandomizedSearchCV cho RNN
random_search_rnn = RandomizedSearchCV(estimator=rnn_model, param_distributions=param_dist,
                                       scoring='neg_mean_squared_error', n_iter=10, cv=3, verbose=1, random_state=42)
random_search_rnn_result = random_search_rnn.fit(X_train, y_train)

# Huấn luyện mô hình RNN với siêu tham số tốt nhất
best_rnn_model = random_search_rnn.best_estimator_.model
best_rnn_model.fit(X_train, y_train, epochs=1000, batch_size=32,
                   validation_data=(X_test, y_test), shuffle=False, verbose=0,
                   callbacks=[checkpoint_rnn, history_rnn])

# Vẽ biểu đồ loss trên tập kiểm thử cho LSTM
plt.figure(figsize=(15, 8))
plt.plot(history_lstm.history['val_loss'],
         label='LSTM Validation Loss', color='blue')
plt.title('LSTM Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

# Vẽ biểu đồ loss trên tập kiểm thử cho GRU
plt.figure(figsize=(15, 8))
plt.plot(history_gru.history['val_loss'],
         label='GRU Validation Loss', color='green')
plt.title('GRU Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()

# Vẽ biểu đồ loss trên tập kiểm thử cho RNN
plt.figure(figsize=(15, 8))
plt.plot(history_rnn.history['val_loss'],
         label='RNN Validation Loss', color='red')
plt.title('RNN Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()
