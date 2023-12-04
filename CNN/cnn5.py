import matplotlib.pyplot as plt
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Đường dẫn đến tập dữ liệu
DATAFILE = 'F:/DATN/data/DulieuVang_dau_Tygia.csv'

# Chỉnh lại tỷ lệ tập huấn luyện và tập kiểm tra
TRAIN_VALID_RATIO = 0.70

# Các hàm đánh giá tùy chỉnh


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def f1macro(y_true, y_pred):
    f_pos = f1_m(y_true, y_pred)
    f_neg = f1_m(1 - y_true, 1 - K.clip(y_pred, 0, 1))
    return (f_pos + f_neg) / 2

# Hàm tạo mô hình CNNpred


def cnnpred_2d(seq_len=60, n_features=4, n_filters=(8, 8, 8), droprate=0.1):
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu"),
        Conv2D(n_filters[1], kernel_size=(3, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),
        Conv2D(n_filters[2], kernel_size=(3, 1), activation="relu"),
        MaxPool2D(pool_size=(2, 1)),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    return model

# Hàm tạo dữ liệu động


def datagen(data, seq_len, batch_size, targetcol, kind):
    batch = []
    while True:
        key = list(data.keys())[0]  # Lấy tên tập dữ liệu duy nhất
        df = data[key]
        input_cols = [c for c in df.columns if c != targetcol]
        if kind == 'train':
            index = df.index[:int(len(df) * TRAIN_VALID_RATIO)]
        elif kind == 'valid':
            index = df.index[int(len(df) * TRAIN_VALID_RATIO):]
        else:
            raise NotImplementedError

        while True:
            t = random.choice(index)
            n = (df.index == t).argmax()
            if n - seq_len + 1 < 0:
                continue
            frame = df.iloc[n - seq_len + 1:n + 1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break

        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []

# Hàm tạo dữ liệu kiểm tra


def testgen(data, seq_len, targetcol):
    batch = []
    key = list(data.keys())[0]
    df = data[key]
    input_cols = [c for c in df.columns if c != targetcol]
    t = df.index[df.index >= TRAIN_VALID_RATIO][0]
    n = (df.index == t).argmax()

    for i in range(n + 1, len(df) + 1):
        frame = df.iloc[i - seq_len:i]
        batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X), 3), np.array(y)


# Đọc dữ liệu
# DATAFILEFLOAT = float(DATAFILE.replace(',', ''))
X = pd.read_csv(DATAFILE, usecols=["DATE", "USD_W", "DT_W", "V_W"])

X.set_index("DATE", inplace=True)
cols = X.columns
X.dropna(inplace=True)


scaler = StandardScaler()
X[cols] = scaler.fit_transform(X[cols])
data = {"single_dataset": X}

print(data)

# Huấn luyện mô hình
# TRAIN_TEST_CUTOFF = X.index[int(len(X) * TRAIN_VALID_RATIO)]
# seq_len = 60
# batch_size = 100
# n_epochs = 10
# n_features = 4
# model = cnnpred_2d(seq_len, n_features)
# model.compile(optimizer="adam", loss="mae", metrics=["acc", f1macro])

# # Chia dữ liệu thành tập huấn luyện và tập kiểm tra tự động
# train_data, test_data = train_test_split(
#     data, test_size=0.3, train_size=0.7, random_state=42)

# # Callback để lưu mô hình tốt nhất
# checkpoint_path = "./cp2d-{epoch}-{val_f1macro:.2f}.h5"
# callbacks = [
#     ModelCheckpoint(checkpoint_path, monitor='val_f1macro', mode="max",
#                     verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
# ]

# # Đào tạo mô hình
# history = model.fit(datagen(train_data, seq_len, batch_size, "Target", "train"),
#                     validation_data=datagen(
#                         train_data, seq_len, batch_size, "Target", "valid"),
#                     epochs=n_epochs, steps_per_epoch=100, validation_steps=10, verbose=1, callbacks=callbacks)

# # Trực quan hóa độ chính xác và mất mát
# train_acc = history.history['acc']
# val_acc = history.history['val_acc']
# train_loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# # Chuẩn bị dữ liệu kiểm tra và kiểm tra mô hình
# test_data, test_target = testgen(test_data, seq_len, "Target")
# test_out = model.predict(test_data)
# test_pred = (test_out > 0.5).astype(int)
# print("Accuracy:", accuracy_score(test_pred, test_target))
# print("MAE:", mean_absolute_error(test_pred, test_target))
# print("F1 Score:", f1_score(test_pred, test_target))
