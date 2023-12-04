import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

DATADIR = 'F:/DATN/data/DulieuVang_dau_Tygia.csv'
TRAIN_TEST_CUTOFF = '2006-01-01'
TRAIN_VALID_RATIO = 0.7

# Custom F1 metric functions


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

# Define the CNN model


def cnnpred_1d(seq_len, n_features, n_filters=(8, 8, 8), droprate=0.1):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Conv1D(n_filters[0], kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(n_filters[1], kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(n_filters[2], kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation='sigmoid')
    ])
    return model

# Data generator


def datagen(data, seq_len, batch_size, targetcol, kind):
    batch = []
    while True:
        key = random.choice(list(data.keys()))
        df = data[key]
        input_cols = [c for c in df.columns if c != targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        assert split > seq_len, "Training data too small for sequence length {}".format(
            seq_len)
        if kind == 'train':
            index = index[:split]
        elif kind == 'valid':
            index = index[split:]
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
            X, y = np.array(X), np.array(y)
            yield X, y
            batch = []

# Test data generator


def testgen(data, seq_len, targetcol):
    batch = []
    for key, df in data.items():
        input_cols = [c for c in df.columns if c != targetcol]
        t = df.index[df.index >= TRAIN_TEST_CUTOFF][0]
        n = (df.index == t).argmax()
        for i in range(n + 1, len(df) + 1):
            frame = df.iloc[i - seq_len:i]
            batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.array(X), np.array(y)


data = {}
X = pd.read_csv(DATADIR, index_col="DATE", parse_dates=True)

# Define columns to be used for features and target
cols = ["USD_W", "DT_W", "V_W"]

# Ensure the "USD_W" column contains numeric values
X["USD_W"] = pd.to_numeric(X["USD_W"], errors="coerce")

# Create the target column "Target" based on the next day's price movement
X["Target"] = (X["USD_W"].pct_change().shift(-1) > 0).astype(int)
X.dropna(inplace=True)

# # Split the data into training and testing sets
# X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

# # Scale the selected columns
# scaler = StandardScaler().fit(X_train[cols])
# X_train[cols] = scaler.transform(X_train[cols])
# X_test[cols] = scaler.transform(X_test[cols])

# # Store the data in the data dictionary
# data["Thoi Gian"] = X_train

# Define model parameters and compile the model

seq_len = 60
batch_size = 128
n_epochs = 10
n_features = len(cols)

# Build and compile the CNN model
model = cnnpred_1d(seq_len, n_features)
model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["acc", f1macro])
model.summary()

# Set up callbacks and fit the model
checkpoint_path = "./cp1d-{epoch}-{val_f1macro:.2f}.h5"
callbacks = [
    ModelCheckpoint(checkpoint_path,
                    monitor='val_f1macro', mode="max",
                    verbose=0, save_best_only=True, save_weights_only=False, save_freq="epoch")
]
model.fit(datagen(data, seq_len, batch_size, "Target", "train"),
          validation_data=datagen(
              data, seq_len, batch_size, "Target", "valid"),
          epochs=n_epochs, steps_per_epoch=100, validation_steps=10, verbose=1, callbacks=callbacks)

# Prepare test data
test_data, test_target = testgen({"Thoi Gian": X_test}, seq_len, "Target")

# Test the model
test_out = model.predict(test_data)
test_pred = (test_out > 0.5).astype(int)
print("Accuracy:", accuracy_score(test_pred, test_target))
print("MAE:", mean_absolute_error(test_pred, test_target))
print("F1 Score:", f1_score(test_pred, test_target))
