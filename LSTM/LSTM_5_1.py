import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')
data['DATE'] = pd.to_datetime(data['DATE'])
data = data.set_index('DATE')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
train_data = scaled_data[:int(len(scaled_data)*0.8), :]
test_data = scaled_data[int(len(scaled_data)*0.8):, :]

# Prepare input and output for the model
train_X = train_data[:, :-1]
train_y = train_data[:, -1]
test_X = test_data[:, :-1]
test_y = test_data[:, -1]

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(
    train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_X, train_y, epochs=200, verbose=0)

# Predict on the test set
predictions = model.predict(test_X)
predictions = scaler.inverse_transform(predictions)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(test_y, predictions)
print('Mean Squared Error:', mse)
