import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data from the source
df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Print the total number of data points
print(f"Total number of data points: {len(df)}")

# Select relevant columns
selected_columns = ['DATE', 'USD_W', 'DT_W', 'V_W']
df_selected = df[selected_columns]

# Convert the 'DATE' column to datetime and set it as the index
df_selected['DATE'] = pd.to_datetime(df_selected['DATE'])
df_selected.set_index('DATE', inplace=True)

# Normalize data using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_selected)

# Split data into training and testing sets (80-20 ratio)
train_data, test_data = train_test_split(
    df_scaled, test_size=0.2, shuffle=False)

# Prepare Data for LSTM


def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


time_steps = 10
X_train, y_train = prepare_data(train_data, time_steps)
X_test, y_test = prepare_data(test_data, time_steps)

# Build and Train LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dense(units=3))  # 3 units for 3 target variables
model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.fit(X_train, y_train, epochs=10, batch_size=32,
               validation_data=(X_test, y_test), shuffle=False)

# Evaluate LSTM Model on Test Data
mse_lstm = model_lstm.evaluate(X_test, y_test)
print(f"Mean Squared Error (LSTM) on Test Data: {mse_lstm}")

# Predictions on test data for LSTM
y_pred_lstm = model_lstm.predict(X_test)

# Calculate MSE for each target variable
mse_lstm_usd = mean_squared_error(y_test[:, 0], y_pred_lstm[:, 0])
mse_lstm_dt = mean_squared_error(y_test[:, 1], y_pred_lstm[:, 1])
mse_lstm_v = mean_squared_error(y_test[:, 2], y_pred_lstm[:, 2])

print(f"Mean Squared Error (LSTM) on Test Data - USD_W: {mse_lstm_usd}")
print(f"Mean Squared Error (LSTM) on Test Data - DT_W: {mse_lstm_dt}")
print(f"Mean Squared Error (LSTM) on Test Data - V_W: {mse_lstm_v}")

# Build and Evaluate Decision Tree Model
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

y_pred_dt = model_dt.predict(X_test.reshape((X_test.shape[0], -1)))
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f"Mean Squared Error (Decision Tree) on Test Data: {mse_dt}")

# Calculate MSE for each target variable for Decision Tree
mse_dt_usd = mean_squared_error(y_test[:, 0], y_pred_dt[:, 0])
mse_dt_dt = mean_squared_error(y_test[:, 1], y_pred_dt[:, 1])
mse_dt_v = mean_squared_error(y_test[:, 2], y_pred_dt[:, 2])

print(f"Mean Squared Error (Decision Tree) on Test Data - USD_W: {mse_dt_usd}")
print(f"Mean Squared Error (Decision Tree) on Test Data - DT_W: {mse_dt_dt}")
print(f"Mean Squared Error (Decision Tree) on Test Data - V_W: {mse_dt_v}")

# Build and Evaluate Random Forest Model
model_rf = RandomForestRegressor()
model_rf.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

y_pred_rf = model_rf.predict(X_test.reshape((X_test.shape[0], -1)))
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Mean Squared Error (Random Forest) on Test Data: {mse_rf}")

# Calculate MSE for each target variable for Random Forest
mse_rf_usd = mean_squared_error(y_test[:, 0], y_pred_rf[:, 0])
mse_rf_dt = mean_squared_error(y_test[:, 1], y_pred_rf[:, 1])
mse_rf_v = mean_squared_error(y_test[:, 2], y_pred_rf[:, 2])

print(f"Mean Squared Error (Random Forest) on Test Data - USD_W: {mse_rf_usd}")
print(f"Mean Squared Error (Random Forest) on Test Data - DT_W: {mse_rf_dt}")
print(f"Mean Squared Error (Random Forest) on Test Data - V_W: {mse_rf_v}")

# Visualize Results for 'USD_W' using LSTM
y_test_inverse_lstm = scaler.inverse_transform(y_test)
y_pred_inverse_lstm = scaler.inverse_transform(y_pred_lstm)

time_steps_test = df_selected.index[train_data.shape[0] + time_steps:]

plt.figure(figsize=(12, 6))
plt.plot(time_steps_test,
         y_test_inverse_lstm[:, 0], label='Actual (USD_W)', marker='o')
plt.plot(time_steps_test,
         y_pred_inverse_lstm[:, 0], label='Predicted (USD_W)', marker='o')
plt.title('USD_W - Actual vs. Predicted (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.show()

# Similar visualizations for 'DT_W' and 'V_W' using LSTM, Decision Tree, and Random Forest
# ...
