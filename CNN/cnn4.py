from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from numpy import array
from sklearn.model_selection import train_test_split
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Set plot attributes
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['text.color'] = 'k'
matplotlib.rcParams['figure.figsize'] = 10, 7

# Load Dataset
dataset = pd.read_csv("F:/DATN/data/DulieuVang_dau_Tygia.csv")

# Visualize Time Series Dataset
# plt.plot(dataset)
# plt.show()

# Decompose different Time Series elements (e.g., trend, seasonality, residual)
decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
decomposition.plot()
plt.show()

# Auto-correlation plot
autocorrelation_plot(dataset)
plt.show()

# Split data into training and testing sets
X_train, X_test = train_test_split(dataset, test_size=0.3, random_state=42)

# Split a multivariate sequence into samples

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences) - 1:
            break
        seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# Choose a number of time steps
n_steps = 3

# Convert data into input/output
X_train, y_train = split_sequences(
    X_train[["USD_W", "DT_W", "V_W"]].values, n_steps)
X_test, y_test = split_sequences(
    X_test[["USD_W", "DT_W", "V_W"]].values, n_steps)

# Reshape data
n_features = X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# Build the CNN model

model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu',
          input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(n_features, activation='linear'))

model.compile(optimizer='adam', loss='mse')

# Fit the model
model.fit(X_train, y_train, epochs=1000, verbose=1)

# Make predictions on the test data
y_pred = model.predict(X_test, verbose=1)

# Report performance for each variable

for i in range(n_features):
    var_name = ["USD_W", "DT_W", "V_W"][i]
    r_squared = r2_score(y_test[:, i], y_pred[:, i])
    print(f"R squared ({var_name}): {r_squared}")

    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    print(f'Mean Absolute Error ({var_name}): {mae}')

    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    print(f'Mean Squared Error ({var_name}): {mse}')

    msle = mean_squared_log_error(y_test[:, i], y_pred[:, i])
    print(f'Mean Squared Log Error ({var_name}): {msle}')

    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error ({var_name}): {rmse}')

import matplotlib.pyplot as plt

# Plot the actual and predicted values for each variable
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(y_test[:, 0], label="Actual (USD_W)", color="blue")
plt.plot(y_pred[:, 0], label="Predicted (USD_W)", color="red")
plt.title("USD_W - Actual vs. Predicted")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(y_test[:, 1], label="Actual (DT_W)", color="blue")
plt.plot(y_pred[:, 1], label="Predicted (DT_W)", color="red")
plt.title("DT_W - Actual vs. Predicted")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(y_test[:, 2], label="Actual (V_W)", color="blue")
plt.plot(y_pred[:, 2], label="Predicted (V_W)", color="red")
plt.title("V_W - Actual vs. Predicted")
plt.legend()

plt.tight_layout()
plt.show()
