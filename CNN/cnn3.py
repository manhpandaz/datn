def TSF_using_CNN():
    # ---------------------------------
    # Load packages
    # ---------------------------------
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm
    from pandas.plotting import autocorrelation_plot
    from sklearn.model_selection import train_test_split
    # ---------------------------------
    # set plot attributes
    # ---------------------------------
    plt.style.use('fivethirtyeight')
    matplotlib.rcParams['axes.labelsize'] = 10
    matplotlib.rcParams['xtick.labelsize'] = 10
    matplotlib.rcParams['ytick.labelsize'] = 10
    matplotlib.rcParams['text.color'] = 'k'
    matplotlib.rcParams['figure.figsize'] = 10, 7

    # ---------------------------------
    # Load Dataset
    # ---------------------------------
    dataset = pd.read_csv("F:/DATN/data/DulieuVang_dau_Tygia.csv")

    # dataset = pd.read_csv("BJsales_in_R.csv")
    # dataset = list(dataset["Temp"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # print dataset
    print()
    print(dataset.shape)
    # print(dataset.head(25))

    # ---------------------------------
    # Visualise Time Series Dataset
    # ---------------------------------
    # Plot Dataset
    plt.plot(dataset)
    plt.show()
    # Decompose diffentent Time Series elements e.g. trand, seasonality, Residual ... ...
    decomposition = sm.tsa.seasonal_decompose(dataset, model='additive')
    decomposition.plot()
    plt.show()

    # Auto-correlation plot
    autocorrelation_plot(dataset)
    plt.show()

    # split a multivariate sequence into samples
    from numpy import array

    def split_sequences(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences)-1:
                break
            # gather input and output parts of the pattern

            seq_x, seq_y = sequences[i:end_ix], sequences[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return array(X), array(y)

    # choose a number of time steps
    n_steps = 3
    # convert into input/output
    X, y = split_sequences(dataset, n_steps)

    print(X.shape)
    print(y)

    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # from numpy import array
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers.convolutional import Conv1D
    from keras.layers.convolutional import MaxPooling1D

    # define model - using CNN model
    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu',
              input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(X, y, epochs=1000, verbose=1)

    # demonstrate prediction
    dataset = pd.read_csv("F:/DATN/data/DulieuVang_dau_Tygia.csv")
    dataset = dataset[["USD_W", "DT_W", "V_W"]]

    # convert into input/output
    X, y = split_sequences(dataset, n_steps)

    x_input = X.reshape((X.shape[0], X.shape[1], n_features))
    yhat = model.predict(x_input, verbose=1)
    # print(yhat)

    df_pred = pd.DataFrame.from_records(yhat, columns=['predicted'])
    df_pred = df_pred.reset_index(drop=True)

    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # report performance
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

    coefficient_of_dermination = r2_score(df_actual, df_pred)
    print("R squared: ", coefficient_of_dermination)

    mae = mean_absolute_error(df_actual, df_pred)
    print('The Mean Absolute Error of our forecasts is {}'.format(round(mae, 2)))

    mse = mean_squared_error(df_actual, df_pred)
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

    msle = mean_squared_log_error(df_actual, df_pred)
    print('The Mean Squared Log Error of our forecasts is {}'.format(round(msle, 2)))

    print('The Root Mean Squared Error of our forecasts is {}'.format(
        round(np.sqrt(mse), 2)))

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('DATE')
    ax.set_ylabel('USD_W')
    plt.legend()
    plt.show()

    # ---------------------------------------------------------------------------
    # Future Predictions
    predictions = model.predict(x_input, verbose=1)
    future_time_steps = 7
    x1 = x_input[-1:, :, :]   # take the last input
    p1 = predictions[-1:]   # take the last prediction

    for i in range(future_time_steps):

        x2 = np.array([[x1[0][1], x1[0][2], p1]])
        p2 = model.predict(x2, verbose=1)
        predictions = np.append(predictions, p2)

        x1 = x2
        p1 = p2

    yhat = predictions
    yhat = np.reshape(yhat, (-1, 1))

    df_pred = pd.DataFrame.from_records(yhat, columns=['predicted'])
    df_pred = df_pred.reset_index(drop=True)

    df_actual = dataset[n_steps:len(dataset)]
    df_actual = df_actual.reset_index(drop=True)

    # plot
    ax = df_actual.plot(label='Observed', figsize=(9, 7))
    df_pred.plot(ax=ax)
    ax.set_xlabel('DATE')
    ax.set_ylabel('USD_W')
    plt.legend()
    plt.show()
    # ---------------------------------------------------------------------------


TSF_using_CNN()
