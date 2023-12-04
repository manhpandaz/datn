import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.df_selected = self.load_data()
        self.setup_ui()

    def load_data(self):
        df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')
        selected_columns = ['DATE', 'USD_W', 'DT_W', 'V_W']
        df_selected = df[selected_columns]
        df_selected['DATE'] = pd.to_datetime(df_selected['DATE'])
        df_selected.set_index('DATE', inplace=True)
        return df_selected

    def setup_ui(self):
        self.setWindowTitle("Financial Analysis")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.v_layout = QVBoxLayout(self.central_widget)

        self.result_label = QLabel("Results will be displayed here.", self)
        self.v_layout.addWidget(self.result_label)

        self.calculate_button = QPushButton("Calculate Market Indices", self)
        self.calculate_button.clicked.connect(self.calculate_market_indices)
        self.v_layout.addWidget(self.calculate_button)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.v_layout.addWidget(self.canvas)

    def calculate_market_indices(self):
        weekly_volatility_usd_w_lstm, _, _ = self.calculate_weekly_volatility_lstm()
        lstm_predictions = self.build_lstm_model()

        if lstm_predictions is not None:
            arima_predictions = self.build_arima_model()
            decision_tree_predictions = self.build_decision_tree_model()
            random_forest_predictions = self.build_random_forest_model()

            arima_rmse = np.sqrt(mean_squared_error(
                self.df_selected['USD_W'].iloc[-len(arima_predictions):], arima_predictions))
            lstm_rmse = np.sqrt(mean_squared_error(
                self.df_selected['USD_W'].iloc[-len(lstm_predictions):], lstm_predictions))
            decision_tree_rmse = np.sqrt(mean_squared_error(
                self.df_selected['USD_W'].iloc[-len(decision_tree_predictions):], decision_tree_predictions))
            random_forest_rmse = np.sqrt(mean_squared_error(
                self.df_selected['USD_W'].iloc[-len(random_forest_predictions):], random_forest_predictions))

            print(f"ARIMA RMSE: {arima_rmse}")
            print(f"LSTM RMSE: {lstm_rmse}")
            print(f"Decision Tree RMSE: {decision_tree_rmse}")
            print(f"Random Forest RMSE: {random_forest_rmse}")

            self.plot_data(weekly_volatility_usd_w_lstm,
                           "Biến động hàng tuần - USD_W (LSTM)")
            self.plot_data(arima_predictions, "ARIMA Predictions - USD_W")
            self.plot_data(decision_tree_predictions,
                           "Decision Tree Predictions - USD_W")
            self.plot_data(random_forest_predictions,
                           "Random Forest Predictions - USD_W")
            self.plot_data(lstm_predictions, "LSTM Predictions - USD_W")

            result_text = f"Biến động hàng tuần - USD_W (LSTM): {weekly_volatility_usd_w_lstm:.4f}\nARIMA RMSE: {arima_rmse:.4f}\nLSTM RMSE: {lstm_rmse:.4f}\nDecision Tree RMSE: {decision_tree_rmse:.4f}\nRandom Forest RMSE: {random_forest_rmse:.4f}"

            self.result_label.setText(result_text)

    def calculate_weekly_volatility_lstm(self):
        try:
            log_returns_usd_w = np.log1p(
                self.df_selected['USD_W'].pct_change())
            weekly_volatility_usd_w = log_returns_usd_w.rolling(
                window=5).std().mean() * np.sqrt(252)

            log_returns_dt_w = np.log1p(self.df_selected['DT_W'].pct_change())
            weekly_volatility_dt_w = log_returns_dt_w.rolling(
                window=5).std().mean() * np.sqrt(252)

            log_returns_v_w = np.log1p(self.df_selected['V_W'].pct_change())
            weekly_volatility_v_w = log_returns_v_w.rolling(
                window=5).std().mean() * np.sqrt(252)

            fig, axes = plt.subplots(3, 1, figsize=(12, 18))

            axes[0].plot(weekly_volatility_usd_w,
                         label='Weekly Volatility - USD_W', marker='o')
            axes[0].set_title('Biến động hàng tuần - USD_W')
            axes[0].set_xlabel('Time Steps')
            axes[0].set_ylabel('Value')
            axes[0].legend()

            axes[1].plot(weekly_volatility_dt_w,
                         label='Weekly Volatility - DT_W', marker='o')
            axes[1].set_title('Biến động hàng tuần - DT_W')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Value')
            axes[1].legend()

            axes[2].plot(weekly_volatility_v_w,
                         label='Weekly Volatility - V_W', marker='o')
            axes[2].set_title('Biến động hàng tuần - V_W')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Value')
            axes[2].legend()

            plt.tight_layout()
            plt.show()

            return weekly_volatility_usd_w.iloc[-1], weekly_volatility_dt_w.iloc[-1], weekly_volatility_v_w.iloc[-1]

        except Exception as e:
            print(f"Error in calculate_weekly_volatility: {e}")
            return None, None, None

    def build_lstm_model(self):
        try:
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(self.df_selected)

            train_size = int(len(df_scaled) * 0.8)
            train_data, test_data = df_scaled[:
                                              train_size], df_scaled[train_size:]

            time_steps = 10
            X_train, y_train = self.prepare_data(train_data, time_steps)
            X_test, y_test = self.prepare_data(test_data, time_steps)

            model = Sequential()
            model.add(LSTM(units=50, activation='relu', input_shape=(
                X_train.shape[1], X_train.shape[2])))
            model.add(Dense(units=3))
            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=50, batch_size=32,
                      validation_data=(X_test, y_test), shuffle=False)

            mse = model.evaluate(X_test, y_test)
            print(f"Mean Squared Error on Test Data (LSTM): {mse}")

            y_pred = model.predict(X_test)
            y_pred_inverse = scaler.inverse_transform(y_pred)

            return y_pred_inverse

        except Exception as e:
            print(f"Error in build_lstm_model: {e}")
            return None

    def build_arima_model(self):
        history = list(self.df_selected['USD_W'][:-20])
        predictions = []
        for t in range(20):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            history.append(yhat)
        return np.array(predictions)

    def build_decision_tree_model(self):
        model = DecisionTreeRegressor()
        X_train, y_train = self.prepare_data(self.df_selected['USD_W'], 10)
        model.fit(X_train, y_train)
        decision_tree_predictions = model.predict(X_train[-20:])
        return decision_tree_predictions

    def build_random_forest_model(self):
        model = RandomForestRegressor()
        X_train, y_train = self.prepare_data(self.df_selected['USD_W'], 10)
        model.fit(X_train, y_train)
        random_forest_predictions = model.predict(X_train[-20:])
        return random_forest_predictions

    def plot_data(self, data, title):
        self.canvas.axes.clear()
        self.canvas.axes.plot(data, label=title, marker='o')
        self.canvas.axes.set_title(title)
        self.canvas.axes.set_xlabel('Time Steps')
        self.canvas.axes.set_ylabel('Value')
        self.canvas.axes.legend()
        self.canvas.draw()

    def prepare_data(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MyMainWindow()
    mainWin.show()
    sys.exit(app.exec_())
