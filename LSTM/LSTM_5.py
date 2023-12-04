import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Đọc dữ liệu từ nguồn
        self.df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')
        selected_columns = ['DATE', 'USD_W', 'DT_W', 'V_W']
        self.df_selected = self.df[selected_columns]
        self.df_selected['DATE'] = pd.to_datetime(self.df_selected['DATE'])
        self.df_selected.set_index('DATE', inplace=True)

        # Chuẩn hóa dữ liệu sử dụng Min-Max Scaler
        self.scaler = MinMaxScaler()
        df_scaled = self.scaler.fit_transform(self.df_selected)

        # Chia thành tập huấn luyện và tập kiểm tra (tỷ lệ 80-20)
        train_size = int(len(df_scaled) * 0.8)
        train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

        # Chuẩn Bị Dữ Liệu cho LSTM
        time_steps = 10
        X_train, y_train = self.prepare_data(train_data, time_steps)
        self.X_test, self.y_test = self.prepare_data(test_data, time_steps)

        # Xây dựng mô hình LSTM
        self.model = Sequential()
        self.model.add(LSTM(units=50, activation='relu', input_shape=(
            X_train.shape[1], X_train.shape[2])))
        self.model.add(Dense(units=3))  # 3 units cho 3 trường dữ liệu
        self.model.compile(optimizer='adam', loss='mse')

        # Huấn luyện mô hình
        self.model.fit(X_train, y_train, epochs=50, batch_size=32,
                       validation_data=(self.X_test, self.y_test), shuffle=False)

        # UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        # Button để kích hoạt dự đoán
        self.predict_button = QPushButton('Dự đoán', self)
        self.predict_button.clicked.connect(self.predict_and_plot)

        # Label để hiển thị kết quả
        self.result_label = QLabel('Kết quả dự đoán sẽ được hiển thị ở đây.', self)

        # Canvas để vẽ biểu đồ
        self.canvas_usd_w = FigureCanvas(plt.Figure())
        self.canvas_dt_w = FigureCanvas(plt.Figure())
        self.canvas_v_w = FigureCanvas(plt.Figure())

        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.canvas_usd_w)
        layout.addWidget(self.canvas_dt_w)
        layout.addWidget(self.canvas_v_w)

        self.central_widget.setLayout(layout)

    def prepare_data(self, data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps])
        return np.array(X), np.array(y)

    def predict_and_plot(self):
        # Dự đoán trên tập kiểm tra
        y_pred = self.model.predict(self.X_test)

        # Đảo ngược chuẩn hóa để có giá trị gốc
        y_test_inverse = self.scaler.inverse_transform(self.y_test)
        y_pred_inverse = self.scaler.inverse_transform(y_pred)

        # Tạo mảng các giá trị thời gian
        time_steps_test = self.df_selected.index[len(self.df_selected) - len(self.y_test):]

        # Trực quan hóa kết quả cho cột USD_W
        self.plot_result(self.canvas_usd_w, time_steps_test, y_test_inverse[:, 0], y_pred_inverse[:, 0], 'USD_W')

        # Trực quan hóa kết quả cho cột DT_W
        self.plot_result(self.canvas_dt_w, time_steps_test, y_test_inverse[:, 1], y_pred_inverse[:, 1], 'DT_W')

        # Trực quan hóa kết quả cho cột V_W
        self.plot_result(self.canvas_v_w, time_steps_test, y_test_inverse[:, 2], y_pred_inverse[:, 2], 'V_W')

        # Hiển thị kết quả trên label
        self.result_label.setText(f"Kết quả dự đoán đã được hiển thị trên các biểu đồ.")

    def plot_result(self, canvas, time_steps, actual, predicted, label):
        ax = canvas.figure.add_subplot(111)
        ax.plot(time_steps, actual, label='Actual', marker='o')
        ax.plot(time_steps, predicted, label='Predicted', marker='o')
        ax.set_title(f'{label} - Actual vs. Predicted')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend()

        # Cập nhật và hiển thị
        canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MyMainWindow()
    mainWin.show()
    sys.exit(app.exec_())
