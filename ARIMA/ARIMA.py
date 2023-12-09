import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')

data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index('DATE', inplace=True)


columns_to_predict = ['USD_W', 'DT_W', 'V_W']

# Lặp qua từng cột dữ liệu
for y_col in columns_to_predict:
    y = data[y_col]

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    train, test = train_test_split(y, train_size=0.8)

    # Tìm kiếm bộ ba tham số tốt nhất
    autoarima_model = pm.auto_arima(
        train, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
    order = autoarima_model.order

    # Xây dựng và huấn luyện mô hình ARIMA với bộ ba tham số tốt nhất
    arima_model = ARIMA(train, order=order)
    fit_model = arima_model.fit()

    # Dự đoán trên tập kiểm tra
    predictions = fit_model.predict(
        start=len(train), end=len(train) + len(test) - 1, typ='levels')

    # Đánh giá hiệu suất mô hình
    mse = mean_squared_error(test, predictions)
    print(f'Mean Squared Error for {y_col}: {mse}')

    # Vẽ đồ thị
    plt.plot(data.index[-len(test):], test, label=f'Actual {y_col}')
    plt.plot(data.index[-len(test):], predictions,
             label=f'ARIMA Predictions {y_col}')
    plt.xlabel('Date')
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
