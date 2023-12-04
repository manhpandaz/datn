import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# Đọc dữ liệu
data = pd.read_excel("data/DulieuVang_dau_Tygia.xlsx")

# Chuyển đổi trường ngày/tháng/năm thành datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Xác định ngày bắt đầu và kết thúc của dữ liệu tự động
start_date = data['DATE'].min()
end_date = data['DATE'].max()

# Trực quan hóa chỉ số kinh tế theo thời gian
plt.figure(figsize=(12, 6))
plt.plot(data['DATE'], data['USD_W'], label='USD_W', color='blue')
plt.plot(data['DATE'], data['DT_W'], label='DT_W', color='green')
plt.plot(data['DATE'], data['V_W'], label='V_W', color='red')
plt.title('Chỉ số kinh tế theo thời gian')
plt.xlabel('Thời gian')
plt.ylabel('Chỉ số')
plt.legend()

# Đặt định dạng cho trục thời gian theo năm
years = YearLocator()
date_format = DateFormatter("%Y")
plt.gca().xaxis.set_major_locator(years)
plt.gca().xaxis.set_major_formatter(date_format)
plt.gcf().autofmt_xdate()

# Hiển thị biểu đồ
plt.show()

# Tiền xử lý dữ liệu
# Chọn các chỉ số kinh tế cần sử dụng
data = data[['DATE', 'USD_W', 'DT_W', 'V_W']]
# Chuẩn bị dữ liệu chuỗi thời gian
sequence_length = 100  # Độ dài của chuỗi thời gian
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data.iloc[i:(i + sequence_length)].drop('DATE', axis=1).values)
    y.append(data.iloc[i + sequence_length]['USD_W'])
X = np.array(X)
y = np.array(y)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler()
X_train = scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(
    X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(
    X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(3))

# Biên dịch và huấn luyện mô hình
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Đánh giá mô hình
train_loss = model.evaluate(X_train, y_train, verbose=1)
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Loss: {test_loss:.4f}')

# Dự báo
predictions = model.predict(X_test)

# kiểm tra độ dài y,xs


def check_test_set_match(y_test, X_test):
    return len(y_test) == X_test.shape[0]


match = check_test_set_match(y_test, X_test)
columns = ['USD_W', 'DT_W', 'V_W']
# print(y_test.shape)
# print(predictions.shape)
if match:
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(columns):
        plt.plot(data['DATE'][:len(predictions)], y_test,
                 label=f'{col} thực tế', color='blue')
        plt.plot(data['DATE'][:len(predictions)],
                 predictions[:, i], label=f'{col} dự báo', color='red')
        plt.title(f'So sánh {col} thực tế và Dự báo')
        plt.xlabel('Thời gian')
        plt.ylabel(col)
        plt.legend()
        plt.gca().xaxis.set_major_locator(years)
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xlim(start_date, end_date)
        plt.gcf().autofmt_xdate()
        plt.show()
else:
    print("Tập dữ liệu kiểm tra không khớp với tập dữ liệu kiểm tra.")
