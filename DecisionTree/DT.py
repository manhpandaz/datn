import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_excel('data/DulieuVang_dau_Tygia.xlsx')
# Chia dữ liệu
X = df[['USD_W', 'DT_W', 'V_W']]
y = df[['USD_W', 'DT_W', 'V_W']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình Decision Tree
model_dt = DecisionTreeRegressor()
model_dt.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm thử
y_pred = model_dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error DecisionTree: {mse}')

# Dự đoán chỉ số cho dữ liệu mới
# new_data = pd.DataFrame({'USD_W': [new_usd_value], 'DT_W': [new_dt_value], 'V_W': [new_v_value]})
# predicted_values = model_dt.predict(new_data)
# print('Predicted Values:', predicted_values)
