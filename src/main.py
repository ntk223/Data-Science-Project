import pandas as pd
import sys
sys.path.append("../src")
from preprocessing import preprocess
# Đọc dữ liệu train và test
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

from catboost import CatBoostRegressor
# Tiền xử lý dữ liệu (đảm bảo bạn đã xử lý dữ liệu train và test giống nhau)
df_train_processed = preprocess(df_train)  # Hàm tiền xử lý bạn đã tạo
df_test_processed = preprocess(df_test)

# Chọn đặc trưng (features) và target từ df_train
X_train = df_train_processed.drop('SalePrice', axis=1)  # Loại bỏ cột SalePrice
y_train = df_train_processed['SalePrice']

# Huấn luyện mô hình
model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0)
model.fit(X_train, y_train)

# Dự đoán trên dữ liệu kiểm thử
X_test = df_test_processed  # Đảm bảo rằng df_test_processed đã được tiền xử lý
y_pred = model.predict(X_test)

# Tạo DataFrame kết quả
submission = pd.DataFrame({
    'Id': df_test['Id'],   # Cột Id từ dữ liệu test
    'SalePrice': y_pred    # Cột dự đoán
})

# Lưu kết quả vào file CSV
submission.to_csv('submission.csv', index=False)
