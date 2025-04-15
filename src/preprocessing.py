import pandas as pd

def load_data():
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv('data/train.csv')
    return df

def preprocess_data(df):
    # Chỉ chọn các cột có kiểu dữ liệu là số
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Điền giá trị missing (NaN) bằng giá trị trung bình cho các cột số
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Bạn có thể thực hiện thêm các bước tiền xử lý khác ở đây (nếu cần)

    # Tách dữ liệu thành X (features) và y (target) nếu cần
    X = df.drop('target_column', axis=1)  # Thay 'target_column' bằng tên cột mục tiêu của bạn
    y = df['target_column']  # Thay 'target_column' bằng tên cột mục tiêu của bạn

    return X, y
