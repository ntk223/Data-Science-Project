import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    return train, test

def basic_cleaning(df):
    # Xử lý đơn giản ví dụ: điền NaN bằng 0
    df = df.fillna(0)
    return df

def scale_features(train_df, test_df, numeric_cols):
    scaler = StandardScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
    return train_df, test_df
