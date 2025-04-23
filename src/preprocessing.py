import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def handle_missing(df):
    df = df.copy()  # Đảm bảo làm việc trên bản sao

    # Thay vì dùng inplace=True, gán kết quả trực tiếp
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    # Cập nhật các cột khác
    for col in ['Alley', 'FireplaceQu', 'GarageType']:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    return df


# Encode categorical features using Label Encoding
def encode_categorical(df):
    df = df.copy()
    label_cols = ['MSZoning', 'Street', 'SaleCondition']
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

# Add new features
# SF = Total Square Footage
# TotalSF = Total Basement + 1st Floor + 2nd Floor
def add_features(df):
    df = df.copy()
    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(df.columns):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    return df

def preprocess(df):
    df = handle_missing(df)
    df = encode_categorical(df)
    df = add_features(df)
    return df
