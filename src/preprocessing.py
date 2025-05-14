import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

def handle_missing(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna("None")
    return df

def add_features(df):
    df = df.copy()

    # Tổng diện tích
    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(df.columns):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Tuổi nhà, tuổi sửa chữa
    if {'YearBuilt', 'YearRemodAdd', 'YrSold'}.issubset(df.columns):
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']

    # Các đặc trưng tương tác
    if {'OverallQual', 'LotArea'}.issubset(df.columns):
        df['OverallQual_LotArea'] = df['OverallQual'] * df['LotArea']

    if {'GrLivArea', 'LotArea'}.issubset(df.columns):
        df['GrLivArea_PerLotArea'] = df['GrLivArea'] / df['LotArea']

    # Các đặc trưng nhị phân
    if 'TotalBsmtSF' in df.columns:
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)

    if 'GarageArea' in df.columns:
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)

    if 'Fireplaces' in df.columns:
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

    if 'YearBuilt' in df.columns and 'YearRemodAdd' in df.columns:
        df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)

    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['IsNew'] = (df['YearBuilt'] == df['YrSold']).astype(int)

    return df


# Xử lý skew cho các biến số
def fix_skewed_features(df, skew_thresh=0.75):
    df = df.copy()
    numeric_feats = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Id'], errors='ignore')
    skewness = numeric_feats.apply(lambda x: skew(x.dropna()))
    skewed_cols = skewness[skewness > skew_thresh].index
    df[skewed_cols] = df[skewed_cols].apply(lambda x: np.log1p(x))
    return df

# Sửa phần encode an toàn hơn
def encode_categorical(df):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
    return df

def preprocess(df):
    df = handle_missing(df)
    df = add_features(df)
    df = fix_skewed_features(df)  # Bổ sung dòng này
    df = encode_categorical(df)
    return df
