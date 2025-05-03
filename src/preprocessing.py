import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    # Thêm TotalSF
    if {'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'}.issubset(df.columns):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    # Thêm Age và RemodelAge
    if {'YearBuilt', 'YearRemodAdd', 'YrSold'}.issubset(df.columns):
        df['Age'] = df['YrSold'] - df['YearBuilt']
        df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    # Thêm OverallQual_LotArea
    if {'OverallQual', 'LotArea'}.issubset(df.columns):
        df['OverallQual_LotArea'] = df['OverallQual'] * df['LotArea']
    # Thêm GrLivArea_PerLotArea
    if {'GrLivArea', 'LotArea'}.issubset(df.columns):
        df['GrLivArea_PerLotArea'] = df['GrLivArea'] / df['LotArea']
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
    df = encode_categorical(df)
    return df
