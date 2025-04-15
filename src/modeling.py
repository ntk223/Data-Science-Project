import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_model(X_train, y_train, model_type='xgboost'):
    if model_type == 'xgboost':
        model = xgb.XGBRegressor()
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Model type not supported")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Dự đoán và tính toán độ chính xác
    y_pred = model.predict(X_test)
    
    # Đánh giá mô hình bằng RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    return rmse
