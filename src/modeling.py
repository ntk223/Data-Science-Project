from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error
import numpy as np

def train_catboost(X_train, y_train):
    model = CatBoostRegressor(iterations=1500, learning_rate=0.1, depth=4, random_seed=42, verbose=0)
    model.fit(X_train, y_train)
    return model

def tune_catboost_grid_search(X_train, y_train):
    model = CatBoostRegressor(random_state=42)
    
    param_grid = {
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    return grid_search.best_estimator_

# Hàm tối ưu hóa tham số với RandomizedSearchCV
def tune_catboost_random_search(X_train, y_train):
    model = CatBoostRegressor(random_state=42)

    param_dist = {
        'iterations': [500, 1000, 1500],
        'learning_rate': uniform(0.01, 0.1),  # phân phối ngẫu nhiên cho learning rate
        'depth': [4, 6, 8]
    }

    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=100, cv=3, n_jobs=-1, verbose=1)
    random_search.fit(X_train, y_train)

    print("Best parameters:", random_search.best_params_)
    print("Best score:", random_search.best_score_)
    return random_search.best_estimator_

def evaluate(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)    # tự lấy căn bậc 2 của MSE
    return rmse


