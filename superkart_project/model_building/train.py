
# Data manipulation
import pandas as pd
import numpy as np

# Preprocessing and Modeling
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV

# Evaluation Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# XGBoost
import xgboost as xgb

# Model Serialization and Tracking
import joblib
import mlflow

# MLflow Configuration
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SuperKart-Sales-Forecasting-Experiment")

# Load Prepared Data
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")

ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

# Numerical Features
numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year"
]

# Categorical Features
categorical_features = [
    "Product_Id",
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type"
]

# Preprocessing Pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42
)

# Hyperparameter Grid
param_grid = {
    "xgbregressor__n_estimators": [100, 200, 300],
    "xgbregressor__max_depth": [3, 5, 7],
    "xgbregressor__learning_rate": [0.01, 0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0]
}

# Pipeline
model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# Start MLflow Run
with mlflow.start_run():

    # Hyperparameter Tuning
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log All Trials
    results = grid_search.cv_results_

    for i in range(len(results["params"])):

         with mlflow.start_run(nested=True):

            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_cv_score",
                float(results["mean_test_score"][i])
            )

            mlflow.log_metric(
                "std_cv_score",
                float(results["std_test_score"][i])
            )

    # Log Best Parameters
    mlflow.log_params(grid_search.best_params_)

    # Best Model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Training Metrics (calculated compatible with scikit-learn 1.6+)
    train_rmse = np.sqrt(
        mean_squared_error(ytrain, y_pred_train)
    )

    train_mae = mean_absolute_error(
        ytrain,
        y_pred_train
    )

    train_r2 = r2_score(
        ytrain,
        y_pred_train
    )

    # Test Metrics
    test_rmse = np.sqrt(
        mean_squared_error(ytest, y_pred_test)
    )

    test_mae = mean_absolute_error(
        ytest,
        y_pred_test
    )

    test_r2 = r2_score(
        ytest,
        y_pred_test
    )

    # Log Metrics to MLflow
    mlflow.log_metrics({
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2
    })

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nModel Performance")
    print(f"Train RMSE : {train_rmse:.2f}")
    print(f"Test RMSE  : {test_rmse:.2f}")

    print(f"Train MAE  : {train_mae:.2f}")
    print(f"Test MAE   : {test_mae:.2f}")

    print(f"Train R²   : {train_r2:.4f}")
    print(f"Test R²    : {test_r2:.4f}")

    # Save Model
    model_path = (
        "superkart_project/deployment/"
        "best_superkart_sales_model_v1.joblib"
    )

    joblib.dump(best_model, model_path)

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(f"\nModel saved to {model_path}")
