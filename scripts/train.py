#!/usr/bin/env python
"""
Unified Offline Training Script for the Sim-to-Real Cost Predictor.
Trains:
1. The baseline Market Price forecaster (Linear Regression on historical data).
2. The final Construction Cost model (Linear Regression Pipeline on combined dataset).
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def train_market_model(excel_path: str, output_dir: str):
    """Trains the pooled linear regression model for baseline market prices."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Historical market data Excel file not found at: {excel_path}")
        
    print(f"\n--- Training Market Price Model ---")
    print(f"Loading market data from: {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name="City - Year - Price")
    
    # Rename columns to standard names
    df = df.rename(columns={
        "Year": "year",
        "City": "city",
        "Price per square meter (EUR/m2)": "price_per_m2"
    })
    
    # Filter strictly for 2020 <= year <= 2024
    df = df[(df["year"] >= 2020) & (df["year"] <= 2024)].copy()
    print(f"Ingested {len(df)} aggregated market data points (2020-2024).")
    
    # One-hot encode the city column
    df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=False)
    features = ["year", "city_Espoo", "city_Helsinki", "city_Oulu", "city_Tampere", "city_Vantaa"]
    
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].astype(float)
            
    X = df[features]
    y = df["price_per_m2"]
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Model coefficients:")
    for col, coef in zip(features, model.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "market_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved market forecaster model to: {model_path}")


def train_cost_model(csv_path: str, model_path: str, mae_path: str):
    """Trains the linear regression pipeline for project construction costs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combined dataset CSV not found at: {csv_path}")
        
    print(f"\n--- Training Construction Cost Model ---")
    print(f"Loading dataset from: {csv_path}...")
    df = pd.read_csv(csv_path, encoding="cp1252")
    df = df.drop(columns=["Unnamed: 13", "Unnamed: 14"], errors="ignore")
    
    features = [
        "gfa_m2",
        "rooms",
        "city",
        "year",
        "price_m2",
        "base_cost",
        "complexity_factor"
    ]
    target = "actual_cost"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    categorical_features = ["city"]
    numeric_features = [
        "gfa_m2",
        "rooms",
        "year",
        "price_m2",
        "base_cost",
        "complexity_factor"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("city_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features)
        ]
    )
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    
    # Save model and MAE metric
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    with open(mae_path, "w") as f:
        f.write(f"{mae:.2f}\n")
        
    print(f"Metrics:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAE: €{mae:,.2f}")
    print(f"  RMSE: €{rmse:,.2f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved MAE to: {mae_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser = argparse.ArgumentParser(description="Train all predictor models.")
    parser.add_argument(
        "--market-excel",
        default=os.path.join(base_dir, "Input 2 - Cities Finland.xlsx"),
        help="Path to market price excel dataset."
    )
    parser.add_argument(
        "--cost-csv",
        default=os.path.join(base_dir, "data", "combined_dataset.csv"),
        help="Path to combined project cost CSV dataset."
    )
    
    args = parser.parse_args()
    
    # Target files locations
    models_dir = os.path.join(base_dir, "models")
    cost_model_path = os.path.join(models_dir, "final_linear_cost_model.pkl")
    cost_mae_path = os.path.join(models_dir, "final_model_mae.txt")
    
    try:
        train_market_model(args.market_excel, models_dir)
        train_cost_model(args.cost_csv, cost_model_path, cost_mae_path)
        print("\nAll models trained and saved successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")


if __name__ == "__main__":
    main()
