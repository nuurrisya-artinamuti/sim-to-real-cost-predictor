#!/usr/bin/env python
"""
Offline training script for the Market price model.
Loads real estate data for Helsinki, Espoo, Vantaa, Tampere, and Oulu from Excel.
Trains a pooled linear regression model to predict base price (€/m²) based on city and year.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


import argparse


def load_market_data(excel_path: str) -> pd.DataFrame:
    """Loads and cleans real estate transaction data from Excel."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Historical market data Excel file not found at: {excel_path}")
        
    print(f"Loading market data from: {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name="City - Year - Price")
    
    # Rename columns to lowercase/standard names
    df = df.rename(columns={
        "Year": "year",
        "City": "city",
        "Price per square meter (EUR/m2)": "price_per_m2"
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Train offline market regression model.")
    parser.add_argument(
        "--excel",
        default="Input 2 - Cities Finland.xlsx",
        help="Path to the Excel file containing historical data."
    )
    args = parser.parse_args()

    df = load_market_data(args.excel)

    # Rule constraint: Filter strictly for 2020 <= year <= 2024
    print("Filtering data ingestion strictly for 2020 <= year <= 2024...")
    df = df[(df["year"] >= 2020) & (df["year"] <= 2024)].copy()
    print(f"Loaded {len(df)} aggregated market data points.")

    # Prepare features: One-hot encode the 'city' column
    df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=False)

    # Columns expected for training
    features = ["year", "city_Espoo", "city_Helsinki", "city_Oulu", "city_Tampere", "city_Vantaa"]

    # Ensure all columns exist (just in case)
    for col in features:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].astype(float)

    X = df[features]
    y = df["price_per_m2"]

    print("Training pooled market forecaster linear regression model...")
    model = LinearRegression()
    model.fit(X, y)

    # Print coefs for inspection
    print("Model coefficients:")
    for col, coef in zip(features, model.coef_):
        print(f"  {col}: {coef:.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "market_model.pkl")
    joblib.dump(model, model_path)
    print(f"Market forecaster model saved successfully to: {model_path}")


if __name__ == "__main__":
    main()
