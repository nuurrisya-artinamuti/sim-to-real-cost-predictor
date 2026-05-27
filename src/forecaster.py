"""
Module 1: Market Price Forecaster.
Loads the regression model and predicts the base market price (€/m²) for a target city and year.
"""

import os
import joblib
import pandas as pd


class MarketForecaster:
    """Predicts base market price per square meter (€/m²) using a trained linear regression model."""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Resolve default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "market_model.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Market model binary not found at: {model_path}. "
                "Please run scripts/train_market_model.py first."
            )
            
        self.model = joblib.load(model_path)
        self.features = ["year", "city_Espoo", "city_Helsinki", "city_Oulu", "city_Tampere", "city_Vantaa"]

    def forecast_price(self, city: str, year: int) -> float:
        """
        Forecasts base market price for a given city and year.
        
        Parameters:
            city (str): Name of the city (Espoo, Helsinki, Oulu, Tampere, Vantaa).
            year (int): Target forecasting year (e.g. 2026).
            
        Returns:
            float: Predicted base market price in €/m².
        """
        city_clean = city.strip().capitalize()
        valid_cities = ["Espoo", "Helsinki", "Oulu", "Tampere", "Vantaa"]
        
        if city_clean not in valid_cities:
            raise ValueError(f"Invalid city '{city}'. Supported cities are: {', '.join(valid_cities)}.")

        # Create input features row with dummy variables
        input_data = {
            "year": [float(year)],
            "city_Espoo": [1.0 if city_clean == "Espoo" else 0.0],
            "city_Helsinki": [1.0 if city_clean == "Helsinki" else 0.0],
            "city_Oulu": [1.0 if city_clean == "Oulu" else 0.0],
            "city_Tampere": [1.0 if city_clean == "Tampere" else 0.0],
            "city_Vantaa": [1.0 if city_clean == "Vantaa" else 0.0]
        }
        
        df = pd.DataFrame(input_data)[self.features]
        
        # Predict base price
        prediction = self.model.predict(df)[0]
        
        # Ensure we return a positive price
        return max(100.0, float(prediction))
