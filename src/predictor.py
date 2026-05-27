"""
Module 2: Cost Predictor.
Loads the Linear Regression cost model, predicts the final cost, and extracts
the confidence interval analytically from the model's MAE using a normal distribution.
"""

import os
import joblib
import numpy as np
import pandas as pd
import scipy.stats


class CostPredictor:
    """Predicts final construction cost and calculates the Confidence Interval using Linear Regression."""
    
    def __init__(self, model_path: str = None, mae_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_path is None:
            model_path = os.path.join(base_dir, "Fixed GFA Approach", "final_linear_cost_model.pkl")
            
        if mae_path is None:
            mae_path = os.path.join(base_dir, "Fixed GFA Approach", "final_model_mae.txt")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cost model binary not found at: {model_path}."
            )
            
        self.model = joblib.load(model_path)
        
        if os.path.exists(mae_path):
            with open(mae_path, "r") as f:
                self.mae = float(f.read().strip())
        else:
            # Fallback to default MAE from training
            self.mae = 30882.89
            
        self.features = [
            "gfa_m2",
            "rooms",
            "city",
            "year",
            "price_m2",
            "base_cost",
            "complexity_factor"
        ]

    def predict_cost(
        self,
        gfa: float,
        rooms: int,
        base_market_price: float,
        city: str,
        year: int,
        confidence_level: float = 90.0
    ) -> dict:
        """
        Predicts total construction cost and computes a custom Confidence Interval.
        
        Parameters:
            gfa (float): Gross Floor Area in m2.
            rooms (int): Number of rooms.
            base_market_price (float): Predicted base market price in €/m2.
            city (str): Name of the city (Espoo, Helsinki, Oulu, Tampere, Vantaa).
            year (int): Target Completion Year (e.g. 2026).
            confidence_level (float): Confidence interval level in percent (e.g. 90.0).
            
        Returns:
            dict: A dictionary containing:
                - 'predicted_cost' (float): Main Linear Regression model prediction.
                - 'ci_lower' (float): Lower confidence interval boundary.
                - 'ci_upper' (float): Upper confidence interval boundary.
                - 'tree_predictions' (list[float]): Sorted simulated predictions from normal error CDF.
        """
        city_clean = city.strip().capitalize()
        base_cost = gfa * base_market_price
        complexity_factor = 1.0 + (rooms * 0.01)

        # Create input features dataframe matching friend's training features
        input_data = pd.DataFrame([{
            "gfa_m2": float(gfa),
            "rooms": int(rooms),
            "city": city_clean,
            "year": int(year),
            "price_m2": float(base_market_price),
            "base_cost": float(base_cost),
            "complexity_factor": float(complexity_factor)
        }])[self.features]
        
        # Calculate prediction using the linear regression pipeline
        predicted_cost = float(self.model.predict(input_data)[0])
        
        # Calculate normal distribution standard deviation from MAE:
        # MAE = sigma * sqrt(2/pi) => sigma = MAE * sqrt(pi/2)
        sigma = self.mae * np.sqrt(np.pi / 2.0)
        
        # Calculate dynamic percentiles based on chosen confidence level
        lower_p = (100.0 - confidence_level) / 2.0
        upper_p = 100.0 - lower_p
        
        ci_lower = float(scipy.stats.norm.ppf(lower_p / 100.0, loc=predicted_cost, scale=sigma))
        ci_upper = float(scipy.stats.norm.ppf(upper_p / 100.0, loc=predicted_cost, scale=sigma))
        
        # Ensure bounds are not negative
        ci_lower = max(0.0, ci_lower)
        ci_upper = max(0.0, ci_upper)
        
        # Generate 150 points along the CDF (from 0.1% to 99.9%) to draw a smooth S-Curve
        quantiles = np.linspace(0.001, 0.999, 150)
        tree_predictions_sorted = [
            max(0.0, float(scipy.stats.norm.ppf(q, loc=predicted_cost, scale=sigma)))
            for q in quantiles
        ]
        
        return {
            "predicted_cost": predicted_cost,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "tree_predictions": tree_predictions_sorted
        }
