"""
Module 2: Cost Predictor.
Loads the Random Forest cost model, predicts the final cost, and extracts
the 90% confidence interval from the variance of the tree estimators.
"""

import os
import joblib
import numpy as np
import pandas as pd


class CostPredictor:
    """Predicts final construction cost and calculates the 90% Confidence Interval using Random Forest."""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Resolve default path relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "cost_model.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Cost model binary not found at: {model_path}. "
                "Please run scripts/train_cost_model.py first."
            )
            
        self.model = joblib.load(model_path)
        self.features = ["gfa", "rooms", "base_market_price"]

    def predict_cost(self, gfa: float, rooms: int, base_market_price: float, confidence_level: float = 90.0) -> dict:
        """
        Predicts total construction cost and computes a custom Confidence Interval.
        
        Parameters:
            gfa (float): Gross Floor Area in m2.
            rooms (int): Number of rooms.
            base_market_price (float): Predicted base market price in €/m2.
            confidence_level (float): Confidence interval level in percent (e.g. 90.0).
            
        Returns:
            dict: A dictionary containing:
                - 'predicted_cost' (float): Main Random Forest model prediction.
                - 'ci_lower' (float): Lower percentile of estimator predictions.
                - 'ci_upper' (float): Upper percentile of estimator predictions.
                - 'tree_predictions' (list[float]): Sorted raw predictions from all estimators.
        """
        # Create feature dataframe
        input_data = pd.DataFrame(
            [[gfa, rooms, base_market_price]],
            columns=self.features
        )
        
        # Calculate main prediction
        predicted_cost = float(self.model.predict(input_data)[0])
        
        # Extract individual tree estimator predictions for the Confidence Interval
        # Rule constraint: pull percentiles from self.model.estimators_
        tree_predictions = []
        for estimator in self.model.estimators_:
            # Each estimator prediction
            pred_val = float(estimator.predict(input_data.values)[0])
            tree_predictions.append(pred_val)
            
        # Compute dynamic percentiles based on chosen confidence level
        lower_p = (100.0 - confidence_level) / 2.0
        upper_p = 100.0 - lower_p
        
        ci_lower = float(np.percentile(tree_predictions, lower_p))
        ci_upper = float(np.percentile(tree_predictions, upper_p))
        
        # Sort tree predictions for easy S-Curve cumulative probability construction
        tree_predictions_sorted = sorted(tree_predictions)
        
        return {
            "predicted_cost": predicted_cost,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "tree_predictions": tree_predictions_sorted
        }
