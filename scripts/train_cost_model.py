#!/usr/bin/env python
"""
Offline training script for the Cost Prediction model.
Generates synthetic floor plan features (GFA, rooms) and simulates "Actual Costs"
using PERT/Beta distributions.
Trains a Random Forest Regressor and saves it to models/cost_model.pkl.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def random_pert(low, likely, high, size=1):
    """Generates samples from a PERT (Beta-based) distribution."""
    # Ensure numpy arrays/floats
    low = np.asarray(low)
    likely = np.asarray(likely)
    high = np.asarray(high)

    # Calculate mean and standard shape parameters alpha, beta
    mean = (low + 4.0 * likely + high) / 6.0

    # Avoid divide by zero if low == high
    mask = (high == low)
    alpha = np.where(mask, 1.0, 1.0 + 4.0 * (mean - low) / (high - low))
    beta = np.where(mask, 1.0, 1.0 + 4.0 * (high - mean) / (high - low))

    # Generate beta samples
    beta_samples = np.random.beta(alpha, beta, size=size)

    # Scale and shift to [low, high]
    return low + beta_samples * (high - low)


def generate_synthetic_cost_data(num_samples: int = 5000) -> pd.DataFrame:
    """Generates synthetic dataset of floor plans and actual construction costs."""
    np.random.seed(42)

    # 1. Generate GFA (Gross Floor Area in m2) using PERT
    gfa = random_pert(low=30.0, likely=85.0, high=300.0, size=num_samples)

    # 2. Generate room counts based on GFA
    # Smaller areas have fewer rooms, larger areas have more
    rooms_low = np.maximum(1, np.round(gfa / 45.0))
    rooms_likely = np.maximum(1, np.round(gfa / 25.0))
    rooms_high = np.maximum(2, np.round(gfa / 15.0))

    rooms = []
    for rl, rlk, rh in zip(rooms_low, rooms_likely, rooms_high):
        r = np.round(random_pert(rl, rlk, rh, size=1)[0])
        rooms.append(int(r))
    rooms = np.array(rooms)

    # 3. Simulate predicted base market price (from Module 1)
    # Range expanded to support Tampere and Oulu (typically €1500 to €6000)
    base_market_price = random_pert(low=1500.0, likely=3800.0, high=6000.0, size=num_samples)

    # 4. Calculate actual cost using complexity multipliers and PERT noise
    # Base cost = GFA * base_market_price
    base_cost = gfa * base_market_price

    # Complexity multiplier based on density of rooms (more rooms per m2 = higher cost density)
    # e.g., standard layout is 1 room per 25 m2. Deviation from this scales cost.
    density_ratio = rooms / gfa
    complexity_multiplier = 1.0 + (density_ratio - 1.0 / 25.0) * 2.0
    # Keep complexity reasonable
    complexity_multiplier = np.clip(complexity_multiplier, 0.8, 1.5)

    # Scale base cost by complexity and PERT noise factor (e.g. fluctuation between 85% and 115%)
    noise_factor = random_pert(low=0.85, likely=1.0, high=1.15, size=num_samples)
    actual_costs = base_cost * complexity_multiplier * noise_factor

    df = pd.DataFrame({
        "gfa": gfa,
        "rooms": rooms,
        "base_market_price": base_market_price,
        "actual_cost": actual_costs
    })
    return df


def main():
    print("Generating synthetic actual cost data...")
    df = generate_synthetic_cost_data()

    print(f"Generated {len(df)} samples.")

    # Features and Target
    features = ["gfa", "rooms", "base_market_price"]
    X = df[features]
    y = df["actual_cost"]

    print("Training RandomForestRegressor cost model...")
    # Using 150 estimators to ensure smooth confidence interval boundaries
    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        min_samples_leaf=5,
        max_depth=15,
        n_jobs=-1
    )
    model.fit(X, y)

    # Check R2 score to ensure the model trained correctly
    r2_score = model.score(X, y)
    print(f"Model trained. R2 score on training data: {r2_score:.4f}")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "cost_model.pkl")
    joblib.dump(model, model_path)
    print(f"Cost model saved successfully to: {model_path}")


if __name__ == "__main__":
    main()
