# PCM Group Project G31: "Sim-to-Real" Cost Predictor
## Project Overview
This project transforms static floor plan data into a dynamic financial decision-support tool for construction cost estimation. It takes an input SVG floor plan, the target city, and the target year, and outputs a 90% Confidence Interval of the predicted construction cost.

## Current Architecture
The application follows a Two-Step Prediction Logic:
1. **Module 0 (`src/extractor.py`)**: Parses uploaded SVG floor plans to extract geometric features (Gross Floor Area in m² and room counts).
2. **Module 1 (`src/forecaster.py`)**: Uses a regression model trained strictly on 2020–2024 historical data to predict the base market price (€/m²) for a specific city and future year (e.g., 2026).
3. **Module 2 (`src/predictor.py`)**: Uses a Random Forest model trained on simulated "Actual Costs" to calculate the final project cost. It extracts the 90% Confidence Interval directly from the variance of the individual decision tree estimators in the forest, eliminating the need for a Monte Carlo simulation.
4. **Entry Point (`main.py`)**: The main pipeline tying the modules together.

## Immediate Mission
The agent's next tasks are to:
1. Create the offline training script (`scripts/train_market_model.py`) to train `models/market_model.pkl` using 2020-2024 data.
2. Create the offline training script (`scripts/train_cost_model.py`) to generate synthetic cost data and train `models/cost_model.pkl` using a Random Forest algorithm.
3. Build the Streamlit dashboard (`app.py`) to replace the CLI output in `main.py`, displaying the S-Curve of the predicted costs.
