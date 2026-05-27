# Generate Offline ML Training Scripts

1. Read the `project_specification.md` to understand the data flow.
2. Create a new directory called `scripts/`.
3. Generate `scripts/train_market_model.py`:
   - Implement a script that creates a synthetic pandas DataFrame simulating real estate data for Helsinki, Espoo, and Vantaa between 2020-2024.
   - Train a Scikit-Learn regression model on this data.
   - Save the model using joblib to `models/market_model.pkl`.
4. Generate `scripts/train_cost_model.py`:
   - Implement a script that generates synthetic floor plan features (GFA, rooms) and simulates "Actual Costs" applying complexity multipliers.
   - Train a Scikit-Learn `RandomForestRegressor` on this dataset.
   - Save the model using joblib to `models/cost_model.pkl`.
5. Ensure both scripts are fully documented and executable from the terminal.
