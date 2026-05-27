import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Output 1 - Combined Dataset.csv"
OUTPUT_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Output 2 - ML"

df = pd.read_csv(DATA_FILE, encoding="cp1252")

df = df.drop(columns=["Unnamed: 13", "Unnamed: 14"], errors="ignore")

print("Dataset loaded.")
print(df.head())


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

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


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 4. PREPROCESSING
# ============================================================

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


# ============================================================
# 5. LINEAR REGRESSION
# ============================================================

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)


# ============================================================
# 6. RANDOM FOREST
# ============================================================

forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ))
    ]
)

forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)


# ============================================================
# 7. METRICS
# ============================================================

def print_metrics(model_name, actual, predicted):
    r2 = r2_score(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5

    print(f"\n{model_name}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: €{mae:,.2f}")
    print(f"RMSE: €{rmse:,.2f}")

    return r2, mae, rmse


linear_r2, linear_mae, linear_rmse = print_metrics(
    "LINEAR REGRESSION RESULTS",
    y_test,
    linear_predictions
)

forest_r2, forest_mae, forest_rmse = print_metrics(
    "RANDOM FOREST RESULTS",
    y_test,
    forest_predictions
)


# ============================================================
# 8. SAVE PREDICTIONS
# ============================================================

results = X_test.copy()
results["actual_cost"] = y_test
results["linear_prediction"] = linear_predictions
results["forest_prediction"] = forest_predictions
results["linear_error"] = results["actual_cost"] - results["linear_prediction"]
results["forest_error"] = results["actual_cost"] - results["forest_prediction"]

results.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved predictions to: {OUTPUT_FILE}")


# ============================================================
# 9. PLOT 1 — ACTUAL VS PREDICTED: LINEAR REGRESSION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(results["actual_cost"], results["linear_prediction"], alpha=0.4)

min_value = min(results["actual_cost"].min(), results["linear_prediction"].min())
max_value = max(results["actual_cost"].max(), results["linear_prediction"].max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual Cost (€)")
plt.ylabel("Predicted Cost (€)")
plt.title("Linear Regression: Actual vs Predicted Cost")
plt.tight_layout()
plt.show()


# ============================================================
# 10. PLOT 2 — ACTUAL VS PREDICTED: RANDOM FOREST
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(results["actual_cost"], results["forest_prediction"], alpha=0.4)

min_value = min(results["actual_cost"].min(), results["forest_prediction"].min())
max_value = max(results["actual_cost"].max(), results["forest_prediction"].max())

plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

plt.xlabel("Actual Cost (€)")
plt.ylabel("Predicted Cost (€)")
plt.title("Random Forest: Actual vs Predicted Cost")
plt.tight_layout()
plt.show()


# ============================================================
# 11. PLOT 3 — MODEL COMPARISON: MAE
# ============================================================

plt.figure(figsize=(7, 5))

model_names = ["Linear Regression", "Random Forest"]
mae_values = [linear_mae, forest_mae]

plt.bar(model_names, mae_values)

plt.ylabel("Mean Absolute Error (€)")
plt.title("Model Comparison: MAE")
plt.tight_layout()
plt.show()


# ============================================================
# 12. PLOT 4 — MODEL COMPARISON: R2
# ============================================================

plt.figure(figsize=(7, 5))

r2_values = [linear_r2, forest_r2]

plt.bar(model_names, r2_values)

plt.ylabel("R² Score")
plt.title("Model Comparison: R²")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ============================================================
# 13. PLOT 5 — ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.hist(results["forest_error"], bins=40)

plt.xlabel("Prediction Error (€)")
plt.ylabel("Frequency")
plt.title("Random Forest Prediction Error Distribution")
plt.tight_layout()
plt.show()