import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATA_FILE = r"c:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Output 1 - Combined Dataset.csv"

MODEL_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\final_linear_cost_model.pkl"


df = pd.read_csv(DATA_FILE, encoding="cp1252")
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
    X,
    y,
    test_size=0.2,
    random_state=42
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

joblib.dump(model, MODEL_FILE)

print("Final Linear Regression model saved.")
print(f"R2: {r2:.4f}")
print(f"MAE: €{mae:,.2f}")
print(f"RMSE: €{rmse:,.2f}")
print(f"Saved model to: {MODEL_FILE}")