import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS — CHANGE FOLDERS IF NEEDED
# ============================================================

CUBICASA_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Input 1 - GFA.csv"

FINNISH_PRICE_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Input 2 - Cities Finland.xlsx"

OUTPUT_FILE = r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Output 1 - Combined Dataset.csv"


# ============================================================
# 2. LOAD DATASETS
# ============================================================

plans = pd.read_csv(CUBICASA_FILE)

prices = pd.read_excel(
    FINNISH_PRICE_FILE,
    sheet_name="City - Year - Price"
)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

plans.columns = plans.columns.str.strip()
prices.columns = prices.columns.str.strip()


# Rename columns to simpler names
prices = prices.rename(columns={
    "City": "city",
    "Year": "year",
    "Price per square meter (EUR/m2)": "price_m2"
})

plans = plans.rename(columns={
    "included_space_count": "rooms"
})


# ============================================================
# 4. KEEP ONLY USEFUL COLUMNS
# ============================================================

plans = plans[["plan_id", "gfa_m2", "rooms"]]
prices = prices[["city", "year", "price_m2"]]


# ============================================================
# 5. REMOVE BAD VALUES
# ============================================================

plans = plans.dropna(subset=["gfa_m2"])
plans = plans[(plans["gfa_m2"] >= 20) & (plans["gfa_m2"] <= 300)]

prices = prices.dropna(subset=["city", "year", "price_m2"])


# ============================================================
# 6. CROSS-JOIN DATASETS
# Every plan receives every city/year option
# ============================================================

plans["key"] = 1
prices["key"] = 1

combined = pd.merge(plans, prices, on="key").drop("key", axis=1)


# ============================================================
# 7. CALCULATE BASE COST
# ============================================================

combined["base_cost"] = combined["gfa_m2"] * combined["price_m2"]


# ============================================================
# 8. ADD SYNTHETIC REALISM FOR ML
# ============================================================

np.random.seed(42)

combined["complexity_factor"] = 1 + (combined["rooms"] * 0.01)

combined["uncertainty_factor"] = np.random.uniform(
    0.90,
    1.20,
    size=len(combined)
)

combined["actual_cost"] = (
    combined["base_cost"]
    * combined["complexity_factor"]
    * combined["uncertainty_factor"]
)

combined["cost_adjustment_factor"] = (
    combined["actual_cost"] / combined["base_cost"]
)

combined["overbudget_amount"] = (
    combined["actual_cost"] - combined["base_cost"]
)

combined["overbudget_percent"] = (
    combined["overbudget_amount"] / combined["base_cost"]
) * 100


# ============================================================
# 9. SAVE FINAL STEP 6 DATASET
# ============================================================

combined.to_csv(OUTPUT_FILE, index=False)

print("Done.")
print(f"Saved combined dataset to: {OUTPUT_FILE}")
print(combined.head())
print(f"Rows created: {len(combined)}")