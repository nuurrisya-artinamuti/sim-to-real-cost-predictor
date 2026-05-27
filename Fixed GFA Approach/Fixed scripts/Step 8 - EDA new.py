#Load dataset
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\rynsc\OneDrive - Politecnico di Milano\File di Nuurrisya Artinamuti - Project Construction Mgmt G31\Fixed GFA Approach\Fixed datasets\New Output 1 - Combined Dataset.csv")

print(df.head())
print(df.describe())


#Plot 1: Area vs Cost
plt.figure()
plt.scatter(df["gfa_m2"], df["actual_cost"])
plt.xlabel("Area (m²)")
plt.ylabel("Actual Cost (€)")
plt.title("Area vs Cost")
plt.show()


#Plot 2: Cost by City
plt.figure(figsize=(10, 6))   # ← ADD THIS

df.groupby("city")["actual_cost"].mean().plot(kind="bar")

plt.title("Average Cost by City")
plt.ylabel("Cost (€)")
plt.xticks(rotation=0)        # keep labels horizontal
plt.tight_layout()            # VERY IMPORTANT

plt.show()


#Plot 3: Overbudget Distribution
plt.hist(df["overbudget_percent"], bins=50)
plt.title("Overbudget Distribution (%)")
plt.xlabel("Overbudget %")
plt.show()


#Plot 4: Rooms vs Cost
plt.scatter(df["rooms"], df["actual_cost"])
plt.xlabel("Rooms")
plt.ylabel("Cost (€)")
plt.title("Rooms vs Cost")
plt.show()


#Plot 5: Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
