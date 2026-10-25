
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("superkart_project/data/SuperKart.csv")

# Optional: Remove identifiers that don't contribute to prediction
# Keep them if you plan to use Target Encoding or Frequency Encoding later

target = "Product_Store_Sales_Total"

X = df.drop(columns=[target])
y = df[target]

# Train-Test Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Save datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared successfully.")
print(f"Training samples : {len(Xtrain)}")
print(f"Testing samples  : {len(Xtest)}")

print("\nFeature Columns:")
print(list(X.columns))
