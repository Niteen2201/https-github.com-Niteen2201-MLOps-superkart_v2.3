
import pandas as pd

RAW_PATH = "superkart_project/data/SuperKart.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Expected columns for SuperKart dataset
expected_columns = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Store_Sales_Total"
]

# Validate columns
missing = [col for col in expected_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}"
    )

# Basic validation checks
if df.empty:
    raise ValueError("Dataset is empty.")

# Check target column
if "Product_Store_Sales_Total" not in df.columns:
    raise ValueError(
        "Target column 'Product_Store_Sales_Total' not found."
    )

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumns:")
print(list(df.columns))

print("\nTarget Variable Statistics:")
print(df["Product_Store_Sales_Total"].describe())

print("\nMissing Values:")
print(df.isnull().sum())
