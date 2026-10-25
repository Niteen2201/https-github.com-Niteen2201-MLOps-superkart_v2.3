
import os
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model_path = os.path.join(
    os.path.dirname(__file__),
    "best_superkart_sales_model_v1.joblib"
)

model = joblib.load(model_path)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SuperKart Sales Forecasting")
st.write(
    "Enter product and store details below to forecast "
    "the expected sales revenue."
)

# --------------------------------------------------
# Product Information
# --------------------------------------------------

st.subheader("🛒 Product Information")

product_id = st.text_input(
    "Product ID",
    value="FD1001"
)

product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=10.0
)

product_sugar_content = st.selectbox(
    "Sugar Content",
    [
        "Low Sugar",
        "Regular",
        "No Sugar"
    ]
)

product_allocated_area = st.number_input(
    "Allocated Display Area Ratio",
    min_value=0.0,
    max_value=1.0,
    value=0.10
)

product_type = st.selectbox(
    "Product Type",
    [
        "Dairy",
        "Snack Foods",
        "Soft Drinks",
        "Hard Drinks",
        "Baking Goods",
        "Bread",
        "Breakfast",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Household",
        "Canned",
        "Health and Hygiene",
        "Meat",
        "Seafood",
        "Starchy Foods",
        "Others"
    ]
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=1.0,
    value=100.0
)

# --------------------------------------------------
# Store Information
# --------------------------------------------------

st.subheader("🏪 Store Information")

store_id = st.text_input(
    "Store ID",
    value="OUT001"
)

store_establishment_year = st.number_input(
    "Store Establishment Year",
    min_value=1980,
    max_value=2026,
    value=2005
)

store_size = st.selectbox(
    "Store Size",
    [
        "High",
        "Medium",
        "Low"
    ]
)

store_location_city_type = st.selectbox(
    "City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3"
    ]
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Supermarket Type 1",
        "Supermarket Type 2",
        "Food Mart"
    ]
)

# --------------------------------------------------
# Prediction Input
# --------------------------------------------------

input_data = pd.DataFrame([{
    "Product_Id": product_id,
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Forecast Sales"):

    predicted_sales = model.predict(input_data)[0]

    st.success(
        f"Predicted Sales Revenue: ₹ {predicted_sales:,.2f}"
    )

    st.metric(
        label="Forecasted Revenue",
        value=f"₹ {predicted_sales:,.2f}"
    )
