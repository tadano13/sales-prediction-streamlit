import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("5000 Sales Records.csv")

# Preprocess data
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Month'] = data['Order Date'].dt.month
data['DayOfWeek'] = data['Order Date'].dt.dayofweek
data['Year'] = data['Order Date'].dt.year

# Encode categorical columns
categorical_cols = ['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Train the model
features = ['Month', 'DayOfWeek', 'Year'] + categorical_cols
X = data[features]
y = data['Units Sold']
model = LinearRegression()
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Sales Predictor", layout="centered")
st.title("📈 Sales Prediction App")
st.write("Predict expected **Units Sold** using future sales data.")

# Inputs
st.subheader("📝 Input Features")
month = st.selectbox("Month", sorted(data["Month"].unique()))
day_of_week = st.selectbox("Day of Week (0=Mon, 6=Sun)", sorted(data["DayOfWeek"].unique()))
year = st.selectbox("Year", sorted(data["Year"].unique()) + [2025, 2026])  # add future years

region = st.selectbox("Region", label_encoders['Region'].classes_)
country = st.selectbox("Country", label_encoders['Country'].classes_)
item_type = st.selectbox("Item Type", label_encoders['Item Type'].classes_)
sales_channel = st.selectbox("Sales Channel", label_encoders['Sales Channel'].classes_)
order_priority = st.selectbox("Order Priority", label_encoders['Order Priority'].classes_)

# Button to trigger prediction
if st.button("🔮 Predict Future Sales"):
    # Encode user input
    region_enc = label_encoders['Region'].transform([region])[0]
    country_enc = label_encoders['Country'].transform([country])[0]
    item_type_enc = label_encoders['Item Type'].transform([item_type])[0]
    sales_channel_enc = label_encoders['Sales Channel'].transform([sales_channel])[0]
    order_priority_enc = label_encoders['Order Priority'].transform([order_priority])[0]

    # Prepare input
    input_data = np.array([[month, day_of_week, year, region_enc, country_enc,
                            item_type_enc, sales_channel_enc, order_priority_enc]])

    # Predict and show result
    predicted_units = model.predict(input_data)[0]
    st.success(f"📦 Predicted Units Sold: **{predicted_units:.0f}**")
