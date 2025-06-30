import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 🎯 Page configuration
st.set_page_config(page_title="🔮 Sales Prediction App", layout="centered")

# 🚀 Title & Description
st.title("📈 Smart Sales Forecast")
st.markdown("Welcome to the **AI-powered sales predictor**! 🧠💼\n\
Use the sliders and dropdowns below to explore how different factors impact **Units Sold**.")

# 📥 Load dataset
data = pd.read_csv("5000 Sales Records.csv")

# 🧹 Preprocess data
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Month'] = data['Order Date'].dt.month
data['DayOfWeek'] = data['Order Date'].dt.dayofweek
data['Year'] = data['Order Date'].dt.year

categorical_cols = ['Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 🧠 Train model
features = ['Month', 'DayOfWeek', 'Year'] + categorical_cols
X = data[features]
y = data['Units Sold']
model = LinearRegression()
model.fit(X, y)

# 🎛️ Input Features
st.markdown("---")
st.subheader("🧾 Fill in your sales context:")

col1, col2, col3 = st.columns(3)
with col1:
    month = st.selectbox("📅 Month", sorted(data["Month"].unique()))
with col2:
    day_of_week = st.selectbox("📆 Day of Week (0=Mon)", sorted(data["DayOfWeek"].unique()))
with col3:
    year = st.selectbox("📈 Year", sorted(data["Year"].unique()) + [2025, 2026])

region = st.selectbox("🌍 Region", label_encoders['Region'].classes_)
country = st.selectbox("🏳️ Country", label_encoders['Country'].classes_)
item_type = st.selectbox("📦 Item Type", label_encoders['Item Type'].classes_)
sales_channel = st.selectbox("🛒 Sales Channel", label_encoders['Sales Channel'].classes_)
order_priority = st.selectbox("⚡ Order Priority", label_encoders['Order Priority'].classes_)

st.markdown("---")

# 🔘 Prediction Button
if st.button("🔮 Predict Future Sales"):
    # Encode inputs
    region_enc = label_encoders['Region'].transform([region])[0]
    country_enc = label_encoders['Country'].transform([country])[0]
    item_type_enc = label_encoders['Item Type'].transform([item_type])[0]
    sales_channel_enc = label_encoders['Sales Channel'].transform([sales_channel])[0]
    order_priority_enc = label_encoders['Order Priority'].transform([order_priority])[0]

    input_data = np.array([[month, day_of_week, year, region_enc, country_enc,
                            item_type_enc, sales_channel_enc, order_priority_enc]])

    predicted_units = model.predict(input_data)[0]

    st.success(f"✅ Estimated Units Sold: **{predicted_units:.0f}** 🚚")

    st.balloons()

else:
    st.info("👆 Select your values above and click **Predict Future Sales** to get started.")
