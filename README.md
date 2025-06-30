# 📈 Sales Prediction Streamlit App

A machine learning-powered Streamlit web app that predicts **Units Sold** using historical sales data and user-input features like region, item type, order date, and more.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 Features

- 📅 Extracts time-based features like Month, Year, Day of Week  
- 🧠 Trained using Linear Regression from Scikit-learn  
- 🔢 Encodes categorical variables for accurate prediction  
- 📊 Predicts **Units Sold** for any selected combination of features  
- 🌐 User-friendly web interface built with Streamlit  

---

## 📂 Folder Structure

sales-prediction-streamlit/<br>
├── app.py # Main Streamlit app<br>
├── 5000 Sales Records.csv # Dataset file<br>
├── requirements.txt # Required Python libraries<br>
└── README.md # This file<br>


---

## 📊 Dataset Used

- **Name:** 5000 Sales Records  
- **Source:** Publicly available (e.g., Kaggle)  
- **Features Used:**  
  - Date-based: `Month`, `Year`, `DayOfWeek`  
  - Categorical: `Region`, `Country`, `Item Type`, `Sales Channel`, `Order Priority`

---

## 🛠️ How to Run

### ⬇️ Clone or Download

```bash
git clone https://github.com/your-username/sales-prediction-streamlit.git
cd sales-prediction-streamlit
