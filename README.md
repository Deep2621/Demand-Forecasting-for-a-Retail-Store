📊 Retail Store Demand Forecasting 📉

This project is a Retail Store Demand Forecasting system built using Machine Learning, LSTM, and Streamlit. The model predicts future sales based on historical data and provides an interactive visualization of sales trends.


---

📌 Features

✅ Forecast future sales using LSTM neural networks
✅ Interactive data visualization with Streamlit
✅ Scalable model with preprocessed time-series data
✅ Real-time model training and prediction
✅ Supports multiple time aggregation (daily, weekly, monthly, yearly)


---

📂 Project Structure

├── app.py                     # Streamlit web application
├── retail_forecasting.ipynb    # Jupyter Notebook for model training
├── retail_store_inventory.csv  # Sample dataset
├── lstm_demand_forecasting.h5  # Saved LSTM model
├── README.md                   # Project documentation


---

🚀 How It Works

The system consists of two main components:

1️⃣ Data Preprocessing & Model Training (Jupyter Notebook)
2️⃣ Web App Deployment (Streamlit)


---

🔹 1️⃣ Data Preprocessing & Model Training (Jupyter Notebook)

This step involves data cleaning, feature extraction, and training an LSTM model.

🔸 Step 1: Load Dataset

import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("retail_store_inventory.csv")
df["Date"] = pd.to_datetime(df["Date"])

# Aggregate daily sales
df_daily = df.groupby("Date")["Units Sold"].sum().reset_index()

🔸 Step 2: Data Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df_daily["Scaled_Sales"] = scaler.fit_transform(df_daily[["Units Sold"]])

🔸 Step 3: Prepare Time-Series Sequences

sequence_length = 30
X, y = [], []
for i in range(len(df_daily) - sequence_length):
    X.append(df_daily["Scaled_Sales"].iloc[i : i + sequence_length].values)
    y.append(df_daily["Scaled_Sales"].iloc[i + sequence_length])

X, y = np.array(X), np.array(y)

🔸 Step 4: Train LSTM Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, batch_size=16, verbose=1)
model.save("lstm_demand_forecasting.h5")


---

🔹 2️⃣ Web App Deployment (Streamlit - app.py)

The Streamlit app allows users to train the model, view sales trends, and forecast future sales.

🔸 Step 1: Install Dependencies

pip install streamlit numpy pandas matplotlib scikit-learn tensorflow

🔸 Step 2: Run the Web App

streamlit run app.py

🔸 Step 3: Interactive Sales Forecasting

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.title("Retail Store Demand Forecasting")

if st.button("Train Model"):
    model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    model.save("lstm_demand_forecasting.h5")
    st.success("Model Trained and Saved!")

if st.button("Load Model and Forecast"):
    model = load_model("lstm_demand_forecasting.h5")
    st.success("Model Loaded! Predicting future sales...")


---

🛠 Installation & Setup

1️⃣ Clone Repository

git clone https://github.com/your-repo/retail-demand-forecasting.git
cd retail-demand-forecasting

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the App

streamlit run app.py
