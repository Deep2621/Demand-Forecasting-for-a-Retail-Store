import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Install required libraries (Run in terminal if needed)
# pip install streamlit numpy pandas matplotlib scikit-learn tensorflow

# Load dataset
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df_daily = df.groupby("Date")["Units Sold"].sum().reset_index()
    return df_daily

def preprocess_data(df_daily, sequence_length=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_daily["Scaled_Sales"] = scaler.fit_transform(df_daily[["Units Sold"]])
    X, y = [], []
    for i in range(len(df_daily) - sequence_length):
        X.append(df_daily["Scaled_Sales"].iloc[i : i + sequence_length].values)
        y.append(df_daily["Scaled_Sales"].iloc[i + sequence_length])
    X, y = np.array(X), np.array(y)
    split_index = int(len(X) * 0.8)
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:], scaler, df_daily

def build_model(sequence_length):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def forecast_future(model, X_test, scaler, df_daily, future_steps=30):
    future_inputs = X_test[-1].reshape(1, 30, 1)
    future_preds = []
    for _ in range(future_steps):
        pred = model.predict(future_inputs)
        future_preds.append(pred[0, 0])
        future_inputs = np.append(future_inputs[:, 1:, :], [[[pred[0, 0]]]], axis=1)
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(start=df_daily["Date"].iloc[-1], periods=future_steps+1, freq='D')[1:]
    return future_dates, future_preds_inv

def aggregate_sales(df_daily, freq):
    return df_daily.resample(freq, on="Date").sum()

df_daily = load_data()
X_train, X_test, y_train, y_test, scaler, df_daily = preprocess_data(df_daily)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

st.title("Retail Store Demand Forecasting")

if st.button("Train Model"):
    model = build_model(30)
    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    model.save("lstm_demand_forecasting.h5")
    st.success("Model Trained and Saved!")

if st.button("Load Model and Forecast"):
    model = load_model("lstm_demand_forecasting.h5")
    future_dates, future_preds_inv = forecast_future(model, X_test, scaler, df_daily)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(future_dates, future_preds_inv, label="Forecasted Sales", color="green", linestyle="dashed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.set_title("30-Day Sales Forecast using LSTM")
    ax.legend()
    st.pyplot(fig)

st.subheader("View Sales Data")
if st.button("Daily Sales"):
    st.line_chart(aggregate_sales(df_daily, 'D')["Units Sold"])
if st.button("Weekly Sales"):
    st.line_chart(aggregate_sales(df_daily, 'W')["Units Sold"])
if st.button("Monthly Sales"):
    st.line_chart(aggregate_sales(df_daily, 'M')["Units Sold"])
if st.button("Yearly Sales"):
    st.line_chart(aggregate_sales(df_daily, 'Y')["Units Sold"])
