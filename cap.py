import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import requests
import tempfile
import os

# Function to load SVR model from a URL
def load_svr_model_from_github(model_url):
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_model_file:
            temp_model_file.write(response.content)
            temp_model_file_path = temp_model_file.name
        svr_model = joblib.load(temp_model_file_path)
        return svr_model
    except Exception as e:
        st.error(f"Error loading SVR model: {e}")
        return None
    finally:
        if 'temp_model_file_path' in locals() and os.path.exists(temp_model_file_path):
            os.unlink(temp_model_file_path)

# Function to train SVR model
def train_svr_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data['Close'].values.reshape(-1, 1))
    X = np.arange(len(data)).reshape(-1, 1)
    y = data['Close'].values
    svr_model = SVR(kernel='rbf')
    svr_model.fit(scaler.transform(X), y)
    return svr_model, scaler

# Streamlit UI
def main():
    st.title('Stock Market Predictor')

    # Sidebar: Input parameters
    st.sidebar.subheader('Input Parameters')
    stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
    start_date = st.sidebar.date_input('Select Start Date', pd.to_datetime('1985-01-01'))
    end_date = st.sidebar.date_input('Select End Date', pd.to_datetime('today'))

    # Fetch stock data
    data = yf.download(stock, start=start_date, end=end_date)

    # Display stock data
    st.subheader('Stock Data')
    st.write(data)

    # Calculate moving averages
    ma_100_days = data['Close'].rolling(window=100).mean()
    ma_200_days = data['Close'].rolling(window=200).mean()

    # Plot moving averages
    st.subheader('Moving Average Plots')
    fig_ma100 = go.Figure()
    fig_ma100.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100'))
    fig_ma100.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_ma100.update_layout(title='Price vs MA100', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_ma100)

    fig_ma200 = go.Figure()
    fig_ma200.add_trace(go.Scatter(x=data.index, y=ma_100_days, mode='lines', name='MA100', line=dict(color='red')))
    fig_ma200.add_trace(go.Scatter(x=data.index, y=ma_200_days, mode='lines', name='MA200', line=dict(color='blue')))
    fig_ma200.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='green')))
    fig_ma200.update_layout(title='Price vs MA100 vs MA200', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_ma200)

    # Load the SVR model from GitHub
    model_url = 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/regressor_model.h5'
    svr_model = load_svr_model_from_github(model_url)
    if svr_model is not None:
        st.success("SVR model loaded successfully!")

    # Train SVR model
    svr_model, svr_scaler = train_svr_model(data)
    st.success("SVR model trained successfully!")

    # Model Training and Prediction
    if svr_model is not None:
        # Make predictions
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data['Close'].values.reshape(-1, 1))
        X_pred = np.arange(len(data), len(data) + 7).reshape(-1, 1)
        X_pred_scaled = scaler.transform(X_pred)
        y_pred = svr_model.predict(X_pred_scaled).flatten()

        # Plot Original vs Predicted Prices
        st.subheader('Original vs Predicted Prices')
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data.index[-7:], y=y_pred, mode='lines', name='Predicted Price',
                                       hovertemplate='Date: %{x}<br>Predicted Price: %{y:.2f}<extra></extra>'))
        fig_pred.add_trace(go.Scatter(x=data.index[-7:], y=data['Close'].values[-7:], mode='lines', name='Original Price',
                                       hovertemplate='Date: %{x}<br>Original Price: %{y:.2f}<extra></extra>'))
        fig_pred.update_layout(title='Original Price vs Predicted Price', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_pred)

        # Forecasted Prices for Next 7 Days
        st.subheader('Next 7 Days Forecasted Close Prices')
        forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=7)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close Price': y_pred})
        st.write(forecast_df)
    else:
        st.error("Please select a valid model.")

if __name__ == "__main__":
    main()
