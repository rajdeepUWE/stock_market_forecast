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
from tensorflow.keras.models import load_model
import urllib.request

# Function to load Keras model from a URL
def load_keras_model_from_github(model_url):
    try:
        # Download the model file
        with urllib.request.urlopen(model_url) as response:
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
                temp_model_file.write(response.read())
                temp_model_file_path = temp_model_file.name

        # Load the Keras model from the temporary file
        keras_model = load_model(temp_model_file_path)
        return keras_model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None
    finally:
        # Clean up: delete the temporary file
        if os.path.exists(temp_model_file_path):
            os.unlink(temp_model_file_path)

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

    # Load the Keras model from GitHub
    model_url = 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/linear_regression_model.h5'
    keras_model = load_keras_model_from_github(model_url)
    if keras_model is not None:
        st.success("Keras Neural Network model loaded successfully!")

        # Make predictions
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data['Close'].values.reshape(-1, 1))
        X_pred = np.arange(len(data), len(data) + 7).reshape(-1, 1)
        X_pred_scaled = scaler.transform(X_pred)
        y_pred = keras_model.predict(X_pred_scaled).flatten()

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
