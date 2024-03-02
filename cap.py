import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA

# Function to load Keras model from a URL
def load_model_from_github(model_url):
    try:
        # Load the model from the GitHub URL
        model = load_model(model_url)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to load ARIMA model from a file
def load_arima_model(model_file):
    try:
        # Load the ARIMA model from the file
        model = ARIMA.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading ARIMA model: {e}")
        return None

# Function to make predictions using the selected model
def make_predictions(model_type, model, X_pred_scaled, scaler):
    if model_type == 'LSTM' or model_type == 'Regressor':
        y_pred = model.predict(X_pred_scaled).flatten()
    elif model_type == 'Random Forest' or model_type == 'Linear Regression':
        y_pred = model.predict(X_pred_scaled)
    else:  # ARIMA
        y_pred = model.forecast(steps=len(X_pred_scaled))[0]
    return scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

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

    # Model selection
    selected_model = st.sidebar.selectbox('Select Model', ['LSTM', 'Regressor', 'Random Forest', 'Linear Regression', 'ARIMA'])

    # Load the selected model
    model_files = {
        'LSTM': 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/LSTM.h5',
        'Regressor': 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/regressor_model.h5',
        'Random Forest': 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/random_forest_model.h5',
        'Linear Regression': 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/linear_regression_model.h5',
        'ARIMA': 'https://github.com/rajdeepUWE/stock_market_forecast/raw/master/arima_model.pkl'
    }

    model_file = model_files.get(selected_model)
    if model_file:
        if selected_model in ['LSTM', 'Regressor']:
            model = load_model_from_github(model_file)
        elif selected_model == 'Random Forest':
            model = RandomForestRegressor()
            model.load(model_file)
        elif selected_model == 'Linear Regression':
            model = LinearRegression()
            model.load(model_file)
        elif selected_model == 'ARIMA':
            model = load_arima_model(model_file)
        else:
            st.error("Invalid model selected.")

        if model:
            st.success(f"{selected_model} model loaded successfully!")

            # Prepare data for predictions
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(data['Close'].values.reshape(-1, 1))
            X_pred = np.arange(len(data), len(data) + 7).reshape(-1, 1)
            X_pred_scaled = scaler.transform(X_pred)

            # Make predictions
            y_pred = make_predictions(selected_model, model, X_pred_scaled, scaler)

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
