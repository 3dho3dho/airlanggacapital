import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from fredapi import Fred
from statsmodels.tsa.api import VAR
from prophet import Prophet
import yfinance as yf

# Set up Streamlit UI
st.title("Financial Forecasting App")
st.sidebar.header("Settings")

# Sidebar: Choose Asset Type
asset_type = st.sidebar.selectbox("Select Asset Type:", ["Bonds", "Stocks", "Bitcoin"])

if asset_type == "Bonds":
    # Replace with your FRED API key
    API_KEY = "d6ec749e444b3522c809f3b045029ba7"
    fred = Fred(api_key=API_KEY)

    # Define U.S. Treasury Yield Curve Series IDs
    maturity_dict = {
        "1M": "DGS1MO",
        "3M": "DGS3MO",
        "6M": "DGS6MO",
        "1Y": "DGS1",
        "2Y": "DGS2",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "7Y": "DGS7",
        "10Y": "DGS10",
        "20Y": "DGS20",
        "30Y": "DGS30"
    }

    # Fetch historical yield data
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')
    df = pd.DataFrame({maturity: fred.get_series(series_id, start_date) for maturity, series_id in maturity_dict.items()})
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)

    # User selects historical comparison date
    one_year_ago = today - datetime.timedelta(days=365)
    historical_date = st.sidebar.date_input("Select Historical Comparison Date:", one_year_ago)
    historical_yield_curve = df.loc[df.index == pd.Timestamp(historical_date)].squeeze()

    # Forecasting using VAR
    forecast_days = st.sidebar.slider("Forecast Horizon (days):", min_value=1, max_value=30, value=7)
    model = VAR(df)
    results = model.fit(maxlags=10, ic='aic')
    forecast = results.forecast(df.values[-results.k_ar:], steps=forecast_days)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_days+1, freq='B')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=df.columns)

    # Plot Yield Curve Forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(maturity_dict.keys()), y=df.iloc[-1].values, mode='lines+markers', name="Today", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(maturity_dict.keys()), y=forecast_df.iloc[-1].values, mode='lines+markers', name=f"Forecast ({forecast_days} Days Ahead)", line=dict(color='red', dash='dash')))

    # Add historical yield curve
    if not historical_yield_curve.empty:
        fig.add_trace(go.Scatter(x=list(maturity_dict.keys()), y=historical_yield_curve.values, mode='lines+markers', name=f"Historical ({historical_date})", line=dict(color='green', dash='dot')))

    fig.update_layout(title="Yield Curve Forecast", xaxis_title="Maturity", yaxis_title="Yield (%)", template="plotly_white")
    st.plotly_chart(fig)

    # Riding the Yield Curve Analysis
    riding_opportunity = "YES" if forecast_df.iloc[-1]['10Y'] < forecast_df.iloc[-1]['5Y'] else "NO"
    st.markdown(f"### Riding the Yield Curve Opportunity: {riding_opportunity}")

    # Example Analysis
    example_text = "If the 5-year yield is currently higher than the 10-year yield, riding the yield curve may not be beneficial."
    st.markdown(example_text)

    # Backtesting Results
    st.markdown("### Backtesting Results")

    # Get actual values for backtesting
    backtest_start = df.index[-forecast_days] if len(df) > forecast_days else df.index[0]
    actual_values = df.loc[backtest_start:].iloc[:forecast_days]

    # Generate past forecast
    past_forecast_values = results.forecast(df.loc[:backtest_start].values[-results.k_ar:], steps=forecast_days)
    past_forecast_df = pd.DataFrame(past_forecast_values, index=actual_values.index, columns=df.columns)

    # Format dates for better readability
    actual_values.index = actual_values.index.strftime('%d-%m-%Y')
    past_forecast_df.index = past_forecast_df.index.strftime('%d-%m-%Y')

    # Display actual values separately
    st.markdown("#### Actual Yield Data")
    st.dataframe(actual_values)

    # Display past forecast values separately
    st.markdown("#### Past Forecasted Yield Data")
    st.dataframe(past_forecast_df)

elif asset_type in ["Stocks", "Bitcoin"]:
    # Sidebar: Select Stock/BTC Ticker
    ticker = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, BTC-USD):", "AAPL" if asset_type == "Stocks" else "BTC-USD")

    # Fetch historical data from Yahoo Finance
    stock_data = yf.download(ticker, period="5y", interval="1d")[['Close']].dropna().reset_index()
    stock_data.columns = ['ds', 'y']

    # Train Prophet Model
    model = Prophet()
    model.fit(stock_data)

    # Sidebar: Select Forecast Horizon
    forecast_days = st.sidebar.slider("Forecast Horizon (days):", min_value=1, max_value=365, value=30)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Plot Forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['ds'], y=stock_data['y'], mode='lines', name="Historical", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Forecast", line=dict(color='red', dash='dash')))
    fig.update_layout(title=f"{ticker} Price Forecast", xaxis_title="Date", yaxis_title="Price (USD)", template="plotly_white")
    st.plotly_chart(fig)

    # Back-testing
    st.markdown(f"### Back-Testing Results ({forecast_days}-Day Horizon)")

    # Get past data for backtesting
    test_start = stock_data['ds'].max() - pd.Timedelta(days=forecast_days)
    test_data = stock_data[stock_data['ds'] >= test_start]

    if not test_data.empty:
        backtest_future = model.make_future_dataframe(periods=len(test_data))
        backtest_forecast = model.predict(backtest_future)

        # Merge actual & predicted values
        comparison = test_data.merge(backtest_forecast[['ds', 'yhat']], on='ds', how='left')
        comparison['Error'] = comparison['y'] - comparison['yhat']
        comparison['APE'] = abs(comparison['Error'] / comparison['y']) * 100  # Absolute Percentage Error

        # Format dates
        comparison['ds'] = comparison['ds'].dt.strftime('%d-%m-%Y')

        # Display results
        st.dataframe(comparison.rename(columns={'ds': 'Date', 'y': 'Actual Price', 'yhat': 'Predicted Price', 'Error': 'Prediction Error', 'APE': 'Absolute % Error'}).set_index('Date'))
        st.markdown(f"**Mean Absolute Percentage Error (MAPE): {comparison['APE'].mean():.2f}%**")
    else:
        st.warning("Not enough recent data for back-testing.")