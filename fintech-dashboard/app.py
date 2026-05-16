import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data.data_manager import fetch_and_save_data
from arch import arch_model
import ta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np
import yfinance as yf

# --- Path Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
MODELS_DIR = os.path.join(APP_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Page Config ---
st.set_page_config(layout="wide", page_title="FinTech Dashboard")

# --- Data Loading ---
@st.cache_data
def load_data(ticker):
    """
    Loads historical data for a given ticker. If the data is not found locally,
    it fetches from yfinance and saves it.
    """
    file_path = os.path.join(DATA_DIR, f'{ticker}_historical.csv')
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_numeric:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)
    except FileNotFoundError:
        st.warning(f"Data for {ticker} not found locally. Fetching from yfinance...")
        # Pass the absolute DATA_DIR to the fetch function
        data = fetch_and_save_data(ticker=ticker, data_dir=DATA_DIR)
        if data is None or data.empty:
            st.error(f"Could not fetch data for {ticker}.")
            return None
    return data

# --- ML Feature Creation ---
@st.cache_data
def create_features(df):
    df_copy = df.copy()
    df_copy = df_copy.loc[:,~df_copy.columns.duplicated()]
    df_copy = ta.add_all_ta_features(df_copy, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)
    df_copy.dropna(inplace=True)
    return df_copy

# --- Portfolio Optimization ---
@st.cache_data
def run_portfolio_optimization(tickers, start_date="2022-01-01"):
    portfolio_data_raw = yf.download(tickers, start=start_date)
    if 'Adj Close' in portfolio_data_raw.columns:
        portfolio_data = portfolio_data_raw['Adj Close']
    else:
        portfolio_data = portfolio_data_raw['Close']
    returns = portfolio_data.pct_change().dropna()
    num_portfolios = 10000
    all_weights = np.zeros((num_portfolios, len(tickers)))
    ret_arr = np.zeros(num_portfolios)
    vol_arr = np.zeros(num_portfolios)
    sharpe_arr = np.zeros(num_portfolios)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        all_weights[i,:] = weights
        ret_arr[i] = np.sum(mean_returns * weights)
        vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_arr[i] = ret_arr[i] / vol_arr[i]
    max_sharpe_idx = np.argmax(sharpe_arr)
    max_sharpe_ret = ret_arr[max_sharpe_idx]
    max_sharpe_vol = vol_arr[max_sharpe_idx]
    max_sharpe_weights = all_weights[max_sharpe_idx,:]
    min_vol_idx = np.argmin(vol_arr)
    min_vol_ret = ret_arr[min_vol_idx]
    min_vol_vol = vol_arr[min_vol_idx]
    min_vol_weights = all_weights[min_vol_idx,:]
    return (ret_arr, vol_arr, sharpe_arr, max_sharpe_ret, max_sharpe_vol, max_sharpe_weights, min_vol_ret, min_vol_vol, min_vol_weights)

# --- Main App ---
st.title("FinTech Dashboard & Predictive Analytics")

# --- Sidebar ---
st.sidebar.header("Single-Stock Analysis")
selected_ticker = st.sidebar.selectbox("Select Stock Ticker", ("MSFT", "AAPL", "GOOGL", "AMZN"))

# --- Main Content ---
data = load_data(selected_ticker)

if data is not None:
    st.header(f"Analysis for {selected_ticker}")
    # 1. Candlestick Chart
    st.subheader("Stock Price Movement")
    fig_candlestick = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Candlestick')])
    fig_candlestick.update_layout(xaxis_rangeslider_visible=False, height=500)
    st.plotly_chart(fig_candlestick, use_container_width=True)

    # 2. GARCH Model for Risk Analysis
    st.subheader("Volatility Analysis (GARCH Model)")
    returns = 100 * data['Close'].pct_change().dropna()
    if len(returns) > 5:
        am = arch_model(returns, vol='Garch', p=1, q=1)
        res = am.fit(update_freq=5, disp='off')
        st.write("GARCH(1,1) Model Results")
        st.text(str(res.summary()))
        volatility_df = res.conditional_volatility
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(x=volatility_df.index, y=volatility_df, mode='lines', name='Conditional Volatility'))
        fig_volatility.update_layout(title='Predicted Conditional Volatility', height=400)
        st.plotly_chart(fig_volatility, use_container_width=True)
    else:
        st.warning("Not enough data for GARCH model.")

    # 3. Predictive Analytics (Machine Learning)
    st.subheader("Predictive Analytics Results")
    data_ml = create_features(data)
    if not data_ml.empty:
        original_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Target']
        cols_to_drop = [col for col in original_cols if col in data_ml.columns]
        X = data_ml.drop(columns=cols_to_drop)
        y = data_ml['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        model_path = os.path.join(MODELS_DIR, f'{selected_ticker}_rf_model.pkl')

        # --- Load or Train Model ---
        if not os.path.exists(model_path):
            st.info(f"Training model for {selected_ticker} for the first time. This may take a moment...")
            if not X_train.empty:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)
            else:
                st.warning("Not enough training data to create a model.")
                model = None
        else:
            model = joblib.load(model_path)

        if model:
            # Note: Accuracy on a historical test set is not a "confidence" for a future prediction.
            # It's a measure of how well the model performed in the past.
            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.write(f"Model Accuracy on Historical Test Data: **{accuracy*100:.2f}%**")
            last_prediction = model.predict(X.iloc[[-1]])[0]
            prediction_text = "Up (Bullish) 🟢" if last_prediction == 1 else "Down (Bearish) 🔴"
            st.metric("Next Day Price Direction Forecast", prediction_text)
        else:
            st.warning("Not enough data for model training after splitting.")
    else:
        st.warning("Not enough data for feature creation.")
else:
    st.error("Could not load data. Please check the ticker or data source.")

st.divider()

# --- 4. Portfolio Optimization ---
st.header("Markowitz Portfolio Optimization")
st.sidebar.header("Portfolio Optimization")
port_tickers_input = st.sidebar.text_input("Enter Tickers (comma-separated)", 'AAPL,MSFT,GOOGL,AMZN')
port_tickers = [ticker.strip().upper() for ticker in port_tickers_input.split(',')]

if len(port_tickers) > 1:
    (p_ret, p_vol, p_sharpe, max_sr_ret, max_sr_vol, max_sr_weights, min_v_ret, min_v_vol, min_v_weights) = run_portfolio_optimization(tickers=port_tickers)

    st.subheader("Efficient Frontier")
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(
        x=p_vol, y=p_ret, mode='markers',
        marker=dict(color=p_sharpe, showscale=True, size=7, line=dict(width=1), colorscale="Viridis", colorbar=dict(title="Sharpe Ratio")),
        name='Simulated Portfolios'
    ))
    fig_portfolio.add_trace(go.Scatter(x=[max_sr_vol], y=[max_sr_ret], mode='markers', marker=dict(color='red', size=14, symbol='star'), name='Max Sharpe Ratio'))
    fig_portfolio.add_trace(go.Scatter(x=[min_v_vol], y=[min_v_ret], mode='markers', marker=dict(color='green', size=14, symbol='star'), name='Min Volatility'))
    fig_portfolio.update_layout(title='Efficient Frontier', xaxis_title='Annualized Volatility (Risk)', yaxis_title='Annualized Return', height=600)
    st.plotly_chart(fig_portfolio, use_container_width=True)

    st.subheader("Optimal Portfolio Allocations")
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Maximum Sharpe Ratio Portfolio")
        max_sr_df = pd.DataFrame(max_sr_weights * 100, index=port_tickers, columns=['Weight (%)'])
        st.dataframe(max_sr_df.style.format("{:.2f}"))
        st.write(f"**Annualized Return:** {max_sr_ret*100:.2f}%")
        st.write(f"**Annualized Volatility:** {max_sr_vol*100:.2f}%")

    with col2:
        st.write("#### Minimum Volatility Portfolio")
        min_vol_df = pd.DataFrame(min_v_weights * 100, index=port_tickers, columns=['Weight (%)'])
        st.dataframe(min_vol_df.style.format("{:.2f}"))
        st.write(f"**Annualized Return:** {min_v_ret*100:.2f}%")
        st.write(f"**Annualized Volatility:** {min_v_vol*100:.2f}%")
else:
    st.warning("Please enter at least two tickers for portfolio analysis.")
