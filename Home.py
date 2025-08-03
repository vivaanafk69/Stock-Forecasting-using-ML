import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import ta
import warnings
import hashlib
import json
from datetime import datetime, timedelta
import time
import requests

warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="FinSight", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and interactivity
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .FinSight-credits {
        text-align: center;
        font-size: 0.95rem;
        color: #764ba2;
        margin-bottom: 2rem;
        margin-top: -0.5rem;
        letter-spacing: 0.01em;
    }
    .metric-container, .prediction-card {
        box-shadow: 0 6px 24px 0 rgba(102,126,234,0.18);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .metric-container:hover, .prediction-card:hover {
        transform: scale(1.025);
        box-shadow: 0 12px 32px 0 rgba(102,126,234,0.25);
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: #fff;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 8px 0 rgba(102,126,234,0.15);
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 4px 16px 0 rgba(102,126,234,0.25);
    }
    .section-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        margin: 2.5rem 0 1.5rem 0;
        opacity: 0.18;
    }
    .error-box {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #efe;
        border: 1px solid #cfc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">☘️FinSight</h1>', unsafe_allow_html=True)
st.markdown('<div class="FinSight-credits">Developed by Vivaan Gandhi<br>& Puranjay Haldankar</div>', unsafe_allow_html=True)

# Popular stock suggestions
POPULAR_STOCKS = {
    "US Stocks": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"],
    "Indian Stocks": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "SBIN.NS"],
    "ETFs": ["SPY", "QQQ", "VOO", "VTI", "ARKK"]
}

# Sidebar Configuration
with st.sidebar:
    st.header("📊 Configuration Panel")
    
    # Stock Selection with suggestions
    st.subheader("🎯 Quick Select")
    
    # Show popular stocks in expandable sections
    for category, stocks in POPULAR_STOCKS.items():
        with st.expander(f"{category}"):
            cols = st.columns(2)
            for i, stock in enumerate(stocks):
                if cols[i % 2].button(stock, key=f"{category}_{stock}"):
                    st.session_state.selected_ticker = stock

    # Manual ticker input
    default_ticker = st.session_state.get('selected_ticker', 'AAPL')
    ticker = st.text_input("Stock Ticker", value=default_ticker, help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)").upper()

    # Time Parameters
    period = st.selectbox("Time Period(for training data)", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Data Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    #Chart Time Period
    # Chart display range toggle
    display_range = st.selectbox(
    "Chart Display Range",
    [
        "1 Day", "1 Week", "1 Month", "3 Months", "6 Months"
    ],
    index=4
    )
    
    # Model Configuration
    st.subheader("🤖 ML Configuration")
    model_type = st.selectbox("Prediction Model", [
        "Ensemble (Recommended)",
        "Random Forest", 
        "XGBoost", 
        "Gradient Boosting",
        "Linear Regression",
        "ARIMA", 
        "Prophet"
    ])
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)
        confidence_interval = st.slider("Confidence Interval %", 80, 99, 95)
        model_seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
        auto_refresh = st.checkbox("Auto-refresh data", value=False)
        retry_attempts = st.slider("API Retry Attempts", 1, 5, 3)
        
    # Data Controls
    show_indicators = st.multiselect("Technical Indicators", [
    "SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "Volume"
], default=["SMA", "EMA", "Bollinger Bands", "RSI", "MACD", "Volume"])

    refresh_btn = st.button("🔄 Refresh Data", type="primary")

# Enhanced data fetching with better error handling
@st.cache_data(ttl=300)
def fetch_stock_data_robust(symbol, period, interval, max_retries=3):
    """Enhanced stock data fetching with multiple fallback strategies"""
    
    def try_fetch_with_delay(symbol, period, interval, delay=1):
        """Try fetching data with delay"""
        time.sleep(delay)
        try:
            # Try different approaches
            approaches = [
                lambda: yf.download(symbol, period=period, interval=interval, progress=False, timeout=10),
                lambda: yf.download(symbol, period=period, interval=interval, progress=False, timeout=15, threads=False),
                lambda: yf.Ticker(symbol).history(period=period, interval=interval, timeout=10)
            ]
            
            for i, approach in enumerate(approaches):
                try:
                    data = approach()
                    if not data.empty:
                        return data, "Success"
                    st.warning(f"Approach {i+1} returned empty data, trying next...")
                except Exception as e:
                    st.warning(f"Approach {i+1} failed: {str(e)}")
                    continue
            
            return pd.DataFrame(), "All approaches failed"
            
        except Exception as e:
            return pd.DataFrame(), f"Error: {str(e)}"
    
    # Alternative ticker formats for Indian stocks
    alternative_symbols = [symbol]
    if symbol.endswith('.NS'):
        base_symbol = symbol.replace('.NS', '')
        alternative_symbols.extend([
            f"{base_symbol}.BO",  # Bombay Stock Exchange
            base_symbol,  # Without suffix
            f"{base_symbol}.BSE"  # Alternative BSE format
        ])
    elif not '.' in symbol and symbol not in ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']:
        # If it might be an Indian stock without suffix
        alternative_symbols.extend([f"{symbol}.NS", f"{symbol}.BO"])
    
    last_error = None
    
    # Try each symbol variation
    for attempt_symbol in alternative_symbols:
        st.info(f"Trying to fetch data for {attempt_symbol}...")
        
        for retry in range(max_retries):
            try:
                delay = retry * 2  # Progressive delay
                data, status = try_fetch_with_delay(attempt_symbol, period, interval, delay)
                
                if not data.empty:
                    # Process the data
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    data = data.reset_index()
                    
                    # Standardize datetime column
                    date_col = 'Datetime' if 'Datetime' in data.columns else 'Date'
                    data['Time'] = pd.to_datetime(data[date_col])
                    
                    # Validate required columns
                    required = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in data.columns for col in required):
                        st.success(f"✅ Successfully fetched data for {attempt_symbol}")
                        return data, f"Success with {attempt_symbol}"
                    else:
                        last_error = f"Missing required OHLCV columns for {attempt_symbol}"
                        
            except Exception as e:
                last_error = f"Attempt {retry+1} for {attempt_symbol}: {str(e)}"
                st.warning(last_error)
                
        st.error(f"❌ Failed to fetch data for {attempt_symbol}")
    
    return pd.DataFrame(), last_error or "All symbol variations failed"

# Improved technical indicators calculation
@st.cache_data
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators with error handling"""
    if df.empty or len(df) < 20:
        return df
    
    data = df.copy()
    
    try:
        close = data['Close'].squeeze()
        high = data['High'].squeeze()
        low = data['Low'].squeeze()
        volume = data['Volume'].squeeze()
        
        # Ensure we have enough data points
        if len(close) < 20:
            st.warning("Not enough data points for technical indicators")
            return data
        
        # Basic trend indicators
        try:
            data['SMA_20'] = ta.trend.sma_indicator(close, window=min(20, len(close)-1))
            data['SMA_50'] = ta.trend.sma_indicator(close, window=min(50, len(close)-1)) if len(close) > 50 else np.nan
            data['EMA_20'] = ta.trend.ema_indicator(close, window=min(20, len(close)-1))
        except Exception as e:
            st.warning(f"Error calculating moving averages: {e}")
        
        # MACD
        try:
            data['MACD'] = ta.trend.macd_diff(close)
            data['MACD_signal'] = ta.trend.macd_signal(close)
        except Exception as e:
            st.warning(f"Error calculating MACD: {e}")
        
        # RSI
        try:
            data['RSI'] = ta.momentum.rsi(close, window=min(14, len(close)-1))
        except Exception as e:
            st.warning(f"Error calculating RSI: {e}")
        
        # Bollinger Bands
        try:
            data['BB_upper'] = ta.volatility.bollinger_hband(close)
            data['BB_lower'] = ta.volatility.bollinger_lband(close)
            data['BB_middle'] = ta.volatility.bollinger_mavg(close)
        except Exception as e:
            st.warning(f"Error calculating Bollinger Bands: {e}")
        
        # Volume indicators
        try:
            data['OBV'] = ta.volume.on_balance_volume(close, volume)
            data['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=min(14, len(close)-1))
        except Exception as e:
            st.warning(f"Error calculating volume indicators: {e}")
        
        # Advanced indicators (only if enough data)
        if len(data) >= 50:
            try:
                data['ATR'] = ta.volatility.average_true_range(high, low, close)
                data['ADX'] = ta.trend.adx(high, low, close)
                data['CCI'] = ta.trend.cci(high, low, close)
                data['ROC'] = ta.momentum.roc(close)
                data['Stoch_K'] = ta.momentum.stoch(high, low, close)
                data['Williams_R'] = ta.momentum.williams_r(high, low, close)
            except Exception as e:
                st.warning(f"Error calculating advanced indicators: {e}")
        
    except Exception as e:
        st.error(f"Major error in technical indicators calculation: {str(e)}")
    
    return data

def create_advanced_features(df):
    """Create sophisticated features for ML models with better error handling"""
    if df.empty or len(df) < 30:
        return None, None
    
    try:
        features = []
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Change'] = df['Close'] - df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            if len(df) > lag:
                df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
                df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
                df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
        
        # Technical indicator features
        tech_features = ['RSI', 'MACD', 'ATR', 'MFI', 'Williams_R', 'Stoch_K']
        for feature in tech_features:
            if feature in df.columns and not df[feature].isna().all():
                try:
                    rolling_mean = df[feature].rolling(20, min_periods=5).mean()
                    rolling_std = df[feature].rolling(20, min_periods=5).std()
                    df[f'{feature}_norm'] = (df[feature] - rolling_mean) / rolling_std
                except:
                    pass
        
        # Moving averages ratios
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['SMA_ratio'] = df['SMA_20'] / df['SMA_50']
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(5, min_periods=3).std()
        df['Volatility_20'] = df['Returns'].rolling(20, min_periods=10).std()
        
        # Select feature columns
        feature_cols = [col for col in df.columns if any(x in col for x in [
            'lag_', '_norm', 'Returns', 'Volatility', 'Price_Change', 'High_Low_Ratio', 
            'Volume_Price_Trend', 'SMA_ratio'
        ])]
        
        # Clean data
        clean_df = df[['Close'] + feature_cols].dropna()
        if len(clean_df) < 20:
            return None, None
        
        X = clean_df[feature_cols].values
        y = clean_df['Close'].values
        
        return X, y
        
    except Exception as e:
        st.error(f"Error creating features: {str(e)}")
        return None, None

class EnsemblePredictor:
    """Advanced ensemble predictor with multiple models"""
    
    def __init__(self, random_state=42):
        try:
            self.models = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=50, random_state=random_state),
                'lr': LinearRegression()
            }
            
            # Try to add XGBoost if available
            try:
                self.models['xgb'] = xgb.XGBRegressor(n_estimators=50, random_state=random_state)
            except:
                st.warning("XGBoost not available, using other models")
                
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            # Fallback to simpler models
            self.models = {
                'lr': LinearRegression(),
                'rf': RandomForestRegressor(n_estimators=20, random_state=random_state)
            }
        
        self.weights = None
        self.is_fitted = False
    
    def fit(self, X, y):
        try:
            if len(X) < 10:
                raise ValueError("Not enough data for training")
                
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            predictions = {}
            successful_models = {}
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    predictions[name] = pred
                    successful_models[name] = model
                except Exception as e:
                    st.warning(f"Model {name} failed: {e}")
                    continue
            
            if not predictions:
                raise ValueError("No models trained successfully")
            
            # Calculate optimal weights based on validation performance
            errors = {}
            for name, pred in predictions.items():
                try:
                    errors[name] = mean_squared_error(y_val, pred)
                except:
                    errors[name] = float('inf')
            
            # Remove models with infinite errors
            errors = {k: v for k, v in errors.items() if v != float('inf')}
            
            if not errors:
                # Equal weights fallback
                self.weights = {name: 1.0/len(successful_models) for name in successful_models.keys()}
            else:
                # Inverse error weighting
                total_inv_error = sum(1/error for error in errors.values())
                self.weights = {name: (1/error)/total_inv_error for name, error in errors.items()}
            
            # Update models dict to only include successful models
            self.models = successful_models
            
            # Refit on full data
            for model in self.models.values():
                model.fit(X, y)
            
            self.is_fitted = True
            
        except Exception as e:
            st.error(f"Error training ensemble: {e}")
            raise
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        try:
            predictions = np.zeros(X.shape[0])
            total_weight = 0
            
            for name, model in self.models.items():
                if name in self.weights:
                    weight = self.weights[name]
                    pred = model.predict(X)
                    predictions += pred * weight
                    total_weight += weight
            
            if total_weight > 0:
                predictions /= total_weight
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_uncertainty(self, X, n_iterations=50):
        """Predict with uncertainty estimation using bootstrap"""
        try:
            predictions = []
            
            for _ in range(n_iterations):
                # Bootstrap sampling
                indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
                X_boot = X[indices]
                pred = self.predict(X_boot)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            return mean_pred, std_pred
            
        except Exception as e:
            st.error(f"Error calculating uncertainty: {e}")
            # Fallback to simple prediction
            pred = self.predict(X)
            return pred, np.zeros_like(pred)

def generate_trading_signals(df):
    """Generate comprehensive trading signals with error handling"""
    signals = []
    
    if df.empty or len(df) < 20:
        return signals
    
    try:
        latest = df.iloc[-1]
        
        # RSI Signals
        if 'RSI' in df.columns and not pd.isna(latest['RSI']):
            rsi = float(latest['RSI'])
            if rsi > 80:
                signals.append({"type": "Strong Sell", "indicator": "RSI", "value": rsi, "confidence": 0.8})
            elif rsi > 70:
                signals.append({"type": "Sell", "indicator": "RSI", "value": rsi, "confidence": 0.6})
            elif rsi < 20:
                signals.append({"type": "Strong Buy", "indicator": "RSI", "value": rsi, "confidence": 0.8})
            elif rsi < 30:
                signals.append({"type": "Buy", "indicator": "RSI", "value": rsi, "confidence": 0.6})
        
        # MACD Signals
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and len(df) >= 2:
            macd = latest['MACD'] if not pd.isna(latest['MACD']) else None
            macd_signal = latest['MACD_signal'] if not pd.isna(latest['MACD_signal']) else None
            
            if macd is not None and macd_signal is not None:
                prev_macd = df.iloc[-2]['MACD'] if not pd.isna(df.iloc[-2]['MACD']) else None
                prev_signal = df.iloc[-2]['MACD_signal'] if not pd.isna(df.iloc[-2]['MACD_signal']) else None
                
                if prev_macd is not None and prev_signal is not None:
                    if macd > macd_signal and prev_macd <= prev_signal:
                        signals.append({"type": "Buy", "indicator": "MACD Crossover", "value": macd-macd_signal, "confidence": 0.7})
                    elif macd < macd_signal and prev_macd >= prev_signal:
                        signals.append({"type": "Sell", "indicator": "MACD Crossover", "value": macd-macd_signal, "confidence": 0.7})
        
        # Moving Average Signals
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma20 = latest['SMA_20'] if not pd.isna(latest['SMA_20']) else None
            sma50 = latest['SMA_50'] if not pd.isna(latest['SMA_50']) else None
            close = float(latest['Close'])
            
            if sma20 is not None and sma50 is not None:
                if sma20 > sma50 and close > sma20:
                    signals.append({"type": "Buy", "indicator": "Golden Cross", "value": (sma20-sma50)/sma50*100, "confidence": 0.8})
                elif sma20 < sma50 and close < sma20:
                    signals.append({"type": "Sell", "indicator": "Death Cross", "value": (sma20-sma50)/sma50*100, "confidence": 0.8})
        
    except Exception as e:
        st.warning(f"Error generating signals: {e}")
    
    return signals

def create_interactive_chart(df, show_indicators):
    """Create sophisticated interactive charts with error handling"""
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Indicators', 'Volume', 'RSI', 'MACD'),
            row_width=[0.2, 0.2, 0.2, 0.4]
        )
        
        # Main price chart
        fig.add_trace(go.Candlestick(
            x=df['Time'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ), row=1, col=1)
        
        # Technical indicators
        if "SMA" in show_indicators and 'SMA_20' in df.columns:
            valid_sma = df['SMA_20'].dropna()
            if not valid_sma.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_sma.index, 'Time'], 
                    y=valid_sma,
                    line=dict(color='orange', width=2),
                    name='SMA 20'
                ), row=1, col=1)
        
        if "EMA" in show_indicators and 'EMA_20' in df.columns:
            valid_ema = df['EMA_20'].dropna()
            if not valid_ema.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_ema.index, 'Time'], 
                    y=valid_ema,
                    line=dict(color='red', width=2),
                    name='EMA 20'
                ), row=1, col=1)
        
        if "Bollinger Bands" in show_indicators and 'BB_upper' in df.columns:
            valid_bb = df[['BB_upper', 'BB_lower']].dropna()
            if not valid_bb.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_bb.index, 'Time'], 
                    y=valid_bb['BB_upper'],
                    line=dict(color='gray', width=1),
                    name='BB Upper',
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_bb.index, 'Time'], 
                    y=valid_bb['BB_lower'],
                    line=dict(color='gray', width=1),
                    fill='tonexty',
                    name='Bollinger Bands'
                ), row=1, col=1)
        
        # Volume
        if "Volume" in show_indicators:
            colors = ['red' if df.iloc[i]['Close'] < df.iloc[i]['Open'] else 'green' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df['Time'], y=df['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ), row=2, col=1)
        
        # RSI
        if "RSI" in show_indicators and 'RSI' in df.columns:
            valid_rsi = df['RSI'].dropna()
            if not valid_rsi.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_rsi.index, 'Time'], 
                    y=valid_rsi,
                    line=dict(color='purple', width=2),
                    name='RSI'
                ), row=3, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if "MACD" in show_indicators and 'MACD' in df.columns:
            valid_macd = df['MACD'].dropna()
            if not valid_macd.empty:
                fig.add_trace(go.Scatter(
                    x=df.loc[valid_macd.index, 'Time'], 
                    y=valid_macd,
                    line=dict(color='blue', width=2),
                    name='MACD'
                ), row=4, col=1)
                
                if 'MACD_signal' in df.columns:
                    valid_signal = df['MACD_signal'].dropna()
                    if not valid_signal.empty:
                        fig.add_trace(go.Scatter(
                            x=df.loc[valid_signal.index, 'Time'], 
                            y=valid_signal,
                            line=dict(color='orange', width=2),
                            name='MACD Signal'
                        ), row=4, col=1)
        
        fig.update_layout(
            title=f"{ticker} - Advanced Technical Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        # Return basic chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Time'], y=df['Close'], name='Close Price'))
        fig.update_layout(title=f"{ticker} - Basic Chart")
        return fig

# Main Application Logic
if ticker:
    # Display troubleshooting info
    st.info(f"🔍 Attempting to fetch data for: **{ticker}**")
    
    # Fetch data with enhanced error handling
    with st.spinner("📡 Fetching market data with enhanced error handling..."):
        df, status = fetch_stock_data_robust(ticker, period, interval, retry_attempts)
    
    if df.empty:
        st.error(f"❌ **Data Fetch Failed**: {status}")
        
        # Provide troubleshooting suggestions
        st.markdown("""
        ### 🔧 **Troubleshooting Suggestions:**
        
        **For Indian Stocks:**
        - ✅ Use `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`)
        - ✅ Try `.BO` suffix (e.g., `RELIANCE.BO`)
        - ✅ Check if the company is listed on NSE/BSE
        
        **For US Stocks:**
        - ✅ Use standard ticker symbols (e.g., `AAPL`, `GOOGL`)
        - ✅ Verify the company is publicly traded
        
        **General Issues:**
        - 🔄 Try refreshing the data
        - ⏰ Market might be closed
        - 🌐 Check your internet connection
        - 📊 Try a different time period
        
        **Alternative Tickers to Try:**
        """)
        
        # Show alternative suggestions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Popular US Stocks:**")
            for stock in POPULAR_STOCKS["US Stocks"][:4]:
                st.write(f"• {stock}")
        
        with col2:
            st.write("**Popular Indian Stocks:**")
            for stock in POPULAR_STOCKS["Indian Stocks"][:4]:
                st.write(f"• {stock}")
        
        with col3:
            st.write("**ETFs:**")
            for stock in POPULAR_STOCKS["ETFs"][:4]:
                st.write(f"• {stock}")
        
        st.stop()
    
    st.success(f"✅ **Successfully loaded data**: {status}")
    
    # Calculate indicators
    with st.spinner("🔧 Calculating technical indicators..."):
        df = calculate_technical_indicators(df)
    
    # Display key metrics
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        price_change = float(latest['Close']) - float(prev['Close'])
        pct_change = (price_change / float(prev['Close'])) * 100 if float(prev['Close']) != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
        
        with col2:
            st.metric("Volume", f"{float(latest['Volume']):,.0f}")
        
        with col3:
            day_range = float(latest['High']) - float(latest['Low'])
            st.metric("Day Range", f"${day_range:.2f}")
        
        with col4:
            if 'ATR' in df.columns and not pd.isna(latest['ATR']):
                st.metric("Volatility (ATR)", f"${float(latest['ATR']):.2f}")
            else:
                st.metric("Data Points", f"{len(df)}")
                
    except Exception as e:
        st.warning(f"Error displaying metrics: {e}")
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Interactive Chart
    # Filter df for chart display range 
    try:
        if display_range != "6 Months":
            now = df['Time'].max()
            if display_range == "1 Day":
                mask = df['Time'] >= now - pd.Timedelta(days=1)
            elif display_range == "1 Week":
                mask = df['Time'] >= now - pd.Timedelta(weeks=1)
            elif display_range == "1 Month":
                mask = df['Time'] >= now - pd.DateOffset(months=1)
            elif display_range == "3 Months":
                mask = df['Time'] >= now - pd.DateOffset(months=3)
            elif display_range == "6 Months":
                mask = df['Time'] >= now - pd.DateOffset(months=6)
            else:
                mask = slice(None)
            df_display = df.loc[mask] if mask is not slice(None) else df
        else:
            df_display = df
            
        st.subheader("📊 Interactive Chart Analysis")
        chart = create_interactive_chart(df_display, show_indicators)
        st.plotly_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
    
    # Advanced Predictions
    st.subheader("🤖 AI-Powered Predictions")
    
    try:
        with st.spinner("🧠 Training AI models..."):
            X, y = create_advanced_features(df)
            
            if X is not None and len(X) > 50:
                try:
                    if model_type == "Ensemble (Recommended)":
                        model = EnsemblePredictor(random_state=model_seed)
                        model.fit(X, y)
                        
                        # Multi-day predictions with proper bounds
                        predictions = []
                        uncertainties = []
                        prediction_dates = []
                        
                        # Get the last known price and date for reference
                        last_price = float(df['Close'].iloc[-1])
                        last_date = df['Time'].iloc[-1]
                        
                        # Use the last features as baseline
                        base_features = X[-1].copy()
                        
                        for day in range(prediction_days):
                            # For first prediction, use actual features
                            if day == 0:
                                current_features = base_features.reshape(1, -1)
                            else:
                                # For subsequent predictions, use a more conservative approach
                                current_features = base_features.reshape(1, -1)
                                # Add small random variation to prevent extreme predictions
                                noise = np.random.normal(0, 0.01, current_features.shape)
                                current_features = current_features + noise
                            
                            pred, uncertainty = model.predict_with_uncertainty(current_features)
                            
                            # Apply reasonable bounds to prevent extreme predictions
                            pred_bounded = max(last_price * 0.5, min(last_price * 2.0, pred[0]))
                            predictions.append(pred_bounded)
                            uncertainties.append(min(uncertainty[0], last_price * 0.1))  # Cap uncertainty
                            
                            # Calculate future dates
                            future_date = last_date + pd.Timedelta(days=day+1)
                            prediction_dates.append(future_date)
                            
                            # Update reference price for next iteration bounds
                            last_price = pred_bounded
                    
                    else:
                        # Single model predictions
                        if model_type == "Random Forest":
                            model = RandomForestRegressor(n_estimators=50, random_state=model_seed)
                        elif model_type == "XGBoost":
                            try:
                                model = xgb.XGBRegressor(n_estimators=50, random_state=model_seed)
                            except:
                                st.warning("XGBoost not available, using Random Forest")
                                model = RandomForestRegressor(n_estimators=50, random_state=model_seed)
                        elif model_type == "Gradient Boosting":
                            model = GradientBoostingRegressor(n_estimators=50, random_state=model_seed)
                        else:
                            model = LinearRegression()
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=model_seed)
                        model.fit(X_train, y_train)
                        
                        # Multi-day predictions for single models with bounds
                        predictions = []
                        uncertainties = []
                        prediction_dates = []
                        
                        # Get the last known price and date for reference
                        last_price = float(df['Close'].iloc[-1])
                        last_date = df['Time'].iloc[-1]
                        base_features = X[-1].copy()
                        
                        for day in range(prediction_days):
                            # Use more conservative feature updating
                            if day == 0:
                                current_features = base_features.reshape(1, -1)
                            else:
                                # Use base features with slight variation
                                current_features = base_features.reshape(1, -1)
                                # Add minimal noise to prevent identical predictions
                                noise = np.random.normal(0, 0.005, current_features.shape)
                                current_features = current_features + noise
                            
                            pred = model.predict(current_features)[0]
                            
                            # Apply reasonable bounds to prevent extreme predictions
                            pred_bounded = max(last_price * 0.8, min(last_price * 1.2, pred))
                            predictions.append(pred_bounded)
                            uncertainties.append(0)  # No uncertainty for single models
                            
                            # Calculate future dates
                            future_date = last_date + pd.Timedelta(days=day+1)
                            prediction_dates.append(future_date)
                            
                            # Update reference price for next iteration bounds
                            last_price = pred_bounded
                    
                    # Create predictions DataFrame with validation
                    current_price = float(latest['Close'])
                    
                    # Validate predictions before creating DataFrame
                    valid_predictions = []
                    valid_dates = []
                    
                    for i, pred in enumerate(predictions):
                        # Only include reasonable predictions
                        if current_price * 0.1 <= pred <= current_price * 10:  # Within 10x range
                            valid_predictions.append(pred)
                            valid_dates.append(prediction_dates[i])
                        else:
                            # Use a more conservative prediction if original is too extreme
                            conservative_pred = current_price * (1 + np.random.uniform(-0.05, 0.05))
                            valid_predictions.append(conservative_pred)
                            valid_dates.append(prediction_dates[i])
                    
                    predictions_df = pd.DataFrame({
                        'Date': valid_dates,
                        'Predicted_Price': valid_predictions,
                        'Price_Change': [pred - current_price for pred in valid_predictions],
                        'Percentage_Change': [((pred - current_price) / current_price) * 100 for pred in valid_predictions]
                    })
                    
                    # Display next day prediction summary
                    next_pred = valid_predictions[0]
                    pred_change = ((next_pred - current_price) / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Next Day Prediction</h3>
                            <h2>${next_pred:.2f}</h2>
                            <p>Change: {pred_change:+.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        direction = "📈 Bullish" if pred_change > 0 else "📉 Bearish"
                        confidence = min(95, max(60, 80 - abs(pred_change)))
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Market Sentiment</h3>
                            <h2>{direction}</h2>
                            <p>Confidence: {confidence:.0f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Calculate average prediction over the period
                        avg_pred = np.mean(valid_predictions)
                        avg_change = ((avg_pred - current_price) / current_price) * 100
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>{prediction_days}-Day Average</h3>
                            <h2>${avg_pred:.2f}</h2>
                            <p>Avg Change: {avg_change:+.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display predictions table
                    st.subheader(f"📅 {prediction_days}-Day Price Predictions")
                    
                    # Format the predictions table for better display
                    display_df = predictions_df.copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                    display_df['Predicted_Price'] = display_df['Predicted_Price'].apply(lambda x: f"${x:.2f}")
                    display_df['Price_Change'] = display_df['Price_Change'].apply(lambda x: f"${x:+.2f}")
                    display_df['Percentage_Change'] = display_df['Percentage_Change'].apply(lambda x: f"{x:+.2f}%")
                    
                    # Rename columns for better presentation
                    display_df.columns = ['Date', 'Predicted Price', 'Price Change', 'Percentage Change']
                    
                    # Display table with color coding
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Create prediction chart
                    fig_pred = go.Figure()
                    
                    # Add historical data (last 30 days)
                    recent_df = df.tail(30)
                    fig_pred.add_trace(go.Scatter(
                        x=recent_df['Time'],
                        y=recent_df['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add predictions
                    fig_pred.add_trace(go.Scatter(
                        x=valid_dates,
                        y=valid_predictions,
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                    
                    # Add uncertainty bands for ensemble model
                    if model_type == "Ensemble (Recommended)" and len(uncertainties) > 0 and any(u > 0 for u in uncertainties):
                        # Use only valid predictions for uncertainty bands
                        valid_uncertainties = uncertainties[:len(valid_predictions)]
                        upper_bound = [pred + 1.96 * unc for pred, unc in zip(valid_predictions, valid_uncertainties)]
                        lower_bound = [pred - 1.96 * unc for pred, unc in zip(valid_predictions, valid_uncertainties)]
                        
                        fig_pred.add_trace(go.Scatter(
                            x=valid_dates + valid_dates[::-1],
                            y=upper_bound + lower_bound[::-1],
                            fill='toself',
                            fillcolor='rgba(255,0,0,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Confidence Interval',
                            showlegend=True
                        ))
                    
                    fig_pred.update_layout(
                        title=f"{ticker} - Price Predictions ({prediction_days} Days)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Model Performance
                    if model_type != "Ensemble (Recommended)":
                        try:
                            y_pred = model.predict(X_test)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            r2 = r2_score(y_test, y_pred)
                            
                            st.subheader("📈 Model Performance")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Mean Absolute Error", f"${mae:.2f}")
                            col2.metric("Root Mean Square Error", f"${rmse:.2f}")
                            col3.metric("R² Score", f"{r2:.3f}")
                        except Exception as e:
                            st.warning(f"Error calculating performance metrics: {e}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.info("Try using a simpler model or longer time period for more data.")
            else:
                st.warning("⚠️ Insufficient data for reliable predictions. Try a longer time period or different stock.")
                
    except Exception as e:
        st.error(f"Error in prediction section: {e}")
    
    # Trading Signals
    st.subheader("🎯 AI Trading Signals")
    try:
        signals = generate_trading_signals(df)
        
        if signals:
            for signal in signals:
                signal_type = signal['type']
                indicator = signal['indicator']
                confidence = signal['confidence']
                
                if 'Buy' in signal_type:
                    st.success(f"🟢 **{signal_type}** - {indicator} (Confidence: {confidence:.0%})")
                else:
                    st.error(f"🔴 **{signal_type}** - {indicator} (Confidence: {confidence:.0%})")
        else:
            st.info("🟡 No strong signals detected. Market appears neutral.")
            
    except Exception as e:
        st.warning(f"Error generating trading signals: {e}")
    
    # Risk Analysis
    st.subheader("⚠️ Risk Analysis")
    
    try:
        if 'Volatility_20' in df.columns and not df['Volatility_20'].isna().all():
            volatility = df['Volatility_20'].iloc[-1] * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                if volatility > 3:
                    st.error(f"🔴 High Volatility: {volatility:.1f}%")
                    st.write("Consider smaller position sizes")
                elif volatility > 1.5:
                    st.warning(f"🟡 Moderate Volatility: {volatility:.1f}%")
                    st.write("Normal risk management applies")
                else:
                    st.success(f"🟢 Low Volatility: {volatility:.1f}%")
                    st.write("Relatively stable conditions")
            
            with col2:
                # Calculate VaR (Value at Risk)
                returns = df['Close'].pct_change().dropna()
                if len(returns) > 10:
                    var_95 = np.percentile(returns, 5) * 100
                    st.metric("Value at Risk (95%)", f"{var_95:.2f}%")
                else:
                    st.metric("Data Points", f"{len(df)}")
        else:
            st.info("Volatility analysis requires more data points.")
            
    except Exception as e:
        st.warning(f"Error in risk analysis: {e}")
    
    # Advanced Analytics
    with st.expander("📊 Advanced Analytics Dashboard"):
        
        try:
            tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Volume Analysis", "🔄 Correlation Matrix"])
            
            with tab1:
                # Price distribution
                try:
                    fig_dist = px.histogram(df, x='Close', nbins=50, title="Price Distribution")
                    st.plotly_chart(fig_dist, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating price distribution: {e}")
            
            with tab2:
                # Volume analysis
                try:
                    fig_vol = px.scatter(df, x='Volume', y='Close', color='Close', title="Volume vs Price")
                    st.plotly_chart(fig_vol, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating volume analysis: {e}")
            
            with tab3:
                # Correlation matrix of technical indicators
                try:
                    tech_cols = ['Close', 'Volume', 'RSI', 'MACD', 'ATR']
                    available_cols = [col for col in tech_cols if col in df.columns and not df[col].isna().all()]
                    
                    if len(available_cols) > 2:
                        corr_data = df[available_cols].corr()
                        fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto", title="Technical Indicators Correlation")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Not enough indicators available for correlation analysis.")
                except Exception as e:
                    st.error(f"Error creating correlation matrix: {e}")
                    
        except Exception as e:
            st.error(f"Error in advanced analytics: {e}")
    
    # Export Options
    st.subheader("💾 Export & Share")
    
    try:
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("📊 Export Data"):
                csv = df.to_csv(index=False)
                st.download_button("Download CSV", csv, f"{ticker}_data.csv", "text/csv")
        
        with col2:
            st.info(f"**Data Summary**: {len(df)} records from {df['Time'].min().strftime('%Y-%m-%d')} to {df['Time'].max().strftime('%Y-%m-%d')}")
            
    except Exception as e:
        st.warning(f"Error in export section: {e}")

else:
    st.info("👆 Please enter a stock ticker symbol to begin analysis.")
    
    # Show some helpful information
    st.markdown("""
    ### 🚀 **Welcome to FinSight!**
    
    **Getting Started:**
    1. 📝 Enter a stock ticker (e.g., AAPL, RELIANCE.NS)
    2. ⚙️ Configure your analysis settings
    3. 🔄 Click "Refresh Data" to load
    4. 📊 Explore charts and predictions
    
    **Supported Markets:**
    - 🇺🇸 **US Markets**: AAPL, GOOGL, MSFT, TSLA
    - 🇮🇳 **Indian Markets**: RELIANCE.NS, TCS.NS, INFY.NS
    - 🌍 **Global ETFs**: SPY, QQQ, VTI
    
    **Quick Tips:**
    - Use `.NS` for NSE stocks (India)
    - Use `.BO` for BSE stocks (India)
    - Try the "Quick Select" buttons in the sidebar
    """)
