# Stock Forecasting using Machine Learning

A full-featured, production-ready **stock price prediction platform** with:
- Real-time stock data via `yfinance`
- 15+ Technical Indicators: RSI, MACD, Bollinger Bands, SMA, EMA, and more
- ML Models: Ensemble(A combination of all ML models), Random Forest, XGBoost, Gradient Boosting, Linear Regression, ARIMA, Prophet
- Performance Metrics: RMSE, R²
- Flexible date range selection
- Built for research, financial analysis, and predictive modeling

---

# Features

- Real-time price chart
- Technical indicators visualization
- Machine learning-based stock forecasting
- Live News with Sentiment Score (Work in Progress)
- Expandable dashboard with Streamlit

---

# Tech Stack

- Python 3.10
- Streamlit
- Scikit-learn
- XGBoost
- Prophet
- Matplotlib, Pandas, NumPy
- TA-Lib / `ta` for indicators

---

## 🛠️ Setup

```bash
git clone https://github.com/vivaanafk69/Stock-Forecasting-using-ML.git
cd Stock-Forecasting-using-ML
pip install -r requirements.txt
streamlit run Home.py
