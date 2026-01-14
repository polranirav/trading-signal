# 💻 DUAL-CHART DASHBOARD - TECHNICAL IMPLEMENTATION GUIDE
## Step-by-Step Code & Architecture (Beginner to Advanced)

---

## PHASE 1: DATA FETCHING & STORAGE

### Module 1: data_fetcher.py

```python
import ccxt
import pandas as pd
from datetime import datetime
import time

class DataFetcher:
    """Fetch live candlestick data from exchange"""
    
    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)()
        
    def fetch_ohlcv(self, symbol='BTC/USD', timeframe='1h', limit=200):
        """
        Fetch candlesticks
        OHLCV = Open, High, Low, Close, Volume
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            return df
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def get_current_price(self, symbol='BTC/USD'):
        """Get latest price"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'price': ticker['close'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

# Usage
if __name__ == "__main__":
    fetcher = DataFetcher('binance')
    
    # Get historical data
    df = fetcher.fetch_ohlcv('BTC/USD', '1h', limit=200)
    print(df.head())
    
    # Get current price
    current = fetcher.get_current_price('BTC/USD')
    print(current)
```

---

## PHASE 2: FEATURE ENGINEERING

### Module 2: feature_engineer.py

```python
import pandas as pd
import talib
import numpy as np

class FeatureEngineer:
    """Calculate technical indicators"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def add_rsi(self, period=14):
        """Relative Strength Index"""
        self.df['rsi'] = talib.RSI(self.df['close'], timeperiod=period)
        return self.df
    
    def add_macd(self):
        """MACD indicator"""
        macd, signal, hist = talib.MACD(
            self.df['close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.df['macd'] = macd
        self.df['macd_signal'] = signal
        self.df['macd_hist'] = hist
        return self.df
    
    def add_bollinger_bands(self, period=20):
        """Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            self.df['close'],
            timeperiod=period
        )
        self.df['bb_upper'] = upper
        self.df['bb_middle'] = middle
        self.df['bb_lower'] = lower
        self.df['bb_width'] = upper - lower
        return self.df
    
    def add_moving_averages(self):
        """Moving Averages"""
        self.df['ma_20'] = talib.SMA(self.df['close'], timeperiod=20)
        self.df['ma_50'] = talib.SMA(self.df['close'], timeperiod=50)
        self.df['ma_200'] = talib.SMA(self.df['close'], timeperiod=200)
        return self.df
    
    def add_atr(self, period=14):
        """Average True Range (volatility)"""
        self.df['atr'] = talib.ATR(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            timeperiod=period
        )
        return self.df
    
    def add_obv(self):
        """On-Balance Volume"""
        self.df['obv'] = talib.OBV(self.df['close'], self.df['volume'])
        return self.df
    
    def add_price_momentum(self):
        """Price momentum"""
        self.df['momentum'] = self.df['close'].pct_change() * 100  # %
        self.df['momentum_lag1'] = self.df['momentum'].shift(1)
        self.df['momentum_lag5'] = self.df['momentum'].shift(5)
        return self.df
    
    def add_volatility(self, period=20):
        """Rolling volatility"""
        returns = self.df['close'].pct_change()
        self.df['volatility'] = returns.rolling(period).std() * 100
        return self.df
    
    def add_labels(self, future_periods=1):
        """Create label: UP or DOWN for next candle"""
        self.df['target'] = (
            (self.df['close'].shift(-future_periods) > self.df['close']).astype(int)
        )
        # 1 = UP, 0 = DOWN
        return self.df
    
    def get_features(self):
        """Calculate all features"""
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_moving_averages()
        self.add_atr()
        self.add_obv()
        self.add_price_momentum()
        self.add_volatility()
        self.add_labels()
        
        # Remove NaN rows (first few rows have NaN)
        self.df = self.df.dropna()
        
        return self.df

# Usage
if __name__ == "__main__":
    # Assume df is your candlestick data
    engineer = FeatureEngineer(df)
    df_features = engineer.get_features()
    print(df_features.head())
    print(df_features.columns)
```

---

## PHASE 3: MODEL TRAINING

### Module 3: ml_model.py

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import pandas as pd

class TradingModel:
    """XGBoost model for price prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_data(self, df, test_size=0.2):
        """Prepare training data"""
        # Select features (exclude timestamp and target)
        feature_cols = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width',
            'ma_20', 'ma_50', 'ma_200',
            'atr', 'obv', 'momentum', 'volatility',
            'volume'
        ]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, df, epochs=100):
        """Train XGBoost model"""
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Initialize XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=epochs,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    
    def predict(self, X):
        """Make prediction"""
        X_scaled = self.scaler.transform(X)
        
        # Get probability (confidence)
        proba = self.model.predict_proba(X_scaled)[0]
        
        return {
            'direction': 'UP' if proba[1] > 0.5 else 'DOWN',
            'confidence': max(proba) * 100,
            'proba_up': proba[1] * 100,
            'proba_down': proba[0] * 100
        }
    
    def predict_next_hours(self, current_data, num_hours=5):
        """Predict next N hours"""
        predictions = []
        
        for i in range(1, num_hours + 1):
            # Use most recent data
            last_row = current_data.iloc[-i:].copy()
            
            # Extract features
            feature_cols = self.feature_columns
            X = last_row[feature_cols]
            
            # Predict
            pred = self.predict(X)
            pred['hour'] = i
            predictions.append(pred)
        
        return predictions
    
    def save_model(self, filepath='model.pkl'):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath='model.pkl'):
        """Load trained model"""
        self.model = joblib.load(filepath)

# Usage
if __name__ == "__main__":
    model = TradingModel()
    metrics = model.train(df_features, epochs=100)
    
    # Make prediction
    current_features = df_features.iloc[-1:][feature_cols]
    pred = model.predict(current_features)
    print(pred)
    
    # Save model
    model.save_model('trading_model.pkl')
```

---

## PHASE 4: DASH DASHBOARD

### Module 4: app.py (Main Application)

```python
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import threading

from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from ml_model import TradingModel

# Initialize
fetcher = DataFetcher('binance')
model = TradingModel()
model.load_model('trading_model.pkl')

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("📊 Dual-Chart Prediction Dashboard"),
            html.P("Live Market Data + ML Predictions")
        ], style={'textAlign': 'center', 'marginBottom': '20px'})
    ]),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Timeframe:"),
            dcc.Dropdown(
                id='timeframe-select',
                options=[
                    {'label': '15 Minutes', 'value': '15m'},
                    {'label': '1 Hour', 'value': '1h'},
                    {'label': '4 Hours', 'value': '4h'},
                    {'label': '1 Day', 'value': '1d'}
                ],
                value='1h',
                style={'width': '150px'}
            )
        ], style={'marginRight': '30px'}),
        
        html.Div([
            html.Label("Asset:"),
            dcc.Dropdown(
                id='asset-select',
                options=[
                    {'label': 'Bitcoin (BTC/USD)', 'value': 'BTC/USD'},
                    {'label': 'Ethereum (ETH/USD)', 'value': 'ETH/USD'},
                    {'label': 'Binance (BNB/USD)', 'value': 'BNB/USD'}
                ],
                value='BTC/USD',
                style={'width': '200px'}
            )
        ]),
        
        dcc.Interval(
            id='live-update-interval',
            interval=5000,  # 5 seconds
            n_intervals=0
        ),
        
        dcc.Interval(
            id='prediction-update-interval',
            interval=3600000,  # 1 hour
            n_intervals=0
        )
    ], style={'display': 'flex', 'marginBottom': '20px', 'padding': '10px'}),
    
    # Main content
    html.Div([
        # Left side: Live data
        html.Div([
            html.Div([
                html.H3("📍 Live Market Data"),
                html.Div(id='price-info')
            ], style={'marginBottom': '20px'}),
            
            dcc.Graph(id='live-chart')
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        # Right side: Predictions
        html.Div([
            html.Div([
                html.H3("🎯 ML Predictions (Next 5 Hours)"),
                html.Div(id='predictions-info')
            ], style={'marginBottom': '20px'}),
            
            dcc.Graph(id='prediction-chart'),
            
            html.Div([
                html.P("Last updated: ", id='last-update')
            ], style={'marginTop': '10px', 'fontSize': '12px'})
        ], style={'width': '48%', 'display': 'inline-block'})
    ])
], style={'padding': '20px', 'fontFamily': 'Arial'})

# Callbacks
@app.callback(
    [Output('live-chart', 'figure'),
     Output('price-info', 'children')],
    [Input('live-update-interval', 'n_intervals')],
    [State('timeframe-select', 'value'),
     State('asset-select', 'value')]
)
def update_live_chart(n, timeframe, asset):
    """Update live candlestick chart"""
    try:
        # Fetch data
        df = fetcher.fetch_ohlcv(asset, timeframe, limit=100)
        
        if df is None:
            return go.Figure(), html.P("Error fetching data")
        
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['time'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=f'{asset} - {timeframe}',
            yaxis_title='Price (USD)',
            xaxis_title='Time',
            template='plotly_dark',
            height=500
        )
        
        # Price info
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        change = ((current_price - prev_price) / prev_price) * 100
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        
        info = html.Div([
            html.P(f"💰 Price: ${current_price:,.2f}", style={'margin': '5px'}),
            html.P(f"📈 Change: {change:+.2f}%", style={'margin': '5px', 'color': 'green' if change > 0 else 'red'}),
            html.P(f"⬆️ High: ${high:,.2f}", style={'margin': '5px'}),
            html.P(f"⬇️ Low: ${low:,.2f}", style={'margin': '5px'}),
            html.P(f"📊 Volume: {df['volume'].iloc[-1]:,.0f}", style={'margin': '5px'})
        ])
        
        return fig, info
    
    except Exception as e:
        return go.Figure(), html.P(f"Error: {str(e)}")

@app.callback(
    [Output('prediction-chart', 'figure'),
     Output('predictions-info', 'children'),
     Output('last-update', 'children')],
    [Input('prediction-update-interval', 'n_intervals')],
    [State('timeframe-select', 'value'),
     State('asset-select', 'value')]
)
def update_predictions(n, timeframe, asset):
    """Update predictions"""
    try:
        # Fetch data
        df = fetcher.fetch_ohlcv(asset, timeframe, limit=200)
        
        if df is None:
            return go.Figure(), html.P("Error fetching data"), ""
        
        # Calculate features
        engineer = FeatureEngineer(df)
        df_features = engineer.get_features()
        
        # Make predictions for next 5 hours
        predictions = model.predict_next_hours(df_features, num_hours=5)
        
        # Create prediction display
        pred_items = []
        pred_dirs = []
        pred_confs = []
        
        for i, pred in enumerate(predictions, 1):
            direction_symbol = '🟢' if pred['direction'] == 'UP' else '🔴'
            confidence = pred['confidence']
            
            pred_items.append(
                html.P(
                    f"{direction_symbol} Hour {i}: {pred['direction']} ({confidence:.1f}%)",
                    style={'margin': '8px', 'fontSize': '16px'}
                )
            )
            
            pred_dirs.append(1 if pred['direction'] == 'UP' else 0)
            pred_confs.append(confidence)
        
        # Average confidence
        avg_confidence = sum(pred_confs) / len(pred_confs)
        
        pred_items.append(html.Hr())
        pred_items.append(
            html.P(
                f"📊 Average Confidence: {avg_confidence:.1f}%",
                style={'margin': '8px', 'fontWeight': 'bold', 'fontSize': '14px'}
            )
        )
        
        # Recommendation
        ups = sum(1 for d in pred_dirs if d == 1)
        if ups >= 3:
            recommendation = "✅ BUY SIGNAL"
            color = 'green'
        elif ups <= 2:
            recommendation = "❌ SELL SIGNAL"
            color = 'red'
        else:
            recommendation = "⚠️ NEUTRAL"
            color = 'orange'
        
        pred_items.append(
            html.P(
                f"Recommendation: {recommendation}",
                style={'margin': '8px', 'fontWeight': 'bold', 'color': color, 'fontSize': '14px'}
            )
        )
        
        # Prediction chart
        fig = go.Figure()
        
        hours = [f"H{i}" for i in range(1, 6)]
        fig.add_trace(go.Bar(
            x=hours,
            y=pred_confs,
            marker_color=['green' if d == 1 else 'red' for d in pred_dirs],
            name='Confidence (%)',
            text=[f"{c:.0f}%" for c in pred_confs],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Prediction Confidence by Hour',
            yaxis_title='Confidence (%)',
            xaxis_title='Prediction Window',
            template='plotly_dark',
            height=300,
            showlegend=False
        )
        
        # Last update time
        last_update = f"Last trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return fig, html.Div(pred_items), last_update
    
    except Exception as e:
        return go.Figure(), html.P(f"Error: {str(e)}"), ""

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```

---

## PHASE 5: REQUIREMENTS & SETUP

### requirements.txt

```
dash==2.14.0
plotly==5.17.0
pandas==2.0.0
numpy==1.24.0
xgboost==2.0.0
scikit-learn==1.3.0
ccxt==3.0.0
joblib==1.3.0
talib==0.4.26
python-dateutil==2.8.2
```

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py

# Access at http://localhost:8050
```

---

## PHASE 6: MONITORING & IMPROVEMENT

### Module 5: monitor.py

```python
import pandas as pd
from datetime import datetime
import json

class PerformanceMonitor:
    """Track prediction performance"""
    
    def __init__(self, log_file='predictions.json'):
        self.log_file = log_file
        self.records = []
    
    def log_prediction(self, timestamp, asset, timeframe, 
                      prediction, actual, confidence):
        """Log a prediction"""
        record = {
            'timestamp': str(timestamp),
            'asset': asset,
            'timeframe': timeframe,
            'prediction': prediction,
            'actual': actual,
            'correct': prediction == actual,
            'confidence': confidence
        }
        self.records.append(record)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.records, f, indent=2)
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.records:
            return {}
        
        df = pd.DataFrame(self.records)
        
        total = len(df)
        correct = df['correct'].sum()
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # By confidence level
        high_conf = df[df['confidence'] > 60]
        high_conf_acc = (high_conf['correct'].sum() / len(high_conf) * 100) if len(high_conf) > 0 else 0
        
        return {
            'total_predictions': total,
            'correct_predictions': correct,
            'overall_accuracy': f"{accuracy:.2f}%",
            'high_confidence_accuracy': f"{high_conf_acc:.2f}%",
            'last_updated': datetime.now().isoformat()
        }

# Usage
monitor = PerformanceMonitor()

# Log predictions
monitor.log_prediction(
    datetime.now(),
    'BTC/USD',
    '1h',
    'UP',
    'UP',  # actual price moved up
    62.5
)

# Check metrics
metrics = monitor.calculate_metrics()
print(metrics)
```

---

## YOUR CHECKLIST

```
☐ Phase 1: Data fetching working (can fetch candlesticks)
☐ Phase 2: Features calculated (have RSI, MACD, etc)
☐ Phase 3: Model trained (accuracy > 55%)
☐ Phase 4: Dashboard running (can see live chart + predictions)
☐ Phase 5: All libraries installed (no import errors)
☐ Phase 6: Monitoring metrics (tracking prediction accuracy)

NEXT STEPS:
☐ Run app.py (should see dashboard at localhost:8050)
☐ Test with paper trading for 1-2 weeks
☐ Monitor accuracy daily
☐ Retrain model weekly
☐ Adjust parameters if needed
☐ Scale up if confident
```

---

**You now have complete code for your dashboard. Start coding! 🚀**
