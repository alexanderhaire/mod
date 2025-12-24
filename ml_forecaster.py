"""ML Forecasting Engine for Commodity Returns

This module implements machine learning models to predict commodity returns
using proprietary ERP supply chain signals and external market data.

Key Models:
1. LSTM: Time-series forecasting for 5-day ahead returns
2. XGBoost: Market regime classification
3. Random Forest: Cross-asset mispricing detection
"""

import logging
from typing import Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

LOGGER = logging.getLogger(__name__)


class FeatureEngineer:
    """Extract predictive features from ERP purchasing data and market data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def engineer_erp_features(self, df_prices: pd.DataFrame, df_volumes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create predictive features from ERP price history.
        
        Args:
            df_prices: DataFrame with Date index and item columns (prices)
            df_volumes: Optional DataFrame with purchase volumes
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df_prices.index)
        
        for col in df_prices.columns:
            # Price momentum features
            features[f'{col}_return_1d'] = df_prices[col].pct_change(1)
            features[f'{col}_return_7d'] = df_prices[col].pct_change(7)
            features[f'{col}_return_30d'] = df_prices[col].pct_change(30)
            
            # Volatility features
            features[f'{col}_vol_7d'] = df_prices[col].pct_change().rolling(7).std()
            features[f'{col}_vol_30d'] = df_prices[col].pct_change().rolling(30).std()
            
            # Trend features
            features[f'{col}_sma_7_30_ratio'] = (
                df_prices[col].rolling(7).mean() / df_prices[col].rolling(30).mean()
            )
            
            # Price acceleration (rate of change of returns)
            returns = df_prices[col].pct_change()
            features[f'{col}_acceleration'] = returns.diff()
            
        # Volume features if available
        if df_volumes is not None:
            for col in df_volumes.columns:
                features[f'{col}_volume_trend'] = df_volumes[col].pct_change(7)
                features[f'{col}_volume_spike'] = (
                    df_volumes[col] / df_volumes[col].rolling(30).mean()
                )
        
        return features.fillna(0)
    
    def engineer_market_regime_features(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for regime detection (Bull/Bear/Choppy).
        
        Args:
            df_prices: Combined ERP + Futures price matrix
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=df_prices.index)
        
        # Portfolio-level features
        returns = df_prices.pct_change()
        
        features['portfolio_return'] = returns.mean(axis=1)
        features['portfolio_vol'] = returns.std(axis=1)
        features['portfolio_trend'] = returns.mean(axis=1).rolling(30).mean()
        
        # Cross-asset correlation
        features['avg_correlation'] = returns.rolling(30).corr().mean(axis=1).groupby(level=0).mean()
        
        # Dispersion (how much assets diverge)
        features['return_dispersion'] = returns.std(axis=1)
        
        # Drawdown from peak
        cumulative = (1 + returns.mean(axis=1)).cumprod()
        features['drawdown'] = cumulative / cumulative.cummax() - 1
        
        return features.fillna(0)


class LSTMForecaster:
    """LSTM model for time-series return forecasting."""
    
    def __init__(self, lookback_days: int = 30, forecast_horizon: int = 5):
        """
        Args:
            lookback_days: Number of historical days to use as input
            forecast_horizon: Number of days ahead to predict
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow required for LSTM. Install with: pip install tensorflow")
            
        self.lookback = lookback_days
        self.horizon = forecast_horizon
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, n_features: int) -> keras.Model:
        """Build LSTM architecture."""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.lookback, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.horizon, activation='linear')  # Predict returns
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, features: pd.DataFrame, target: pd.Series):
        """Create sliding window sequences for LSTM."""
        X, y = [], []
        
        for i in range(len(features) - self.lookback - self.horizon + 1):
            X.append(features.iloc[i:i+self.lookback].values)
            y.append(target.iloc[i+self.lookback:i+self.lookback+self.horizon].values)
            
        return np.array(X), np.array(y)
    
    def train(self, features: pd.DataFrame, target_returns: pd.Series, epochs: int = 50, batch_size: int = 32):
        """
        Train LSTM model.
        
        Args:
            features: Engineered features from FeatureEngineer
            target_returns: Future returns to predict
            epochs: Training epochs
            batch_size: Batch size for training
        """
        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.fit_transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # Prepare sequences
        X, y = self.prepare_sequences(features_scaled, target_returns)
        
        # Build model
        if self.model is None:
            self.model = self.build_model(n_features=X.shape[2])
        
        # Train with validation split
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        LOGGER.info(f"LSTM trained. Final loss: {history.history['loss'][-1]:.4f}")
        
        return history
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict future returns.
        
        Args:
            features: Recent features for prediction
            
        Returns:
            Array of predicted returns for next N days
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Scale and prepare
        features_scaled = pd.DataFrame(
            self.scaler.transform(features),
            index=features.index,
            columns=features.columns
        )
        
        # Take last lookback window
        X = features_scaled.iloc[-self.lookback:].values.reshape(1, self.lookback, -1)
        
        predictions = self.model.predict(X, verbose=0)
        return predictions[0]


class RegimeClassifier:
    """XGBoost-based market regime classifier."""
    
    def __init__(self):
        """Initialize regime classifier."""
        if not HAS_XGBOOST:
            LOGGER.warning("XGBoost not available. Using RandomForest fallback.")
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10)
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=3
            )
        
        self.scaler = StandardScaler()
        self.regime_map = {0: 'Bull', 1: 'Bear', 2: 'Choppy'}
        
    def label_regimes(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """
        Label historical data with regime classes.
        
        Args:
            returns: Portfolio returns
            volatility: Portfolio volatility
            
        Returns:
            Series of regime labels (0=Bull, 1=Bear, 2=Choppy)
        """
        labels = pd.Series(index=returns.index, dtype=int)
        
        # Bull: Positive returns + low vol
        bull_mask = (returns > 0.001) & (volatility < volatility.median())
        
        # Bear: Negative returns + high vol
        bear_mask = (returns < -0.001) & (volatility > volatility.median())
        
        # Choppy: Everything else
        labels[bull_mask] = 0
        labels[bear_mask] = 1
        labels[~(bull_mask | bear_mask)] = 2
        
        return labels
    
    def train(self, features: pd.DataFrame, regimes: pd.Series):
        """Train regime classifier."""
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled, regimes)
        LOGGER.info("Regime classifier trained.")
        
    def predict(self, features: pd.DataFrame) -> str:
        """Predict current market regime."""
        features_scaled = self.scaler.transform(features)
        regime_id = self.model.predict(features_scaled[-1:])
        return self.regime_map[int(regime_id[0])]


def train_forecasting_system(df_erp: pd.DataFrame, df_futures: pd.DataFrame) -> dict[str, Any]:
    """
    Train complete forecasting system.
    
    Args:
        df_erp: ERP price history
        df_futures: Futures price history
        
    Returns:
        Dictionary with trained models
    """
    LOGGER.info("Starting ML training pipeline...")
    
    # Merge data
    df_combined = pd.concat([df_erp, df_futures], axis=1).dropna()
    
    # Feature engineering
    engineer = FeatureEngineer()
    features = engineer.engineer_erp_features(df_combined)
    regime_features = engineer.engineer_market_regime_features(df_combined)
    
    # Train LSTM for each major commodity
    lstm_models = {}
    for ticker in ['Crude Oil (WTI)', 'Natural Gas (Henry Hub)', 'Urea (NOLA)']:
        if ticker in df_futures.columns:
            LOGGER.info(f"Training LSTM for {ticker}...")
            forecaster = LSTMForecaster(lookback_days=30, forecast_horizon=5)
            
            # Target is future returns
            target_returns = df_futures[ticker].pct_change().shift(-5)
            
            # Train
            forecaster.train(features.dropna(), target_returns.dropna(), epochs=30)
            lstm_models[ticker] = forecaster
    
    # Train regime classifier
    LOGGER.info("Training regime classifier...")
    regime_clf = RegimeClassifier()
    
    returns = df_combined.pct_change().mean(axis=1)
    volatility = df_combined.pct_change().std(axis=1)
    regimes = regime_clf.label_regimes(returns, volatility)
    
    regime_clf.train(regime_features.dropna(), regimes.dropna())
    
    return {
        'lstm_models': lstm_models,
        'regime_classifier': regime_clf,
        'feature_engineer': engineer
    }
