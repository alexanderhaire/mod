"""
ML Procurement Optimizer

Machine learning module that learns from historical ERP data to predict
optimal buy windows and maximize asset value through intelligent procurement timing.

Features used:
- Price trends and volatility
- Usage patterns and seasonality  
- Inventory coverage
- Vendor credit terms
- Market signals
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyodbc

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

LOGGER = logging.getLogger(__name__)

# Paths for model persistence
MODEL_DIR = Path(__file__).parent / "data" / "ml_models"
PREDICTION_LOG_FILE = MODEL_DIR / "procurement_predictions.json"


@dataclass
class ProcurementFeatures:
    """Feature vector for a single item at a point in time."""
    item_number: str
    feature_date: datetime.date
    
    # Price features
    current_price: float = 0.0
    price_7d_change: float = 0.0
    price_30d_change: float = 0.0
    price_90d_change: float = 0.0
    price_percentile_52w: float = 0.5  # 0-1, where item is in 52-week range
    price_volatility_30d: float = 0.0
    price_trend_slope: float = 0.0
    
    # Seasonal features
    month_of_year: int = 1
    quarter: int = 1
    is_month_end: bool = False
    seasonal_price_index: float = 1.0  # This month vs annual avg
    
    # Inventory features
    days_of_coverage: float = 0.0
    usage_30d_avg: float = 0.0
    usage_trend_slope: float = 0.0
    usage_volatility: float = 0.0
    reorder_point_distance: float = 0.0  # % above/below reorder point
    
    # Vendor/Credit features
    vendor_payment_days: int = 30
    vendor_last_volume: float = 0.0
    days_since_last_buy: int = 999
    vendor_name: str = ""
    vendor_reliability_score: float = 1.0
    vendor_reliability_score: float = 1.0
    credit_cost_factor: float = 0.0  # Working capital cost
    vendor_reliability_score: float = 1.0
    credit_cost_factor: float = 0.0  # Working capital cost
    planning_lead_time: int = 14  # Days to receive goods
    actual_lead_time: int = 0 # Historical average lead time
    qty_on_order: float = 0.0 # Quantity currently on order
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.current_price,
            self.price_7d_change,
            self.price_30d_change,
            self.price_90d_change,
            self.price_percentile_52w,
            self.price_volatility_30d,
            self.price_trend_slope,
            self.month_of_year / 12.0,  # Normalize
            self.quarter / 4.0,
            float(self.is_month_end),
            self.seasonal_price_index,
            self.days_of_coverage / 90.0,  # Normalize to ~3 months
            self.usage_30d_avg,
            self.usage_trend_slope,
            self.usage_volatility,
            self.reorder_point_distance,
            self.vendor_payment_days / 60.0,  # Normalize
            self.vendor_reliability_score,
            self.credit_cost_factor,
        ])
    
    @staticmethod
    def feature_names() -> list[str]:
        """Return list of feature names for interpretability."""
        return [
            "current_price", "price_7d_change", "price_30d_change",
            "price_90d_change", "price_percentile_52w", "price_volatility_30d",
            "price_trend_slope", "month_normalized", "quarter_normalized",
            "is_month_end", "seasonal_price_index", "days_of_coverage_norm",
            "usage_30d_avg", "usage_trend_slope", "usage_volatility",
            "reorder_point_distance", "payment_days_norm", "vendor_reliability",
            "credit_cost_factor"
        ]


class ProcurementFeatureBuilder:
    """
    Extracts ML features from ERP data for procurement predictions.
    """
    
    def __init__(self, cursor: pyodbc.Cursor):
        self.cursor = cursor
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._usage_cache: dict[str, pd.DataFrame] = {}
        self._vendor_cache: dict[str, dict] = {}
    
    def build_features(
        self, 
        item_number: str, 
        as_of_date: datetime.date | None = None
    ) -> ProcurementFeatures:
        """
        Build feature vector for an item as of a specific date.
        
        Args:
            item_number: The GP item number
            as_of_date: Date for features (default: today)
        
        Returns:
            ProcurementFeatures with all computed values
        """
        as_of_date = as_of_date or datetime.date.today()
        
        features = ProcurementFeatures(
            item_number=item_number,
            feature_date=as_of_date,
            month_of_year=as_of_date.month,
            quarter=(as_of_date.month - 1) // 3 + 1,
            is_month_end=as_of_date.day >= 25,
        )
        
        # Get price history
        price_df = self._get_price_history(item_number, as_of_date)
        if not price_df.empty:
            features = self._compute_price_features(features, price_df, as_of_date)
        
        # Get usage history
        usage_df = self._get_usage_history(item_number, as_of_date)
        if not usage_df.empty:
            features = self._compute_usage_features(features, usage_df, as_of_date)
        
        # Get vendor info
        vendor_info = self._get_vendor_info(item_number)
        if vendor_info:
            features = self._compute_vendor_features(features, vendor_info)
        
        # Get inventory status
        inv_status = self._get_inventory_status(item_number)
        if inv_status:
            features = self._compute_inventory_features(features, inv_status, usage_df)
        
        return features
    
    def _get_price_history(
        self, 
        item_number: str, 
        as_of_date: datetime.date
    ) -> pd.DataFrame:
        """Fetch price history from purchase receipts."""
        cache_key = f"{item_number}_{as_of_date.year}"
        if cache_key in self._price_cache:
            df = self._price_cache[cache_key]
            return df[df['TransactionDate'] <= pd.Timestamp(as_of_date)]
        
        try:
            # POP30300 = Receipt Header, POP30310 = Receipt Lines (has item details)
            query = """
            SELECT 
                h.RECEIPTDATE AS TransactionDate,
                l.UNITCOST AS UnitCost,
                l.EXTDCOST,
                CASE WHEN l.UNITCOST > 0 THEN l.EXTDCOST / l.UNITCOST ELSE 0 END AS Quantity,
                h.VENDORID
            FROM POP30300 h
            JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
            WHERE l.ITEMNMBR = ?
              AND h.RECEIPTDATE >= DATEADD(year, -2, ?)
              AND h.RECEIPTDATE <= ?
              AND l.EXTDCOST > 0
            ORDER BY h.RECEIPTDATE
            """
            self.cursor.execute(query, (item_number, as_of_date, as_of_date))
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            data = [{
                'TransactionDate': row.TransactionDate,
                'UnitCost': float(row.UnitCost) if row.UnitCost else 0,
                'Quantity': float(row.Quantity) if row.Quantity else 0,
                'VendorId': row.VENDORID.strip() if row.VENDORID else ''
            } for row in rows]
            
            df = pd.DataFrame(data)
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            self._price_cache[cache_key] = df
            return df
            
        except Exception as e:
            LOGGER.warning(f"Error fetching price history for {item_number}: {e}")
            return pd.DataFrame()
    
    def _get_usage_history(
        self, 
        item_number: str, 
        as_of_date: datetime.date
    ) -> pd.DataFrame:
        """Fetch usage/consumption history from inventory transactions."""
        try:
            query = """
            SELECT 
                DOCDATE AS TransactionDate,
                ABS(TRXQTY) AS Quantity
            FROM IV30300
            WHERE ITEMNMBR = ?
              AND TRXLOCTN = 'MAIN'  -- Filter by main location
              AND DOCTYPE IN (1, 5)  -- Sales, Adjustments (out)
              AND DOCDATE >= DATEADD(year, -1, ?)
              AND DOCDATE <= ?
              AND TRXQTY < 0  -- Outflows
            ORDER BY DOCDATE
            """
            self.cursor.execute(query, (item_number, as_of_date, as_of_date))
            rows = self.cursor.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            data = [{
                'TransactionDate': row.TransactionDate,
                'Quantity': float(row.Quantity) if row.Quantity else 0,
            } for row in rows]
            
            df = pd.DataFrame(data)
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
            return df
            
        except Exception as e:
            LOGGER.warning(f"Error fetching usage history for {item_number}: {e}")
            return pd.DataFrame()
    
    def _get_vendor_info(self, item_number: str) -> dict | None:
        """Get primary vendor info including payment terms."""
        try:
            query = """
            SELECT TOP 1
                v.VENDORID,
                v.VENDNAME,
                v.PYMTRMID,
                COALESCE(t.DUEDTDS, 30) AS PaymentDays,
                iv.PLANNINGLEADTIME,
                (
                    SELECT TOP 1 l.EXTDCOST / NULLIF(l.UNITCOST, 0)
                    FROM POP30310 l
                    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
                    WHERE l.ITEMNMBR = iv.ITEMNMBR AND h.VENDORID = iv.VENDORID
                    ORDER BY h.RECEIPTDATE DESC
                ) AS LastVolume,
                (
                    SELECT AVG(DATEDIFF(day, po.DOCDATE, h.RECEIPTDATE))
                    FROM POP30310 l
                    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
                    JOIN POP30100 po ON l.PONUMBER = po.PONUMBER
                    WHERE l.ITEMNMBR = iv.ITEMNMBR 
                      AND h.VENDORID = iv.VENDORID 
                      AND h.RECEIPTDATE > DATEADD(year, -2, GETDATE())
                      AND l.PONUMBER <> ''
                      AND h.RECEIPTDATE >= po.DOCDATE
                ) AS AvgGapDays,
                (
                    SELECT TOP 1 h.RECEIPTDATE
                    FROM POP30310 l
                    JOIN POP30300 h ON l.POPRCTNM = h.POPRCTNM
                    WHERE l.ITEMNMBR = iv.ITEMNMBR AND h.VENDORID = iv.VENDORID
                    ORDER BY h.RECEIPTDATE DESC
                ) AS LastDate
            FROM IV00103 iv  -- Item Vendor Master
            JOIN PM00200 v ON iv.VENDORID = v.VENDORID
            LEFT JOIN SY03300 t ON v.PYMTRMID = t.PYMTRMID
            WHERE iv.ITEMNMBR = ?
            ORDER BY iv.LSTORDDT DESC  -- Most recent orders first
            """
            self.cursor.execute(query, (item_number,))
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'vendor_id': row.VENDORID.strip() if row.VENDORID else '',
                    'vendor_name': row.VENDNAME.strip() if row.VENDNAME else '',
                    'payment_days': int(row.PaymentDays) if row.PaymentDays else 30,
                    'lead_time': int(row.PLANNINGLEADTIME) if row.PLANNINGLEADTIME else 14,
                    'actual_lead_time': int(row.AvgGapDays) if row.AvgGapDays else 0,
                    'last_volume': float(row.LastVolume) if row.LastVolume else 0.0,
                    'last_date': row.LastDate,  # Date object
                }
            return None
            
        except Exception as e:
            LOGGER.warning(f"Error fetching vendor info for {item_number}: {e}")
            return None
    
    def _get_inventory_status(self, item_number: str) -> dict | None:
        """Get current inventory levels."""
        try:
            query = """
            SELECT 
                QTYONHND AS OnHand,
                QTYONORD AS OnOrder,
                ORDRPNTQTY AS ReorderPoint
            FROM IV00102
            WHERE ITEMNMBR = ?
              AND LOCNCODE = 'MAIN'  -- Specific to main location

            """
            self.cursor.execute(query, (item_number,))
            row = self.cursor.fetchone()
            
            if row:
                return {
                    'on_hand': float(row.OnHand) if row.OnHand else 0,
                    'on_order': float(row.OnOrder) if row.OnOrder else 0,
                    'reorder_point': float(row.ReorderPoint) if row.ReorderPoint else 0,
                }
            return None
            
        except Exception as e:
            LOGGER.warning(f"Error fetching inventory for {item_number}: {e}")
            return None
    
    def _compute_price_features(
        self, 
        features: ProcurementFeatures,
        price_df: pd.DataFrame,
        as_of_date: datetime.date
    ) -> ProcurementFeatures:
        """Compute price-related features from historical data."""
        # Current price (most recent)
        features.current_price = float(price_df['UnitCost'].iloc[-1])
        
        # Weighted average by quantity
        if price_df['Quantity'].sum() > 0:
            weighted_avg = (price_df['UnitCost'] * price_df['Quantity']).sum() / price_df['Quantity'].sum()
        else:
            weighted_avg = features.current_price
        
        # Price changes
        as_of_ts = pd.Timestamp(as_of_date)
        
        def avg_price_before(days: int) -> float:
            cutoff = as_of_ts - pd.Timedelta(days=days)
            subset = price_df[price_df['TransactionDate'] <= cutoff]
            if subset.empty:
                return features.current_price
            return float(subset['UnitCost'].mean())
        
        price_7d_ago = avg_price_before(7)
        price_30d_ago = avg_price_before(30)
        price_90d_ago = avg_price_before(90)
        
        if price_7d_ago > 0:
            features.price_7d_change = (features.current_price - price_7d_ago) / price_7d_ago
        if price_30d_ago > 0:
            features.price_30d_change = (features.current_price - price_30d_ago) / price_30d_ago
        if price_90d_ago > 0:
            features.price_90d_change = (features.current_price - price_90d_ago) / price_90d_ago
        
        # 52-week percentile
        year_ago = as_of_ts - pd.Timedelta(days=365)
        year_data = price_df[price_df['TransactionDate'] >= year_ago]['UnitCost']
        if len(year_data) > 1:
            min_price = year_data.min()
            max_price = year_data.max()
            if max_price > min_price:
                features.price_percentile_52w = (features.current_price - min_price) / (max_price - min_price)
        
        # 30-day volatility
        recent_30d = price_df[price_df['TransactionDate'] >= (as_of_ts - pd.Timedelta(days=30))]['UnitCost']
        if len(recent_30d) > 1:
            features.price_volatility_30d = float(recent_30d.std() / recent_30d.mean()) if recent_30d.mean() > 0 else 0
        
        # Trend slope (linear regression)
        if len(price_df) >= 3:
            price_df = price_df.copy()
            price_df['days'] = (price_df['TransactionDate'] - price_df['TransactionDate'].min()).dt.days
            if price_df['days'].nunique() >= 2:
                slope, _ = np.polyfit(price_df['days'], price_df['UnitCost'], 1)
                features.price_trend_slope = slope
        
        # Seasonal index (this month vs annual average)
        this_month = as_of_date.month
        month_prices = price_df[price_df['TransactionDate'].dt.month == this_month]['UnitCost']
        annual_avg = price_df['UnitCost'].mean()
        if annual_avg > 0 and len(month_prices) > 0:
            features.seasonal_price_index = float(month_prices.mean() / annual_avg)
        
        return features
    
    def _compute_usage_features(
        self,
        features: ProcurementFeatures,
        usage_df: pd.DataFrame,
        as_of_date: datetime.date
    ) -> ProcurementFeatures:
        """Compute usage/consumption features."""
        as_of_ts = pd.Timestamp(as_of_date)
        
        # 30-day average usage
        recent_30d = usage_df[usage_df['TransactionDate'] >= (as_of_ts - pd.Timedelta(days=30))]
        if not recent_30d.empty:
            features.usage_30d_avg = float(recent_30d['Quantity'].sum() / 30)
        
        # Usage trend
        if len(usage_df) >= 3:
            usage_df = usage_df.copy()
            usage_df['days'] = (usage_df['TransactionDate'] - usage_df['TransactionDate'].min()).dt.days
            if usage_df['days'].nunique() >= 2:
                slope, _ = np.polyfit(usage_df['days'], usage_df['Quantity'], 1)
                features.usage_trend_slope = slope
        
        # Usage volatility (coefficient of variation)
        if len(usage_df) > 1:
            cv = usage_df['Quantity'].std() / usage_df['Quantity'].mean() if usage_df['Quantity'].mean() > 0 else 0
            features.usage_volatility = float(cv)
        
        return features
    
    def _compute_vendor_features(
        self,
        features: ProcurementFeatures,
        vendor_info: dict
    ) -> ProcurementFeatures:
        """Compute vendor-related features."""
        features.vendor_payment_days = vendor_info.get('payment_days', 30)
        features.planning_lead_time = vendor_info.get('lead_time', 14)
        if features.planning_lead_time == 0:
            features.planning_lead_time = 14  # Default if 0 in DB
        
        features.actual_lead_time = vendor_info.get('actual_lead_time', 0)
        
        # Last purchase volume and recency
        features.vendor_last_volume = vendor_info.get('last_volume', 0.0)
        features.vendor_name = vendor_info.get('vendor_name', '')
        
        last_date = vendor_info.get('last_date')
        if last_date:
            if isinstance(last_date, datetime.datetime):
                last_date = last_date.date()
            delta = (features.feature_date - last_date).days
            features.days_since_last_buy = max(0, delta)
        
        # Credit cost factor: opportunity cost of payment terms
        # Net 30 = baseline (0), longer terms = negative cost (good)
        baseline_days = 30
        cost_of_capital = 0.05  # 5% annual
        daily_rate = cost_of_capital / 365
        days_diff = features.vendor_payment_days - baseline_days
        features.credit_cost_factor = -days_diff * daily_rate  # Negative = saves money
        
        return features
    
    def _compute_inventory_features(
        self,
        features: ProcurementFeatures,
        inv_status: dict,
        usage_df: pd.DataFrame
    ) -> ProcurementFeatures:
        """Compute inventory-related features."""
        on_hand = inv_status.get('on_hand', 0)
        reorder_point = inv_status.get('reorder_point', 0)
        features.qty_on_order = inv_status.get('on_order', 0.0)
        
        # Days of coverage
        if features.usage_30d_avg > 0:
            features.days_of_coverage = on_hand / features.usage_30d_avg
        elif reorder_point > 0:
            # Estimate from reorder point
            features.days_of_coverage = on_hand / (reorder_point / 30) if reorder_point > 0 else 90
        
        # Distance to reorder point (% above/below)
        if reorder_point > 0:
            features.reorder_point_distance = (on_hand - reorder_point) / reorder_point
        
        return features


class BuyWindowPredictor:
    """
    ML model that predicts optimal buy timing.
    
    Outputs:
    - buy_score: 0-100 score (higher = better time to buy)
    - expected_savings: % savings vs waiting
    - confidence: Model confidence in prediction
    """
    
    def __init__(self):
        self._model = None
        self._scaler = StandardScaler() if HAS_SKLEARN else None
        self._is_trained = False
        self._feature_importance: dict[str, float] = {}
    
    def train(
        self, 
        features_list: list[ProcurementFeatures],
        labels: list[float],  # 1.0 = good buy, 0.0 = bad buy
    ) -> dict[str, float]:
        """
        Train the model on historical data.
        
        Args:
            features_list: List of feature vectors
            labels: Target labels (was this a good time to buy?)
        
        Returns:
            Training metrics
        """
        if not HAS_SKLEARN:
            LOGGER.error("sklearn not available for training")
            return {"error": "sklearn not installed"}
        
        if len(features_list) < 50:
            LOGGER.warning(f"Only {len(features_list)} training samples, model may be unreliable")
        
        # Convert to arrays
        X = np.array([f.to_array() for f in features_list])
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)
        
        # Train Gradient Boosting model (good for tabular data)
        self._model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self._model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self._model.score(X_train_scaled, y_train)
        test_score = self._model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_names = ProcurementFeatures.feature_names()
        for name, importance in zip(feature_names, self._model.feature_importances_):
            self._feature_importance[name] = float(importance)
        
        self._is_trained = True
        
        LOGGER.info(f"Model trained: R² train={train_score:.3f}, test={test_score:.3f}")
        
        return {
            "train_r2": train_score,
            "test_r2": test_score,
            "n_samples": len(features_list),
            "feature_importance": dict(sorted(
                self._feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5
        }
    
    def predict(self, features: ProcurementFeatures) -> dict[str, Any]:
        """
        Predict buy score for given features.
        
        Returns:
            - buy_score: 0-100 (higher = better time to buy)
            - confidence: 0-1 confidence level
            - top_factors: Key drivers of the recommendation
        """
        if not self._is_trained:
            return self._rule_based_fallback(features)
        
        X = features.to_array().reshape(1, -1)
        X_scaled = self._scaler.transform(X)
        
        # Get prediction (0-1 range)
        raw_score = float(self._model.predict(X_scaled)[0])
        buy_score = max(0, min(100, raw_score * 100))
        
        # Confidence based on how extreme features are
        confidence = self._estimate_confidence(features)
        
        # Top factors
        top_factors = self._explain_prediction(features)
        
        return {
            "buy_score": buy_score,
            "confidence": confidence,
            "recommendation": self._score_to_recommendation(buy_score),
            "top_factors": top_factors,
        }
    
    def _rule_based_fallback(self, features: ProcurementFeatures) -> dict[str, Any]:
        """Fallback when model isn't trained."""
        score = 50.0  # Neutral baseline
        
        # Price signals
        if features.price_percentile_52w < 0.25:
            score += 20  # Low in 52-week range = good
        elif features.price_percentile_52w > 0.75:
            score -= 15  # High = bad
        
        if features.price_trend_slope < 0:
            score += 10  # Falling prices = wait might be good, but buy now locks in
        
        # Inventory urgency
        if features.days_of_coverage < 14:
            score += 30  # Must buy soon
        elif features.days_of_coverage < 30:
            score += 15
        elif features.days_of_coverage > 60:
            score -= 10  # Can wait
        
        # Credit terms
        if features.vendor_payment_days >= 45:
            score += 5  # Good terms
        
        score = max(0, min(100, score))
        
        return {
            "buy_score": score,
            "confidence": 0.5,  # Low confidence for rule-based
            "recommendation": self._score_to_recommendation(score),
            "top_factors": ["Rule-based fallback (model not trained)"],
        }
    
    def _score_to_recommendation(self, score: float) -> str:
        """Convert numeric score to recommendation."""
        if score >= 80:
            return "BUY NOW - Optimal window"
        elif score >= 60:
            return "GOOD TO BUY - Favorable conditions"
        elif score >= 40:
            return "FAIR - Can buy if needed"
        elif score >= 20:
            return "WAIT IF POSSIBLE - Conditions may improve"
        else:
            return "DELAY - Poor timing"
    
    def _estimate_confidence(self, features: ProcurementFeatures) -> float:
        """Estimate confidence based on feature quality."""
        confidence = 0.7  # Base confidence
        
        # Lower confidence if data is sparse
        if features.current_price == 0:
            confidence -= 0.3
        if features.days_of_coverage == 0:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _explain_prediction(self, features: ProcurementFeatures) -> list[str]:
        """Generate human-readable explanation of key factors."""
        factors = []
        
        if features.price_percentile_52w < 0.25:
            factors.append(f"Price in bottom 25% of 52-week range")
        elif features.price_percentile_52w > 0.75:
            factors.append(f"Price in top 25% of 52-week range")
        
        if features.days_of_coverage < 14:
            factors.append(f"Low inventory coverage ({features.days_of_coverage:.0f} days)")
        elif features.days_of_coverage > 60:
            factors.append(f"High inventory coverage ({features.days_of_coverage:.0f} days)")
        
        if features.price_trend_slope < -0.01:
            factors.append("Declining price trend")
        elif features.price_trend_slope > 0.01:
            factors.append("Rising price trend")
        
        if features.vendor_payment_days >= 45:
            factors.append(f"Favorable payment terms (Net {features.vendor_payment_days})")
        
        return factors[:3] if factors else ["Baseline conditions"]
    
    def save(self, path: Path | None = None) -> None:
        """Save model to disk."""
        import pickle
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = path or (MODEL_DIR / "buy_window_model.pkl")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self._model,
                'scaler': self._scaler,
                'feature_importance': self._feature_importance,
                'is_trained': self._is_trained,
            }, f)
        
        LOGGER.info(f"Model saved to {path}")
    
    def load(self, path: Path | None = None) -> bool:
        """Load model from disk."""
        import pickle
        
        path = path or (MODEL_DIR / "buy_window_model.pkl")
        
        if not path.exists():
            LOGGER.warning(f"Model file not found: {path}")
            return False
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self._model = data['model']
                self._scaler = data['scaler']
                self._feature_importance = data.get('feature_importance', {})
                self._is_trained = data.get('is_trained', True)
            
            LOGGER.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")
            return False


class ProcurementMLOptimizer:
    """
    High-level interface for ML-based procurement optimization.
    """
    
    def __init__(self, cursor: pyodbc.Cursor):
        self.cursor = cursor
        self.feature_builder = ProcurementFeatureBuilder(cursor)
        self.predictor = BuyWindowPredictor()
        
        # Try to load existing model
        self.predictor.load()
    
    def get_buy_recommendation(
        self, 
        item_number: str,
        as_of_date: datetime.date | None = None
    ) -> dict[str, Any]:
        """
        Get ML-based buy recommendation for an item.
        
        Returns:
            Dictionary with buy_score, recommendation, and factors
        """
        features = self.feature_builder.build_features(item_number, as_of_date)
        prediction = self.predictor.predict(features)
        
        # Calculate Must Buy Date
        # Stockout = Today + Coverage
        # Must Buy = Stockout - Lead Time - Safety (7 days)
        analysis_dt = as_of_date or datetime.date.today()
        days_coverage = features.days_of_coverage
        
        # Use Actual Lead Time if available and valid (>0), otherwise fallback to Planning
        lead_time = features.actual_lead_time if features.actual_lead_time > 0 else features.planning_lead_time
        
        safety_buffer = 7
        
        days_until_order = days_coverage - lead_time - safety_buffer
        must_buy_date = analysis_dt + datetime.timedelta(days=int(days_until_order))
        
        return {
            "item_number": item_number,
            "as_of_date": str(analysis_dt),
            "must_buy_date": str(must_buy_date),
            **prediction,
            "features": {
                "current_price": features.current_price,
                "price_52w_percentile": features.price_percentile_52w,
                "days_of_coverage": features.days_of_coverage,
                "lead_time": features.planning_lead_time,
                "vendor_payment_days": features.vendor_payment_days,
                "vendor_last_volume": features.vendor_last_volume,
                "vendor_name": features.vendor_name,
            }
        }
    
    def train_model(self, lookback_months: int = 24) -> dict[str, Any]:
        """
        Train the model on historical purchase data.
        
        Args:
            lookback_months: How many months of history to use
        
        Returns:
            Training metrics
        """
        LOGGER.info(f"Starting model training with {lookback_months} months of data")
        
        # Get historical purchases with outcomes
        training_data = self._build_training_dataset(lookback_months)
        
        if len(training_data['features']) < 50:
            return {"error": f"Insufficient training data: {len(training_data['features'])} samples"}
        
        # Train model
        metrics = self.predictor.train(
            training_data['features'],
            training_data['labels']
        )
        
        # Save trained model
        self.predictor.save()
        
        return metrics
    
    def _build_training_dataset(
        self, 
        lookback_months: int
    ) -> dict[str, list]:
        """
        Build labeled training dataset from historical purchases.
        
        Label: Was this purchase in the bottom 30% of prices for that period?
        """
        features_list = []
        labels = []
        
        try:
            # Get all purchases in the lookback period
            # POP30300 = Receipt Header, POP30310 = Receipt Lines
            query = f"""
            SELECT DISTINCT
                l.ITEMNMBR,
                h.RECEIPTDATE AS TRXDATE,
                l.UNITCOST
            FROM POP30300 h
            JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -{lookback_months}, GETDATE())
              AND l.EXTDCOST > 0
            ORDER BY l.ITEMNMBR, h.RECEIPTDATE
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            # Group by item
            item_purchases: dict[str, list] = {}
            for row in rows:
                item = row.ITEMNMBR.strip()
                if item not in item_purchases:
                    item_purchases[item] = []
                item_purchases[item].append({
                    'date': row.TRXDATE,
                    'price': float(row.UNITCOST) if row.UNITCOST else 0
                })
            
            # For each item, label purchases as good/bad
            for item_number, purchases in item_purchases.items():
                if len(purchases) < 5:
                    continue  # Need enough history
                
                # Calculate percentiles for labeling
                prices = [p['price'] for p in purchases if p['price'] > 0]
                if not prices:
                    continue
                    
                p30 = np.percentile(prices, 30)
                
                for purchase in purchases:
                    if purchase['price'] <= 0:
                        continue
                    
                    # Build features as of purchase date
                    try:
                        features = self.feature_builder.build_features(
                            item_number, 
                            purchase['date'].date() if hasattr(purchase['date'], 'date') else purchase['date']
                        )
                        
                        # Label: 1.0 if price was in bottom 30%, 0.0 otherwise
                        label = 1.0 if purchase['price'] <= p30 else 0.0
                        
                        features_list.append(features)
                        labels.append(label)
                        
                    except Exception as e:
                        LOGGER.debug(f"Error building features for {item_number}: {e}")
                        continue
            
            LOGGER.info(f"Built training dataset with {len(features_list)} samples")
            
        except Exception as e:
            LOGGER.error(f"Error building training dataset: {e}")
        
        return {'features': features_list, 'labels': labels}
    
    def get_batch_recommendations(
        self, 
        item_numbers: list[str],
        as_of_date: datetime.date | None = None
    ) -> pd.DataFrame:
        """
        Get recommendations for multiple items at once.
        
        Returns:
            DataFrame with item numbers and recommendations
        """
        results = []
        
        for item in item_numbers:
            try:
                rec = self.get_buy_recommendation(item, as_of_date=as_of_date)
                results.append({
                    'ItemNumber': item,
                    'BuyScore': rec['buy_score'],
                    'Recommendation': rec['recommendation'],
                    'Confidence': rec['confidence'],
                    'Price52wPct': rec['features']['price_52w_percentile'],
                    'DaysCoverage': rec['features']['days_of_coverage'],
                    'CurrentPrice': rec['features'].get('current_price', 0.0),
                    'MustBuyDate': rec['must_buy_date'],
                    'LeadTime': rec['features'].get('lead_time', 14),
                    'VendorName': rec['features'].get('vendor_name', ''),
                    'VendorLastVol': rec['features'].get('vendor_last_volume', 0.0),
                    'QtyOnOrder': rec['features'].get('qty_on_order', 0.0)
                })
            except Exception as e:
                LOGGER.warning(f"Error getting recommendation for {item}: {e}")
                results.append({
                    'ItemNumber': item,
                    'BuyScore': 50,
                    'Recommendation': 'Error',
                    'Confidence': 0,
                    'Price52wPct': None,
                    'DaysCoverage': None,
                })
        
        return pd.DataFrame(results).sort_values('BuyScore', ascending=False)


class WalkForwardValidator:
    """
    Walk-forward validation for procurement ML model.
    
    Performs rolling window backtesting with:
    - In-sample training
    - Out-of-sample prediction
    - Bootstrap confidence intervals for accuracy
    """
    
    def __init__(
        self, 
        cursor: pyodbc.Cursor,
        train_window_months: int = 12,
        test_window_months: int = 3,
        n_bootstrap: int = 1000
    ):
        self.cursor = cursor
        self.train_window = train_window_months
        self.test_window = test_window_months
        self.n_bootstrap = n_bootstrap
        self.feature_builder = ProcurementFeatureBuilder(cursor)
        self._validation_results: list[dict] = []
    
    def run_validation(
        self, 
        lookback_months: int = 24,
        progress_callback: callable = None
    ) -> dict[str, Any]:
        """
        Run walk-forward validation across historical data.
        
        Returns:
            - accuracy_in_sample: Training accuracy
            - accuracy_out_sample: True out-of-sample accuracy  
            - confidence_interval_95: (lower, upper) for OOS accuracy
            - confidence_interval_99: (lower, upper) for 99% confidence
            - is_significant_99: Whether 99% CI > 50% (better than random)
            - detailed_results: Per-period breakdown
        """
        LOGGER.info(f"Starting walk-forward validation with {lookback_months} months lookback")
        
        # Get all historical purchases
        all_data = self._load_all_purchases(lookback_months)
        
        if len(all_data) < 100:
            return {"error": f"Insufficient data: {len(all_data)} purchases (need 100+)"}
        
        # Calculate walk-forward splits
        dates = sorted(all_data.keys())
        n_periods = (lookback_months - self.train_window) // self.test_window
        
        if n_periods < 2:
            return {"error": "Not enough data for walk-forward validation"}
        
        all_predictions = []
        all_actuals = []
        period_results = []
        
        for period_idx in range(n_periods):
            if progress_callback:
                progress_callback(period_idx / n_periods)
            
            # Define train/test windows
            train_end_offset = self.train_window + (period_idx * self.test_window)
            test_end_offset = train_end_offset + self.test_window
            
            # Get train data
            train_data = self._filter_by_window(all_data, dates, 0, train_end_offset)
            test_data = self._filter_by_window(all_data, dates, train_end_offset, test_end_offset)
            
            if len(train_data['features']) < 30 or len(test_data['features']) < 10:
                continue
            
            # Train on in-sample
            predictor = BuyWindowPredictor()
            train_metrics = predictor.train(train_data['features'], train_data['labels'])
            
            if 'error' in train_metrics:
                continue
            
            # Predict on out-of-sample
            oos_predictions = []
            oos_actuals = []
            
            for features, label in zip(test_data['features'], test_data['labels']):
                pred = predictor.predict(features)
                # Convert buy_score to binary: score >= 60 = "should buy"
                predicted_buy = 1.0 if pred['buy_score'] >= 60 else 0.0
                oos_predictions.append(predicted_buy)
                oos_actuals.append(label)
            
            period_accuracy = np.mean(np.array(oos_predictions) == np.array(oos_actuals))
            
            period_results.append({
                'period': period_idx + 1,
                'train_samples': len(train_data['features']),
                'test_samples': len(test_data['features']),
                'train_r2': train_metrics.get('train_r2', 0),
                'oos_accuracy': period_accuracy,
            })
            
            all_predictions.extend(oos_predictions)
            all_actuals.extend(oos_actuals)
        
        if not all_predictions:
            return {"error": "No valid predictions generated"}
        
        # Calculate overall metrics
        predictions_arr = np.array(all_predictions)
        actuals_arr = np.array(all_actuals)
        overall_accuracy = np.mean(predictions_arr == actuals_arr)
        
        # Bootstrap confidence intervals
        ci_95 = self._bootstrap_ci(predictions_arr, actuals_arr, 0.95)
        ci_99 = self._bootstrap_ci(predictions_arr, actuals_arr, 0.99)
        
        # Statistical significance: is 99% CI lower bound > 0.5?
        is_significant = ci_99[0] > 0.5
        
        self._validation_results = period_results
        
        return {
            "accuracy_in_sample": np.mean([p['train_r2'] for p in period_results]),
            "accuracy_out_sample": overall_accuracy,
            "confidence_interval_95": ci_95,
            "confidence_interval_99": ci_99,
            "is_significant_99": is_significant,
            "total_predictions": len(all_predictions),
            "n_periods": len(period_results),
            "detailed_results": period_results,
        }
    
    def _load_all_purchases(self, months: int) -> dict[datetime.date, list]:
        """Load all purchases grouped by date."""
        purchases_by_date: dict[datetime.date, list] = {}
        
        try:
            query = f"""
            SELECT 
                l.ITEMNMBR,
                h.RECEIPTDATE,
                l.UNITCOST,
                CASE WHEN l.UNITCOST > 0 THEN l.EXTDCOST / l.UNITCOST ELSE 0 END AS ACTLSHIP
            FROM POP30300 h
            JOIN POP30310 l ON h.POPRCTNM = l.POPRCTNM
            WHERE h.RECEIPTDATE >= DATEADD(month, -{months}, GETDATE())
              AND l.EXTDCOST > 0
              AND l.UNITCOST > 0
            ORDER BY h.RECEIPTDATE
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()
            
            # Group by item to calculate percentiles
            item_prices: dict[str, list] = {}
            for row in rows:
                item = row.ITEMNMBR.strip()
                if item not in item_prices:
                    item_prices[item] = []
                item_prices[item].append(float(row.UNITCOST))
            
            # Calculate 30th percentile per item
            item_p30 = {item: np.percentile(prices, 30) for item, prices in item_prices.items() if len(prices) >= 3}
            
            # Build labeled dataset
            for row in rows:
                item = row.ITEMNMBR.strip()
                if item not in item_p30:
                    continue
                
                date = row.RECEIPTDATE.date() if hasattr(row.RECEIPTDATE, 'date') else row.RECEIPTDATE
                price = float(row.UNITCOST)
                
                # Label: 1.0 if this was a good buy (price <= 30th percentile)
                label = 1.0 if price <= item_p30[item] else 0.0
                
                if date not in purchases_by_date:
                    purchases_by_date[date] = []
                
                try:
                    features = self.feature_builder.build_features(item, date)
                    purchases_by_date[date].append({
                        'features': features,
                        'label': label,
                        'item': item,
                        'price': price,
                    })
                except:
                    pass
            
        except Exception as e:
            LOGGER.error(f"Error loading purchases: {e}")
        
        return purchases_by_date
    
    def _filter_by_window(
        self, 
        data: dict, 
        dates: list, 
        start_idx: int, 
        end_idx: int
    ) -> dict[str, list]:
        """Filter data by date window index."""
        features = []
        labels = []
        
        # Convert month offsets to date indices
        total_dates = len(dates)
        start_date_idx = int((start_idx / (end_idx + self.test_window)) * total_dates)
        end_date_idx = int((end_idx / (end_idx + self.test_window)) * total_dates)
        
        selected_dates = dates[start_date_idx:min(end_date_idx, total_dates)]
        
        for date in selected_dates:
            for record in data.get(date, []):
                features.append(record['features'])
                labels.append(record['label'])
        
        return {'features': features, 'labels': labels}
    
    def _bootstrap_ci(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray, 
        confidence: float
    ) -> tuple[float, float]:
        """Calculate bootstrap confidence interval for accuracy."""
        n = len(predictions)
        accuracies = []
        
        for _ in range(self.n_bootstrap):
            indices = np.random.randint(0, n, size=n)
            boot_acc = np.mean(predictions[indices] == actuals[indices])
            accuracies.append(boot_acc)
        
        alpha = 1 - confidence
        lower = np.percentile(accuracies, 100 * alpha / 2)
        upper = np.percentile(accuracies, 100 * (1 - alpha / 2))
        
        return (float(lower), float(upper))


class CriticalBuyFilter:
    """
    Filters items to only show those that MUST be bought today.
    
    Criteria for "Must Buy":
    - Buy score >= 80 (ML recommendation)
    - Confidence >= 0.7 (model is confident)
    - Days of coverage < 14 (running low)
    - OR price is at 52-week low AND coverage < 30
    """
    
    def __init__(self, confidence_threshold: float = 0.7, coverage_threshold: int = 14):
        self.confidence_threshold = confidence_threshold
        self.coverage_threshold = coverage_threshold
    
    def filter_critical(
        self, 
        recommendations: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter to only critical buy items.
        
        Args:
            recommendations: DataFrame from get_batch_recommendations()
        
        Returns:
            Filtered DataFrame with only "must buy today" items
        """
        if recommendations.empty:
            return recommendations
        
        # Criteria 1: High score + high confidence + low coverage
        mask_urgent = (
            (recommendations['BuyScore'] >= 80) &
            (recommendations['Confidence'] >= self.confidence_threshold) &
            (recommendations['DaysCoverage'] < self.coverage_threshold)
        )
        
        # Criteria 2: Price opportunity (52w low) with moderate coverage concern
        mask_opportunity = (
            (recommendations['Price52wPct'] <= 0.1) &  # Bottom 10% of 52-week range
            (recommendations['DaysCoverage'] < 30) &
            (recommendations['Confidence'] >= 0.6)
        )
        
        critical = recommendations[mask_urgent | mask_opportunity].copy()
        critical['CriticalReason'] = ''
        
        # Add reason
        critical.loc[mask_urgent, 'CriticalReason'] = 'LOW STOCK - Must buy'
        critical.loc[mask_opportunity & ~mask_urgent, 'CriticalReason'] = 'PRICE OPPORTUNITY - 52w low'
        
        return critical.sort_values('DaysCoverage', ascending=True)
    
    def get_summary(self, critical_items: pd.DataFrame) -> str:
        """Generate a summary message for critical items."""
        if critical_items.empty:
            return "No critical items requiring immediate purchase today."
        
        n_urgent = (critical_items['CriticalReason'] == 'LOW STOCK - Must buy').sum()
        n_opportunity = len(critical_items) - n_urgent
        
        summary = f"**{len(critical_items)} Critical Items for Today**\n"
        if n_urgent > 0:
            summary += f"- {n_urgent} items with low stock (must buy)\n"
        if n_opportunity > 0:
            summary += f"- {n_opportunity} items at price opportunity\n"
        
        return summary

