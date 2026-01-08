"""
Cash Flow Analysis Module

Provides cash flow forecasting, payment terms analysis, and self-learning
prediction capabilities for procurement spend tracking.
"""

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyodbc

LOGGER = logging.getLogger(__name__)

# Path for storing learned payment patterns
PATTERNS_FILE = Path(__file__).parent / "data" / "payment_patterns.json"
PREDICTIONS_FILE = Path(__file__).parent / "data" / "cashflow_predictions.json"

from calendar_utils import get_week_start


@dataclass
class CashFlowError:
    """Structured error information for cash flow operations."""
    code: str
    message: str
    user_message: str
    details: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "error_code": self.code,
            "error_message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


class CashFlowException(Exception):
    """Exception with structured error information."""
    
    def __init__(self, error: CashFlowError):
        self.error = error
        super().__init__(error.message)


def _ensure_data_dir() -> None:
    """Ensure the data directory exists."""
    PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_patterns() -> dict:
    """Load previously learned payment patterns."""
    _ensure_data_dir()
    if PATTERNS_FILE.exists():
        try:
            with open(PATTERNS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            LOGGER.warning(f"Invalid JSON in payment patterns file: {e}")
        except PermissionError as e:
            LOGGER.error(f"Permission denied reading payment patterns: {e}")
        except Exception as e:
            LOGGER.warning(f"Could not load payment patterns: {e}")
    return {"vendors": {}, "last_updated": None}


def _save_patterns(patterns: dict) -> None:
    """Save learned payment patterns."""
    _ensure_data_dir()
    try:
        with open(PATTERNS_FILE, "w") as f:
            json.dump(patterns, f, indent=2, default=str)
    except PermissionError as e:
        LOGGER.error(f"Permission denied saving payment patterns: {e}")
    except Exception as e:
        LOGGER.error(f"Could not save payment patterns: {e}")


def _load_predictions() -> dict:
    """Load prediction history."""
    _ensure_data_dir()
    if PREDICTIONS_FILE.exists():
        try:
            with open(PREDICTIONS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            LOGGER.warning(f"Invalid JSON in predictions file: {e}")
        except PermissionError as e:
            LOGGER.error(f"Permission denied reading predictions: {e}")
        except Exception as e:
            LOGGER.warning(f"Could not load predictions: {e}")
    return {"predictions": [], "last_updated": None}


def _save_predictions(predictions: dict) -> None:
    """Save prediction history."""
    _ensure_data_dir()
    try:
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=2, default=str)
    except PermissionError as e:
        LOGGER.error(f"Permission denied saving predictions: {e}")
    except Exception as e:
        LOGGER.error(f"Could not save predictions: {e}")


def _handle_db_error(e: Exception, operation: str) -> CashFlowError:
    """Convert database exceptions to user-friendly error information."""
    error_str = str(e).lower()
    
    if isinstance(e, pyodbc.OperationalError):
        if "login" in error_str or "authentication" in error_str:
            return CashFlowError(
                code="AUTH_ERROR",
                message=str(e),
                user_message="Database authentication failed. Please check your credentials.",
                details=operation,
            )
        elif "network" in error_str or "connection" in error_str or "server" in error_str:
            return CashFlowError(
                code="CONNECTION_ERROR",
                message=str(e),
                user_message="Could not connect to the database. Please check your network connection.",
                details=operation,
            )
    elif isinstance(e, pyodbc.ProgrammingError):
        if "invalid object" in error_str or "invalid column" in error_str:
            return CashFlowError(
                code="SCHEMA_ERROR",
                message=str(e),
                user_message="Database schema mismatch. Required tables or columns may be missing.",
                details=operation,
            )
    elif isinstance(e, pyodbc.DataError):
        return CashFlowError(
            code="DATA_ERROR",
            message=str(e),
            user_message="Data format error. Some values could not be processed.",
            details=operation,
        )
    elif isinstance(e, pyodbc.IntegrityError):
        return CashFlowError(
            code="INTEGRITY_ERROR",
            message=str(e),
            user_message="Data integrity constraint violation.",
            details=operation,
        )
    
    # Generic database error
    return CashFlowError(
        code="DATABASE_ERROR",
        message=str(e),
        user_message=f"Database error during {operation}. Please try again later.",
        details=str(type(e).__name__),
    )


def get_cash_position_summary(cursor: pyodbc.Cursor) -> dict[str, Any]:
    """
    Get a summary of current cash position based on open POs and expected outflows.
    
    Returns a dictionary with:
        - next_30_day_outflow: Total cash due in next 30 days
        - next_60_day_outflow: Total cash due in days 31-60
        - open_po_count: Number of open purchase orders
        - potential_delay_savings_7day: Working capital savings if delayed 7 days
        - accuracy_score: Current prediction accuracy percentage
        - predictions_made: Total predictions made
        - vendors_with_patterns: Vendors with learned patterns
        - error: Error information if query failed (optional)
    """
    today = datetime.date.today()
    day_30 = today + datetime.timedelta(days=30)
    day_60 = today + datetime.timedelta(days=60)
    
    result = {
        "next_30_day_outflow": 0,
        "next_60_day_outflow": 0,
        "open_po_count": 0,
        "potential_delay_savings_7day": 0,
        "accuracy_score": 0,
        "predictions_made": 0,
        "vendors_with_patterns": 0,
        "error": None,
    }
    
    try:
        # Query open POs with expected payment dates
        # POP10100 = PO Header, POP10110 = PO Line
        query = """
        SELECT 
            h.PONUMBER,
            h.VENDORID,
            h.DUEDATE,
            h.PRMDATE,
            SUM(l.EXTDCOST) AS TotalCost
        FROM POP10100 h
        JOIN POP10110 l ON h.PONUMBER = l.PONUMBER
        WHERE h.POSTATUS IN (1, 2, 3)  -- New, Released, Change Order
        GROUP BY h.PONUMBER, h.VENDORID, h.DUEDATE, h.PRMDATE
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        total_30 = 0.0
        total_60 = 0.0
        po_count = 0
        
        for row in rows:
            po_count += 1
            due_date = row.DUEDATE if row.DUEDATE else row.PRMDATE
            if not due_date:
                continue
            
            # Convert to date if datetime
            if hasattr(due_date, 'date'):
                due_date = due_date.date()
            
            cost = float(row.TotalCost) if row.TotalCost else 0
            
            if due_date <= day_30:
                total_30 += cost
            elif due_date <= day_60:
                total_60 += cost
        
        result["next_30_day_outflow"] = total_30
        result["next_60_day_outflow"] = total_60
        result["open_po_count"] = po_count
        
        # Calculate potential savings from 7-day delay
        # Assuming 5% annual cost of capital
        cost_of_capital = 0.05
        daily_rate = cost_of_capital / 365
        delay_days = 7
        result["potential_delay_savings_7day"] = total_30 * daily_rate * delay_days
        
    except pyodbc.Error as e:
        error = _handle_db_error(e, "cash position analysis")
        LOGGER.error(f"Database error in get_cash_position_summary: {error.message}")
        result["error"] = error.to_dict()
    except Exception as e:
        LOGGER.exception(f"Unexpected error in get_cash_position_summary: {e}")
        result["error"] = CashFlowError(
            code="UNEXPECTED_ERROR",
            message=str(e),
            user_message="An unexpected error occurred while analyzing cash position.",
        ).to_dict()
    
    # Add accuracy metrics from learning system
    patterns = _load_patterns()
    predictions = _load_predictions()
    
    result["vendors_with_patterns"] = len(patterns.get("vendors", {}))
    result["predictions_made"] = len(predictions.get("predictions", []))
    
    # Calculate accuracy score
    if result["predictions_made"] > 0:
        preds = predictions.get("predictions", [])
        accurate = sum(1 for p in preds if abs(p.get("days_off", 999)) <= 3)
        result["accuracy_score"] = (accurate / len(preds)) * 100
    
    return result


def get_cash_forecast_summary(cursor: pyodbc.Cursor, days_ahead: int = 90) -> dict[str, Any]:
    """
    Get a cash outflow forecast broken down by week and vendor.
    
    Returns:
        - by_week: List of {WeekStart, Amount} dictionaries
        - by_vendor: List of {VENDORID, VENDNAME, Amount} dictionaries
        - total_outflow: Total forecasted outflow
        - po_count: Number of POs included
        - error: Error information if query failed (optional)
    """
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_ahead)
    
    result = {
        "by_week": [],
        "by_vendor": [],
        "total_outflow": 0,
        "po_count": 0,
        "error": None,
    }
    
    try:
        query = """
        SELECT 
            h.PONUMBER,
            h.VENDORID,
            v.VENDNAME,
            h.DUEDATE,
            h.PRMDATE,
            SUM(l.EXTDCOST) AS TotalCost
        FROM POP10100 h
        JOIN POP10110 l ON h.PONUMBER = l.PONUMBER
        LEFT JOIN PM00200 v ON h.VENDORID = v.VENDORID
        WHERE h.POSTATUS IN (1, 2, 3)
        GROUP BY h.PONUMBER, h.VENDORID, v.VENDNAME, h.DUEDATE, h.PRMDATE
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        weekly_totals: dict[str, float] = {}
        vendor_totals: dict[str, dict] = {}
        total = 0.0
        count = 0
        
        for row in rows:
            due_date = row.DUEDATE if row.DUEDATE else row.PRMDATE
            if not due_date:
                continue
            
            if hasattr(due_date, 'date'):
                due_date = due_date.date()
            
            if not (today <= due_date <= end_date):
                continue
            
            cost = float(row.TotalCost) if row.TotalCost else 0
            count += 1
            total += cost
            
            # Group by week
            week_start = get_week_start(due_date)
            week_key = week_start.isoformat()
            weekly_totals[week_key] = weekly_totals.get(week_key, 0) + cost
            
            # Group by vendor
            vendor_id = row.VENDORID.strip() if row.VENDORID else "UNKNOWN"
            vendor_name = row.VENDNAME.strip() if row.VENDNAME else vendor_id
            if vendor_id not in vendor_totals:
                vendor_totals[vendor_id] = {"VENDORID": vendor_id, "VENDNAME": vendor_name, "Amount": 0}
            vendor_totals[vendor_id]["Amount"] += cost
        
        result["by_week"] = [
            {"WeekStart": k, "Amount": v}
            for k, v in sorted(weekly_totals.items())
        ]
        result["by_vendor"] = sorted(
            vendor_totals.values(),
            key=lambda x: x["Amount"],
            reverse=True
        )[:10]  # Top 10 vendors
        result["total_outflow"] = total
        result["po_count"] = count
        
    except pyodbc.Error as e:
        error = _handle_db_error(e, "cash forecast")
        LOGGER.error(f"Database error in get_cash_forecast_summary: {error.message}")
        result["error"] = error.to_dict()
    except Exception as e:
        LOGGER.exception(f"Unexpected error in get_cash_forecast_summary: {e}")
        result["error"] = CashFlowError(
            code="UNEXPECTED_ERROR",
            message=str(e),
            user_message="An unexpected error occurred while generating cash forecast.",
        ).to_dict()
    
    return result


def analyze_payment_terms_impact(cursor: pyodbc.Cursor) -> pd.DataFrame:
    """
    Analyze vendor payment terms and their impact on cash flow.
    
    Returns a DataFrame with columns:
        - VENDORID, VENDNAME, PaymentTerms, DaysToPayment
        - Last12MonthSpend, AnnualCashFlowImpact
        
    If an error occurs, returns an empty DataFrame. Check LOGGER for details.
    """
    try:
        # Get vendor payment terms and spending
        query = """
        SELECT 
            v.VENDORID,
            v.VENDNAME,
            v.PYMTRMID AS PaymentTerms,
            COALESCE(t.DUEDTDS, 30) AS DaysToPayment,
            ISNULL(SUM(ph.DOCAMNT), 0) AS Last12MonthSpend
        FROM PM00200 v
        LEFT JOIN SY03300 t ON v.PYMTRMID = t.PYMTRMID
        LEFT JOIN PM30200 ph ON v.VENDORID = ph.VENDORID
            AND ph.DOCDATE >= DATEADD(month, -12, GETDATE())
        WHERE v.VENDSTTS = 1
        GROUP BY v.VENDORID, v.VENDNAME, v.PYMTRMID, t.DUEDTDS
        HAVING SUM(ph.DOCAMNT) > 0
        ORDER BY Last12MonthSpend DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        data = []
        for row in rows:
            days = int(row.DaysToPayment) if row.DaysToPayment else 30
            spend = float(row.Last12MonthSpend) if row.Last12MonthSpend else 0
            
            # Calculate cash flow impact vs Net 30 baseline
            # Positive = better (more days), Negative = worse (fewer days)
            baseline_days = 30
            days_diff = days - baseline_days
            
            # Annual impact based on 5% cost of capital
            cost_of_capital = 0.05
            daily_rate = cost_of_capital / 365
            annual_impact = spend * daily_rate * days_diff
            
            data.append({
                "VENDORID": row.VENDORID.strip() if row.VENDORID else "",
                "VENDNAME": row.VENDNAME.strip() if row.VENDNAME else "",
                "PaymentTerms": row.PaymentTerms.strip() if row.PaymentTerms else "Unknown",
                "DaysToPayment": days,
                "Last12MonthSpend": spend,
                "AnnualCashFlowImpact": annual_impact,
            })
        
        return pd.DataFrame(data)
        
    except pyodbc.Error as e:
        error = _handle_db_error(e, "payment terms analysis")
        LOGGER.error(f"Database error in analyze_payment_terms_impact: {error.message}")
        return pd.DataFrame()
    except Exception as e:
        LOGGER.exception(f"Unexpected error in analyze_payment_terms_impact: {e}")
        return pd.DataFrame()


def get_monthly_spend_comparison(cursor: pyodbc.Cursor, months: int = 24) -> pd.DataFrame:
    """
    Get monthly procurement spend with year-over-year comparison.
    
    Returns a DataFrame with columns:
        - Period, Year, Month, Spend, PriorYearSpend, YoYChange
        
    If an error occurs, returns an empty DataFrame. Check LOGGER for details.
    """
    try:
        query = f"""
        SELECT 
            YEAR(DOCDATE) AS Year,
            MONTH(DOCDATE) AS Month,
            SUM(DOCAMNT) AS Spend
        FROM PM30200
        WHERE DOCDATE >= DATEADD(month, -{months}, GETDATE())
        GROUP BY YEAR(DOCDATE), MONTH(DOCDATE)
        ORDER BY Year, Month
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        data = []
        spend_by_period: dict[tuple, float] = {}
        
        for row in rows:
            year = int(row.Year)
            month = int(row.Month)
            spend = float(row.Spend) if row.Spend else 0
            spend_by_period[(year, month)] = spend
        
        for (year, month), spend in spend_by_period.items():
            prior_year_spend = spend_by_period.get((year - 1, month), 0)
            yoy_change = ((spend - prior_year_spend) / prior_year_spend * 100) if prior_year_spend > 0 else 0
            
            period_str = f"{year}-{month:02d}"
            data.append({
                "Period": period_str,
                "Year": year,
                "Month": month,
                "Spend": spend,
                "PriorYearSpend": prior_year_spend,
                "YoYChange": yoy_change,
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(["Year", "Month"])
        return df
        
    except pyodbc.Error as e:
        error = _handle_db_error(e, "monthly spend comparison")
        LOGGER.error(f"Database error in get_monthly_spend_comparison: {error.message}")
        return pd.DataFrame()
    except Exception as e:
        LOGGER.exception(f"Unexpected error in get_monthly_spend_comparison: {e}")
        return pd.DataFrame()


def calculate_po_delay_savings(
    cursor: pyodbc.Cursor,
    delay_days: int = 7,
    cost_of_capital: float = 0.05
) -> dict[str, Any]:
    """
    Calculate potential working capital savings from delaying PO payments.
    
    Args:
        cursor: Database cursor
        delay_days: Number of days to delay payments
        cost_of_capital: Annual cost of capital (default 5%)
    
    Returns:
        Dictionary with savings calculations and optional error info
    """
    result = {
        "delay_days": delay_days,
        "cost_of_capital": cost_of_capital,
        "total_open_po_value": 0,
        "estimated_annual_savings": 0,
        "by_vendor": [],
        "error": None,
    }
    
    try:
        query = """
        SELECT 
            h.VENDORID,
            v.VENDNAME,
            SUM(l.EXTDCOST) AS TotalCost
        FROM POP10100 h
        JOIN POP10110 l ON h.PONUMBER = l.PONUMBER
        LEFT JOIN PM00200 v ON h.VENDORID = v.VENDORID
        WHERE h.POSTATUS IN (1, 2, 3)
        GROUP BY h.VENDORID, v.VENDNAME
        ORDER BY TotalCost DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()
        
        daily_rate = cost_of_capital / 365
        total = 0.0
        
        for row in rows:
            cost = float(row.TotalCost) if row.TotalCost else 0
            total += cost
            
            # Annualized savings from delay (rough estimate)
            # Assumes POs turn over roughly monthly
            savings = cost * daily_rate * delay_days * 12
            
            result["by_vendor"].append({
                "VENDORID": row.VENDORID.strip() if row.VENDORID else "",
                "VENDNAME": row.VENDNAME.strip() if row.VENDNAME else "",
                "OpenPOValue": cost,
                "EstimatedAnnualSavings": savings,
            })
        
        result["total_open_po_value"] = total
        result["estimated_annual_savings"] = total * daily_rate * delay_days * 12
        
    except pyodbc.Error as e:
        error = _handle_db_error(e, "PO delay savings calculation")
        LOGGER.error(f"Database error in calculate_po_delay_savings: {error.message}")
        result["error"] = error.to_dict()
    except Exception as e:
        LOGGER.exception(f"Unexpected error in calculate_po_delay_savings: {e}")
        result["error"] = CashFlowError(
            code="UNEXPECTED_ERROR",
            message=str(e),
            user_message="An unexpected error occurred while calculating savings.",
        ).to_dict()
    
    return result


def get_accuracy_report() -> dict[str, Any]:
    """
    Get a report on the prediction accuracy of the self-learning system.
    
    Returns:
        Dictionary with accuracy metrics and vendor patterns
    """
    patterns = _load_patterns()
    predictions = _load_predictions()
    
    pred_list = predictions.get("predictions", [])
    
    # Calculate overall accuracy
    total_predictions = len(pred_list)
    
    if total_predictions == 0:
        return {
            "accuracy_score": 0,
            "total_predictions": 0,
            "avg_days_off": 0,
            "most_predictable_vendors": [],
            "least_predictable_vendors": [],
            "last_updated": patterns.get("last_updated"),
        }
    
    # Calculate average days off
    days_off_values = [abs(p.get("days_off", 0)) for p in pred_list]
    avg_days_off = sum(days_off_values) / len(days_off_values) if days_off_values else 0
    
    # Accuracy: predictions within 3 days are considered "accurate"
    accurate_count = sum(1 for d in days_off_values if d <= 3)
    accuracy_score = (accurate_count / total_predictions) * 100
    
    # Aggregate by vendor
    vendor_stats: dict[str, list] = {}
    for p in pred_list:
        vendor_id = p.get("vendor_id", "UNKNOWN")
        days_off = p.get("days_off", 0)
        if vendor_id not in vendor_stats:
            vendor_stats[vendor_id] = []
        vendor_stats[vendor_id].append(days_off)
    
    # Calculate vendor averages
    vendor_averages = []
    for vendor_id, offsets in vendor_stats.items():
        avg_offset = sum(offsets) / len(offsets) if offsets else 0
        vendor_averages.append({
            "vendor_id": vendor_id,
            "avg_days_offset": avg_offset,
            "data_points": len(offsets),
        })
    
    # Sort by predictability (lowest average offset = most predictable)
    vendor_averages.sort(key=lambda x: abs(x["avg_days_offset"]))
    
    return {
        "accuracy_score": accuracy_score,
        "total_predictions": total_predictions,
        "avg_days_off": avg_days_off,
        "most_predictable_vendors": vendor_averages[:5],
        "least_predictable_vendors": vendor_averages[-5:][::-1] if len(vendor_averages) > 5 else [],
        "last_updated": patterns.get("last_updated"),
    }


def record_prediction(
    vendor_id: str,
    po_number: str,
    predicted_date: datetime.date,
    actual_date: datetime.date | None = None
) -> None:
    """
    Record a payment prediction for learning purposes.
    
    Args:
        vendor_id: The vendor ID
        po_number: The PO number
        predicted_date: The predicted payment date
        actual_date: The actual payment date (if known)
    """
    predictions = _load_predictions()
    
    days_off = None
    if actual_date:
        days_off = (actual_date - predicted_date).days
    
    predictions.setdefault("predictions", []).append({
        "vendor_id": vendor_id,
        "po_number": po_number,
        "predicted_date": predicted_date.isoformat(),
        "actual_date": actual_date.isoformat() if actual_date else None,
        "days_off": days_off,
        "recorded_at": datetime.datetime.now().isoformat(),
    })
    predictions["last_updated"] = datetime.datetime.now().isoformat()
    
    _save_predictions(predictions)
    
    # Update vendor patterns if we have actuals
    if actual_date and days_off is not None:
        patterns = _load_patterns()
        vendor_data = patterns.setdefault("vendors", {}).setdefault(vendor_id, {
            "total_offset": 0,
            "count": 0,
            "average_offset": 0,
        })
        vendor_data["total_offset"] = vendor_data.get("total_offset", 0) + days_off
        vendor_data["count"] = vendor_data.get("count", 0) + 1
        vendor_data["average_offset"] = vendor_data["total_offset"] / vendor_data["count"]
        patterns["last_updated"] = datetime.datetime.now().isoformat()
        
        _save_patterns(patterns)


def get_vendor_adjustment(vendor_id: str) -> float:
    """
    Get the learned adjustment factor for a vendor's payment timing.
    
    Returns the average number of days this vendor typically pays
    earlier (negative) or later (positive) than predicted.
    """
    patterns = _load_patterns()
    vendor_data = patterns.get("vendors", {}).get(vendor_id, {})
    return vendor_data.get("average_offset", 0.0)
