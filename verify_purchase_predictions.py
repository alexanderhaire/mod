import pyodbc
import pandas as pd
import datetime
import logging
from procurement_ml import ProcurementFeatureBuilder, BuyWindowPredictor
from db_pool import get_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("PredictionVerifier")

def verify_predictions(days_back=90):
    """
    Backtest the model against actual user purchases.
    """
    print(f"--- Verifying Model Accuracy vs Actual Purchases (Last {days_back} Days) ---")
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Fetch recent purchases (Check BOTH History and Open/Work tables)
        query = """
        SELECT 
            h.PONUMBER,
            h.DOCDATE,
            l.ITEMNMBR,
            l.ITEMDESC,
            l.UNITCOST,
            l.QTYORDER,
            h.VENDORID,
            'HISTORY' as Source
        FROM POP30100 h
        JOIN POP30110 l ON h.PONUMBER = l.PONUMBER
        WHERE h.DOCDATE >= DATEADD(day, ?, GETDATE())
          AND l.ITEMNMBR <> ''
        
        UNION ALL
        
        SELECT 
            h.PONUMBER,
            h.DOCDATE,
            l.ITEMNMBR,
            l.ITEMDESC,
            l.UNITCOST,
            l.QTYORDER,
            h.VENDORID,
            'OPEN' as Source
        FROM POP10100 h
        JOIN POP10110 l ON h.PONUMBER = l.PONUMBER
        WHERE h.DOCDATE >= DATEADD(day, ?, GETDATE())
          AND l.ITEMNMBR <> ''
          
        ORDER BY DOCDATE DESC
        """
        cursor.execute(query, (-days_back, -days_back))
        purchases = cursor.fetchall()
        
        if not purchases:
            print("No purchases found in the specified period (checked both History and Open tables).")
            return

        print(f"Analyzing {len(purchases)} purchase events...")
        
        # 2. Initialize Model Components
        feature_builder = ProcurementFeatureBuilder(cursor)
        predictor = BuyWindowPredictor()
        
        # Mock training with some data if needed (or load model if it persists)
        # For this verification, we assume the model logic (even if rule-based fallback) 
        # is what we want to test.
        
        results = []
        
        for i, row in enumerate(purchases):
            po_num = row.PONUMBER
            buy_date = row.DOCDATE
            if isinstance(buy_date, str):
                buy_date = datetime.datetime.fromisoformat(buy_date).date()
            elif isinstance(buy_date, datetime.datetime):
                buy_date = buy_date.date()
                
            item = row.ITEMNMBR.strip()
            desc = row.ITEMDESC.strip() if row.ITEMDESC else item
            
            # We want to know: On the day you bought it, did the model agree?
            # And: Did it see it coming 3 days before?
            
            # Check "Day Of Deal"
            try:
                features = feature_builder.build_features(item, as_of_date=buy_date)
                prediction = predictor.predict(features)
                
                score = prediction['buy_score']
                rec = prediction['recommendation']
                factors = prediction['top_factors']
                
                results.append({
                    "Date": buy_date,
                    "Item": item,
                    "Description": desc,
                    "Action": "BOUGHT",
                    "Model Score": score,
                    "Model Rec": rec,
                    "Top Factor": factors[0] if factors else "None"
                })
                
                if i % 5 == 0:
                    print(f"Processed {i}/{len(purchases)}: {item} -> Score {score:.1f}")
                    
            except Exception as e:
                LOGGER.error(f"Failed to analyze {item}: {e}")
                
        # 3. Analyze Results
        df = pd.DataFrame(results)
        
        print("\n--- PERFORMANCE SUMMARY ---")
        
        # Define "Success" as Score >= 60 (Good to Buy)
        df['Model Approved'] = df['Model Score'] >= 60
        
        approval_rate = df['Model Approved'].mean()
        avg_score = df['Model Score'].mean()
        
        print(f"Total Purchases Analyzed: {len(df)}")
        print(f"Model Approval Rate:      {approval_rate:.1%} (Times model said 'YES' when you bought)")
        print(f"Average Buy Score:        {avg_score:.1f}/100")
        
        print("\n--- TOP AGREEMENTS (Model Validated You) ---")
        top_agreements = df[df['Model Approved']].sort_values('Model Score', ascending=False).head(5)
        if not top_agreements.empty:
            print(top_agreements[['Date', 'Item', 'Model Score', 'Top Factor']].to_string(index=False))
        else:
            print("None.")
            
        print("\n--- TOP DISAGREEMENTS (Model thought you shouldn't buy) ---")
        disagreements = df[~df['Model Approved']].sort_values('Model Score', ascending=True).head(5)
        if not disagreements.empty:
            print(disagreements[['Date', 'Item', 'Model Score', 'Top Factor']].to_string(index=False))
        else:
            print("None.")
            
        # Export for detailed review
        df.to_csv("prediction_accuracy_report.csv", index=False)
        print("\nDetailed report saved to 'prediction_accuracy_report.csv'")

if __name__ == "__main__":
    verify_predictions()
