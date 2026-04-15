import pandas as pd
import pyodbc
from datetime import date
from secrets_loader import build_connection_string

def generate_financial_report():
    print("--- GENERATING FINANCIAL IMPACT REPORT ---")
    
    # 1. READ CSV (The adjustments we just generated)
    try:
        csv_file = "year_end_adjustments.csv"
        adj_df = pd.read_csv(csv_file)
        print(f"Loaded {len(adj_df)} adjustments from {csv_file}")
    except Exception as e:
        print(f"FAILED to read CSV: {e}")
        return

    # 2. GET GL CODES
    # We need to fetch GL codes again because they weren't in the CSV (standard GP import doesn't always need them if item master has them, but we need them for reporting)
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        query = """
        SELECT 
            T1.ITEMNMBR, 
            T3.ACTNUMST as GLCode,
            T4.ACTDESCR as GLDesc
        FROM IV00101 T1
        LEFT JOIN GL00105 T3 ON T1.IVIVINDX = T3.ACTINDX
        LEFT JOIN GL00100 T4 ON T1.IVIVINDX = T4.ACTINDX
        """
        gl_df = pd.read_sql(query, conn)
        conn.close()
        
        # Merge GL info
        # Standardize Item Number
        adj_df['Item Number'] = adj_df['Item Number'].astype(str).str.strip()
        gl_df['ITEMNMBR'] = gl_df['ITEMNMBR'].astype(str).str.strip()
        
        merged = pd.merge(adj_df, gl_df, left_on='Item Number', right_on='ITEMNMBR', how='left')
        
        # Calculate Amount
        merged['Amount'] = merged['Trx Qty'] * merged['Unit Cost']
        
        merged['GLCode'] = merged['GLCode'].fillna('UNKNOWN-GL')
        merged['GLDesc'] = merged['GLDesc'].fillna('Unknown Account')
        
    except Exception as e:
        print(f"FAILED to query DB: {e}")
        return

    # 3. GROUP BY GL
    gl_impact = merged.groupby(['GLCode', 'GLDesc'])['Amount'].sum().reset_index()
    gl_impact = gl_impact.sort_values('Amount')
    
    total_impact = gl_impact['Amount'].sum()
    
    # 4. GENERATE MARKDOWN REPORT
    report_content = f"""# Financial Impact Report - Year End Close 2025

**Generated**: {date.today()}
**Source**: `year_end_adjustments.csv`

## Summary
The proposed inventory adjustments will result in a total net P&L movement of: **${total_impact:,.2f}**

*(Negative = Inventory Write-Down / Expense Increase)*

## Journal Entry Preview

| Account | Description | Debit | Credit |
| :--- | :--- | :--- | :--- |
"""
    
    for _, row in gl_impact.iterrows():
        amount = row['Amount']
        if amount > 0:
            # Inventory Increase (Debit Inventory, Credit Adj Account)
            # But here we are showing the NET impact solely on inventory asset value
            # Actually, let's just list the net movement.
            debit = f"${amount:,.2f}"
            credit = "$0.00"
        else:
            # Inventory Decrease (Credit Inventory)
            debit = "$0.00"
            credit = f"${abs(amount):,.2f}"
            
        report_content += f"| **{row['GLCode']}** | {row['GLDesc']} | {debit} | {credit} |\n"

    report_content += f"""
| | **TOTAL NET CHANGE** | | **${total_impact:,.2f}** |

> [!NOTE]
> This table shows the net change to the Inventory Asset accounts. The offset (balancing entry) will be posted to your configured **Inventory Offset / Variance / Shrinkage** expense account automatically by the ERP upon posting.

## Top 10 Adjustments (By Value)
"""
    
    merged['AbsAmount'] = merged['Amount'].abs()
    top_10 = merged.sort_values('AbsAmount', ascending=False).head(10)
    
    report_content += "| Item | Qty Adj | Unit Cost | Total Impact |\n| :--- | :--- | :--- | :--- |\n"
    for _, row in top_10.iterrows():
        report_content += f"| {row['Item Number']} | {row['Trx Qty']:,.2f} | ${row['Unit Cost']:,.2f} | **${row['Amount']:,.2f}** |\n"

    # Write to File
    with open("financial_impact.md", "w") as f:
        f.write(report_content)
        
    print(gl_impact.to_string())
    print(f"\nReport generated: financial_impact.md")

if __name__ == "__main__":
    generate_financial_report()
