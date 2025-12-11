"""
Stress test script for complex supply chain scenarios.
Generates SQL and reasoning for difficult multi-step questions.
"""
import datetime
import json
import sys
from openai_clients import call_openai_sql_generator

# Hardcoded schema summary for testing
SCHEMA_SUMMARY = """
BM00101: ITEMNMBR (char), BILLSTAT (smallint), DESIGNID (char)
BM00111: ITEMNMBR (char), CMPTITNM (char), Design_Qty (numeric)
BM010115: PPN_I (char), CPN_I (char), QUANTITY_I (numeric), UOFM (char)
GL00100: ACTINDX (int), ACTNUMBR_1 (char), ACTDESCR (char)
GL20000: JRNENTRY (int), TRXDATE (datetime), CRDTAMNT (numeric), DEBITAMT (numeric)
IV00101: ITEMNMBR (char), ITEMDESC (char), ITMCLSCD (char), STNDCOST (numeric), CURRCOST (numeric)
IV00102: ITEMNMBR (char), LOCNCODE (char), QTYONHND (numeric), ATYALLOC (numeric)
IV30200: DOCNUMBR (char), IVDOCTYP (smallint), DOCDATE (datetime)
IV30300: DOCNUMBR (char), DOCTYPE (smallint), ITEMNMBR (char), TRXQTY (numeric), TRXLOCTN (char)
PM00200: VENDORID (char), VENDNAME (char)
PM30200: VENDORID (char), DOCNUMBR (char), DOCDATE (datetime), DOCAMNT (numeric)
POP10100: PONUMBER (char), VENDORID (char), DOCDATE (datetime)
POP10110: PONUMBER (char), ITEMNMBR (char), QTYORDER (numeric), UNITCOST (numeric)
POP30100: PONUMBER (char), VENDORID (char), RECEIPTDATE (datetime)
POP30110: PONUMBER (char), ITEMNMBR (char), QTYINVCD (numeric), UNITCOST (numeric)
SOP10100: SOPNUMBE (char), SOPTYPE (smallint), DOCDATE (datetime), CUSTNMBR (char)
SOP10200: SOPNUMBE (char), SOPTYPE (smallint), ITEMNMBR (char), QUANTITY (numeric), XTNDPRCE (numeric)
SOP30200: SOPNUMBE (char), SOPTYPE (smallint), DOCDATE (datetime), CUSTNMBR (char)
SOP30300: SOPNUMBE (char), SOPTYPE (smallint), ITEMNMBR (char), QUANTITY (numeric), EXTDCOST (numeric), XTNDPRCE (numeric)
"""

COMPLEX_SCENARIOS = [
    "If we increase production of SOARBLM02 by 20% next month, what raw materials will we be short on given current inventory and open POs?",
    "Compare the vendor performance of 'CHEMSUPPLY' vs 'RAWMATINC' based on price variance and on-time delivery over the last year.",
    "Calculate the optimal reorder point for NPKACEK based on the average daily usage over the last 6 months and a 14-day lead time.",
    "Show me a breakdown of total spend by item category for Q3, but exclude any one-time purchases under $500.",
    "Identify items where the standard cost is more than 10% different from the last purchase price, and show the potential financial impact based on current stock.",
    "Forecast demand for the next 3 months for all 'RAW' category items based on a 3-month moving average of usage, and list the ones where we will run out of stock.",
]

def run_stress_test():
    today = datetime.date(2025, 12, 1)
    print(f"Running Stress Test for {len(COMPLEX_SCENARIOS)} Complex Scenarios")
    print(f"Date: {today}")
    print("=" * 80)

    for i, prompt in enumerate(COMPLEX_SCENARIOS, 1):
        print(f"\nScenario {i}: {prompt}")
        print("-" * 80)
        
        try:
            result = call_openai_sql_generator(
                prompt=prompt,
                today=today,
                schema_summary=SCHEMA_SUMMARY
            )
            
            if not result:
                print("❌ Failed to generate SQL.")
                continue

            print("✅ SQL Generated:")
            print(result.get("sql"))
            
            if "reasoning" in result:
                print("\n🧠 Reasoning Steps:")
                for step in result["reasoning"]:
                    print(f"  - {step}")
            
            print(f"\n📝 Summary: {result.get('summary')}")
            
            # Basic validation
            sql = result.get("sql", "").upper()
            if "IS NOT NULL" in sql or "IS NULL" in sql:
                 print("  [Check] NULL handling detected.")
            if "WITH" in sql:
                 print("  [Check] CTE used for complexity.")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_stress_test()
