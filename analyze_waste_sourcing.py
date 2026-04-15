import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re

def analyze_waste_sourcing_v3():
    print("--- QUANTITATIVE RESEARCH: Industrial Waste Stream Sourcing (Final) ---")
    
    waste_map = {
        'SULFURIC': ['Copper Smelters', 'Zinc Smelters', 'Oil Refineries (Spent Acid)', 'Power Plants (Scrubbers)'],
        'ACID': ['Metal Pickling (Steel)', 'Semiconductor Etching', 'Fertilizer Plants'],
        'IRON': ['Steel Mills (Pickling Lines)', 'Titanium Dioxide Production (Copperas)', 'Scrap Metal Recyclers'],
        'FERROUS': ['Steel Mills (Pickling Lines)', 'Titanium Dioxide Production (Copperas)'],
        'ZINC': ['Galvanizing Plants (Dross/Skimmings)', 'Brass Mills', 'Electric Arc Furnace Dust'],
        'AMMONIA': ['Coke Plants (Steel)', 'Livestock Farms (unlikely for food grade, ok for fert)', 'Biogas Plants'],
        'NITROGEN': ['Haber Process By-products', 'Coke Ovens'],
        'GLYCEROL': ['Biodiesel Plants (Crude Glycerin)', 'Soap Manufacturing'],
        'PHOSPHOR': ['Phosphate Mining', 'Steel Coating (Phosphating)'],
        'POTASSIUM': ['Cement Kiln Dust (Potash)', 'Biomass Ash'],
        'ASH': ['Biomass Power Plants', 'Incinerators'],
        'LIME': ['Paper Mills', 'Acetylene Production (Carbide Lime)'],
        'CALCIUM': ['Paper Mills', 'Acetylene Production'],
        'MAGNESIUM': ['Desalination Plants (Bitterns)', 'Potash Mining'],
        'SULFATE': ['Battery Recycling', 'Viscose Rayon Production'],
        'CITRIC': ['Corn Fermentation (Cargill/ADM)', 'Ethanol Plants'],
        'UREA': ['Diesel Exhaust Fluid (DEF) Makers', 'Coal Gasification'],
        'MANGANESE': ['Battery Recycling', 'Steel Mills'],
        'COPPER': ['Circuit Board Etching (Spent Etchant)', 'Plating Shops']
    }

    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        print("Fetching Top Volume Chemicals...")
        
        query = """
        SELECT TOP 5000
            T1.ITEMNMBR,
            T1.ITEMDESC,
            T1.UOFM,
            T1.ACTLSHIP,
            T1.EXTDCOST
        FROM POP30310 T1
        JOIN POP30300 T2 ON T1.POPRCTNM = T2.POPRCTNM
        WHERE T2.receiptdate >= DATEADD(day, -730, GETDATE())
          AND T1.EXTDCOST > 100
        """
        
        df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} receipt lines.")
        
        if df.empty:
            print("No receipts found.")
            return

        # Ensure numeric
        df['ACTLSHIP'] = pd.to_numeric(df['ACTLSHIP'], errors='coerce').fillna(0)
        df['EXTDCOST'] = pd.to_numeric(df['EXTDCOST'], errors='coerce').fillna(0)

        # Sum explicitly
        grouped = df.groupby(['ITEMNMBR', 'ITEMDESC', 'UOFM'])[['ACTLSHIP', 'EXTDCOST']].sum().reset_index()
        
        grouped = grouped.rename(columns={'ACTLSHIP': 'TotalQty', 'EXTDCOST': 'TotalSpend'})
        grouped = grouped.sort_values('TotalSpend', ascending=False).head(50)
        
        # 2. Map to Waste Streams
        print("\n--- WASTE STREAM LEADS ---")
        
        leads = []
        
        for idx, row in grouped.iterrows():
            desc = str(row['ITEMDESC']).upper()
            item = row['ITEMNMBR']
            
            # Find matching keywords
            sources = set()
            for key, industries in waste_map.items():
                if key in desc:
                    for ind in industries:
                        sources.add(ind)
                        
            if sources:
                source_list = ", ".join(sorted(list(sources)))
                leads.append({
                    'Item': item,
                    'Description': row['ITEMDESC'],
                    'Volume': row['TotalQty'],
                    'Spend': row['TotalSpend'],
                    'Potential Sources': source_list
                })
        
        # 3. Report
        if leads:
            leads_df = pd.DataFrame(leads).sort_values('Spend', ascending=False)
            
            print(leads_df[['Description', 'Potential Sources', 'Spend']].head(20).to_string(formatters={
                'Spend': '${:,.0f}'.format
            }))
            leads_df.to_csv('waste_stream_leads.csv', index=False)
        else:
            print("No waste stream mappings found.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    analyze_waste_sourcing_v3()
