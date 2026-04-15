import pandas as pd
import pyodbc
from secrets_loader import build_connection_string
import re
import matplotlib.pyplot as plt
import seaborn as sns
from shared_ledger import SharedLedger

def analyze_purity_arbitrage():
    print("--- QUANTITATIVE RESEARCH: Chemical Purity Arbitrage (Ledger Verified) ---")
    
    # Initialize Shared Ledger (Mock Blockchain)
    ledger = SharedLedger()
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Fetch items with relevant descriptions AND Vendor info
        print("Fetching chemical dataset and linking to Vendor Master...")
        query = """
        SELECT 
            i.ITEMNMBR, 
            i.ITEMDESC, 
            i.CURRCOST, 
            i.STNDCOST, 
            i.PRCHSUOM,
            v.VENDORID,
            v.VENDNAME
        FROM IV00101 i
        LEFT JOIN IV00103 iv ON i.ITEMNMBR = iv.ITEMNMBR -- Vendor Item
        LEFT JOIN PM00200 v ON iv.VENDORID = v.VENDORID
        WHERE (i.ITEMDESC LIKE '%[%]%' OR i.ITEMDESC LIKE '%Grade%')
          AND i.ITEMDESC NOT LIKE '%DO NOT USE%'
          AND i.ITEMDESC NOT LIKE '%OBSOLETE%'
          AND i.CURRCOST > 0
        """
        df = pd.read_sql(query, conn)
        print(f"Dataset Size: {len(df)} items")
        
        # 1. PARSE PURITY & CHECK LEDGER
        def get_purity_data(row):
            desc_purity = None
            verified_purity = None
            
            # Parse from description (Legacy/Manual method)
            match = re.search(r'(\d+(?:\.\d+)?)\s*%', str(row['ITEMDESC']))
            if match:
                val = float(match.group(1))
                if 5 <= val <= 100:
                    desc_purity = val
            
            # Check Shared Ledger (Trustless method)
            if row['VENDORID']:
                verified_purity = ledger.get_verified_purity(row['VENDORID'], row['ITEMNMBR'])
                
            return pd.Series([desc_purity, verified_purity])

        df[['ClaimedPurity', 'VerifiedPurity']] = df.apply(get_purity_data, axis=1)
        
        # Drop items where we can't determine ANY purity
        df = df.dropna(subset=['ClaimedPurity', 'VerifiedPurity'], how='all').copy()
        
        # Determine "Effective Purity" for Arbitrage
        # If Verified exists, IT OVERRIDES claims (Source of Truth). 
        df['EffectivePurity'] = df['VerifiedPurity'].combine_first(df['ClaimedPurity'])
        df['IsVerified'] = df['VerifiedPurity'].notna()
        
        # 2. STANDARDIZE PRICE (Cost per 1% Purity)
        df['UnitCost'] = df['CURRCOST']
        df['CostPerPct'] = df['UnitCost'] / df['EffectivePurity']
        
        # 3. IDENTIFY CHEMICAL FAMILIES
        def get_family(desc):
            clean = re.sub(r'\d+.*', '', str(desc)).strip()
            words = clean.split()
            if len(words) >= 2:
                return f"{words[0]} {words[1]}"
            return words[0] if words else "Unknown"
            
        df['Family'] = df['ITEMDESC'].apply(get_family)
        
        # 4. FIND ARBITRAGE
        print("\n--- TRUSTLESS ARBITRAGE OPPORTUNITIES ---")
        
        fam_counts = df['Family'].value_counts()
        major_families = fam_counts[fam_counts > 1].index
        
        arbitrage_found = []
        
        for fam in major_families:
            group = df[df['Family'] == fam].sort_values('EffectivePurity')
            
            min_cost = group['CostPerPct'].min()
            max_cost = group['CostPerPct'].max()
            
            if max_cost > min_cost * 1.2:
                spread = (max_cost - min_cost) / min_cost * 100
                best_buy = group.loc[group['CostPerPct'].idxmin()]
                worst_buy = group.loc[group['CostPerPct'].idxmax()]
                
                # Confidence Score
                confidence = "HIGH (VERIFIED)" if best_buy['IsVerified'] else "LOW (UNVERIFIED)"
                
                arbitrage_found.append({
                    'Family': fam,
                    'Spread': spread,
                    'Confidence': confidence,
                    'BestItem': best_buy['ITEMNMBR'],
                    'BestDesc': best_buy['ITEMDESC'],
                    'BestPurity': best_buy['EffectivePurity'],
                    'BestUnitVal': min_cost,
                    'BestVendor': best_buy['VENDORID'],
                    'WorstDesc': worst_buy['ITEMDESC'],
                    'WorstUnitVal': max_cost
                })
        
        arb_df = pd.DataFrame(arbitrage_found).sort_values(['Confidence', 'Spread'], ascending=[True, False]) # Verified first (alphabetical H < L is wrong, so sort explicit?)
        # Let's sort verified first manually
        arb_df['sort_key'] = arb_df['Confidence'].apply(lambda x: 0 if "HIGH" in x else 1)
        arb_df = arb_df.sort_values(['sort_key', 'Spread'], ascending=[True, False]).drop(columns=['sort_key'])
        
        if not arb_df.empty:
            print(arb_df[['Family', 'Confidence', 'Spread', 'BestDesc', 'BestVendor']].to_string(formatters={
                'Spread': '{:,.1f}%'.format
            }))
            
            arb_df.to_csv('verified_arbitrage.csv', index=False)
            print("\nSaved report to: verified_arbitrage.csv")
            
            # Visualization for Top Verified Opportunity
            verified_ops = arb_df[arb_df['Confidence'].str.contains("HIGH")]
            if not verified_ops.empty:
                top_fam = verified_ops.iloc[0]['Family']
                data = df[df['Family'] == top_fam]
                
                plt.figure(figsize=(10, 6))
                # Plot unverified as circles, verified as stars
                unverified = data[~data['IsVerified']]
                verified = data[data['IsVerified']]
                
                plt.scatter(unverified['EffectivePurity'], unverified['CostPerPct'], 
                          s=100, c='gray', alpha=0.5, label='Unverified (Risky)')
                plt.scatter(verified['EffectivePurity'], verified['CostPerPct'], 
                          s=200, c='gold', marker='*', edgecolors='black', label='Ledger Verified (Trustless)')
                
                plt.title(f"{top_fam}: Validated Arbitrage Curve")
                plt.xlabel("Effective Purity %")
                plt.ylabel("Cost per 1% Active Ingredient")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.savefig(f'verified_arb_{top_fam.replace(" ", "_")}.png')
                print(f"Saved chart: verified_arb_{top_fam.replace(' ', '_')}.png")
                
        else:
            print("No significant arbitrage opportunities found.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()
        
if __name__ == "__main__":
    analyze_purity_arbitrage()
