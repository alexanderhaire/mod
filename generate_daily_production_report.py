import datetime
import argparse
from db_pool import get_connection
from production_queries import fetch_completed_production, fetch_open_orders_buckets

def generate_report(target_date_str: str = None):
    # Default to "Yesterday" for production, "Today" for Open Orders reference
    today = datetime.date.today()
    if target_date_str:
        try:
            target_date = datetime.datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
            return
    else:
        target_date = today - datetime.timedelta(days=1)

    print(f"Generating Report for Production Date: {target_date} (Run Date: {today})\n")

    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Fetch Production Data
        print("Fetching Production Data...")
        prod_data = fetch_completed_production(cursor, target_date)
        
        # 2. Fetch Open Orders
        print("Fetching Open Orders...")
        orders_data = fetch_open_orders_buckets(cursor, today)

    # --- FORMAT OUTPUT ---
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"DAILY PRODUCTION & INVENTORY REPORT")
    report_lines.append(f"Production Date: {target_date.strftime('%A, %b %d, %Y')}")
    report_lines.append("="*60)
    report_lines.append("")
    
    # PRODUCTION SECTION
    report_lines.append(f"--- PRODUCTION ({target_date}) ---")
    
    # Mixing
    report_lines.append(f"\n[MIXING SHEETS (X)]")
    if prod_data['mixing']:
        for item in prod_data['mixing']:
            report_lines.append(f"  {item['mo_number']:<15} | {item['item_number']:<15} | {item['quantity']:>10.2f} {item['uofm']} | {item['description']}")
    else:
        report_lines.append("  No mixing activity recorded.")

    # Canning
    report_lines.append(f"\n[CANNING SHEETS (C)]")
    if prod_data['canning']:
        for item in prod_data['canning']:
            report_lines.append(f"  {item['mo_number']:<15} | {item['item_number']:<15} | {item['quantity']:>10.2f} {item['uofm']} | {item['description']}")
    else:
        report_lines.append("  No canning activity recorded.")

    report_lines.append("")
    report_lines.append("-" * 60)
    report_lines.append("")

    # ORDER LOGISTICS SECTION
    buckets = orders_data['buckets']
    
    # Helper to print bucket
    def print_bucket(title, items):
        report_lines.append(f"\n{title} ({len(items)} Items)")
        report_lines.append(f"{'Order':<15} | {'Date':<10} | {'Customer':<20} | {'Item':<15} | {'Qty':>8}")
        report_lines.append("-" * 80)
        
        # Group by Order? Or just list? Listing for now.
        # Limit to top 20 lines per bucket to avoid huge dumps in console
        count = 0
        for item in items:
            if count >= 20: 
                report_lines.append(f"  ... and {len(items) - 20} more lines.")
                break
            cust_short = item['customer'][:20]
            report_lines.append(f"{item['order_number']:<15} | {item['req_date']} | {cust_short:<20} | {item['item_number']:<15} | {item['quantity']:>8.2f} {item['uofm']}")
            count += 1

    print_bucket(f"[1] PAST DUE (Before {today})", buckets['past_due'])
    print_bucket(f"[2] DUE TODAY ({today})", buckets['due_today'])
    print_bucket(f"[3] DUE TOMORROW ({today + datetime.timedelta(days=1)})", buckets['due_tomorrow'])
    
    # Future - maybe summarize?
    future_items = buckets['future']
    report_lines.append(f"\n[4] FUTURE ORDERS (Next 7 Days Preview)")
    # Filter future for next 7 days only for the report
    next_week = today + datetime.timedelta(days=7)
    near_future = [i for i in future_items if i['req_date'] <= next_week.strftime('%Y-%m-%d')]
    
    report_lines.append(f"{'Order':<15} | {'Date':<10} | {'Customer':<20} | {'Item':<15} | {'Qty':>8}")
    report_lines.append("-" * 80)
    count = 0
    for item in near_future:
        if count >= 20: 
            report_lines.append(f"  ... and {len(near_future) - 20} more lines.")
            break
        cust_short = item['customer'][:20]
        report_lines.append(f"{item['order_number']:<15} | {item['req_date']} | {cust_short:<20} | {item['item_number']:<15} | {item['quantity']:>8.2f} {item['uofm']}")
        count += 1
        
    final_output = "\n".join(report_lines)
    print(final_output)
    
    # Save to file
    with open("daily_production_report.txt", "w") as f:
        f.write(final_output)
    print("\nReport saved to 'daily_production_report.txt'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target production date YYYY-MM-DD")
    args = parser.parse_args()
    generate_report(args.date)
