
def fetch_vendor_list(cursor):
    """Fetch list of active vendors (ID, Name) from PM00200."""
    try:
        cursor.execute("SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDSTAT = 1 ORDER BY VENDNAME")
        return cursor.fetchall()
    except Exception as e:
        print(f"Error fetching vendors: {e}")
        return []

def render_vendor_portal(cursor):
    # ... existing code ...
    vendors = fetch_vendor_list(cursor)
    vendor_options = [f"{v.VENDNAME} ({v.VENDORID})" for v in vendors]
    selected_vendor = st.selectbox("Select Vendor Identity", vendor_options)
    # ... usage of selected_vendor ...
