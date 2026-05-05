"""
On-Hand Counts – Finished Goods & Raw Materials
=================================================
Pulls every active item classified as a Finished Good or Raw Material from
IV00101 + IV00102 (MAIN location by default) and displays current on-hand
quantities with search, category filters, and CSV export.
"""

import streamlit as st
import pandas as pd
from db_pool import get_connection
from constants import (
    PRIMARY_LOCATION,
    RAW_MATERIAL_CLASS_CODES,
    FINISHED_GOOD_PREFIXES,
    RAW_MATERIAL_PREFIXES,
)

st.set_page_config(layout="wide", page_title="On-Hand Counts")

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* KPI cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    /* Section headers */
    .section-hdr {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.2rem 0 0.4rem 0;
        padding-bottom: 6px;
        border-bottom: 2px solid #3b82f6;
        display: inline-block;
    }
    .fg-hdr { color: #3b82f6; border-color: #3b82f6; }
    .rm-hdr { color: #22c55e; border-color: #22c55e; }
</style>
""", unsafe_allow_html=True)


# ── Data fetching ────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_on_hand(location: str):
    """
    Return a DataFrame with every active item that carries on-hand inventory,
    tagged as Finished Good or Raw Material.

    Classification logic (mirrors existing constants):
      1. ITMCLSCD starts with a known raw-material class code  → Raw Material
      2. ITEMNMBR starts with a finished-good prefix            → Finished Good
      3. ITEMNMBR starts with a raw-material prefix             → Raw Material
      4. Otherwise                                              → Other
    """
    query = """
        SELECT
            RTRIM(m.ITEMNMBR)  AS ItemNumber,
            RTRIM(m.ITEMDESC)  AS Description,
            RTRIM(m.ITMCLSCD)  AS ClassCode,
            RTRIM(m.UOMSCHDL)  AS UoM,
            m.CURRCOST          AS CurrentCost,
            SUM(s.QTYONHND)    AS OnHand
        FROM IV00101 m
        JOIN IV00102 s
          ON m.ITEMNMBR = s.ITEMNMBR
        WHERE s.LOCNCODE = ?
          AND m.ITEMTYPE <> 2           -- exclude Discontinued
        GROUP BY m.ITEMNMBR, m.ITEMDESC, m.ITMCLSCD, m.UOMSCHDL, m.CURRCOST
        HAVING SUM(s.QTYONHND) <> 0
        ORDER BY m.ITEMNMBR
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params=[location])

    if df.empty:
        return df

    # Strip whitespace
    for col in ("ItemNumber", "Description", "ClassCode", "UoM"):
        df[col] = df[col].str.strip()

    # Classify items
    def classify(row):
        cls = (row["ClassCode"] or "").upper()
        itm = (row["ItemNumber"] or "").upper()

        # 1. Class-code match for raw materials
        for rc in RAW_MATERIAL_CLASS_CODES:
            if cls.startswith(rc):
                return "Raw Material"

        # 2. Prefix match – finished goods first (more specific prefixes)
        for fp in FINISHED_GOOD_PREFIXES:
            if itm.startswith(fp):
                return "Finished Good"

        # 3. Prefix match – raw materials
        for rp in RAW_MATERIAL_PREFIXES:
            if itm.startswith(rp):
                return "Raw Material"

        return "Other"

    df["Category"] = df.apply(classify, axis=1)
    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    location = st.text_input("Location Code", value=PRIMARY_LOCATION)
    show_other = st.checkbox("Include items classified as 'Other'", value=False)
    search = st.text_input("🔍 Search item number or description")
    min_qty = st.number_input("Minimum On-Hand Qty", value=0.0, step=1.0)
    if st.button("🔄 Refresh Data"):
        load_on_hand.clear()
        st.rerun()

# ── Load & filter ────────────────────────────────────────────────────────────
with st.spinner("Loading on-hand data from GP…"):
    df_all = load_on_hand(location)

if df_all.empty:
    st.warning("No on-hand data returned. Check your location code or database connection.")
    st.stop()

# Apply filters
df = df_all.copy()
if not show_other:
    df = df[df["Category"] != "Other"]
if search:
    mask = (
        df["ItemNumber"].str.contains(search, case=False, na=False)
        | df["Description"].str.contains(search, case=False, na=False)
    )
    df = df[mask]
if min_qty > 0:
    df = df[df["OnHand"] >= min_qty]

# ── Title & KPIs ─────────────────────────────────────────────────────────────
st.title("📊 On-Hand Inventory Counts")
st.caption(f"Location: **{location}** · {len(df):,} items shown")

fg = df[df["Category"] == "Finished Good"]
rm = df[df["Category"] == "Raw Material"]
ot = df[df["Category"] == "Other"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Finished Goods SKUs", f"{len(fg):,}")
k2.metric("Raw Material SKUs", f"{len(rm):,}")
k3.metric("FG Total On-Hand", f"{fg['OnHand'].sum():,.0f}")
k4.metric("RM Total On-Hand", f"{rm['OnHand'].sum():,.0f}")

# ── Finished Goods table ─────────────────────────────────────────────────────
st.markdown('<div class="section-hdr fg-hdr">🏭 Finished Goods</div>', unsafe_allow_html=True)
if fg.empty:
    st.info("No finished goods match the current filters.")
else:
    st.dataframe(
        fg[["ItemNumber", "Description", "ClassCode", "OnHand", "UoM", "CurrentCost"]].reset_index(drop=True),
        use_container_width=True,
        height=420,
        column_config={
            "ItemNumber": st.column_config.TextColumn("Item #", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "ClassCode": st.column_config.TextColumn("Class", width="small"),
            "OnHand": st.column_config.NumberColumn("On Hand", format="%.2f"),
            "UoM": st.column_config.TextColumn("UoM", width="small"),
            "CurrentCost": st.column_config.NumberColumn("Curr Cost", format="$%.4f"),
        },
    )

# ── Raw Materials table ───────────────────────────────────────────────────────
st.markdown('<div class="section-hdr rm-hdr">🧪 Raw Materials</div>', unsafe_allow_html=True)
if rm.empty:
    st.info("No raw materials match the current filters.")
else:
    st.dataframe(
        rm[["ItemNumber", "Description", "ClassCode", "OnHand", "UoM", "CurrentCost"]].reset_index(drop=True),
        use_container_width=True,
        height=420,
        column_config={
            "ItemNumber": st.column_config.TextColumn("Item #", width="medium"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "ClassCode": st.column_config.TextColumn("Class", width="small"),
            "OnHand": st.column_config.NumberColumn("On Hand", format="%.2f"),
            "UoM": st.column_config.TextColumn("UoM", width="small"),
            "CurrentCost": st.column_config.NumberColumn("Curr Cost", format="$%.4f"),
        },
    )

# ── Other (optional) ─────────────────────────────────────────────────────────
if show_other and not ot.empty:
    with st.expander(f"🔹 Other / Unclassified ({len(ot):,} items)"):
        st.dataframe(
            ot[["ItemNumber", "Description", "ClassCode", "OnHand", "UoM", "CurrentCost"]].reset_index(drop=True),
            use_container_width=True,
            height=350,
            column_config={
                "OnHand": st.column_config.NumberColumn("On Hand", format="%.2f"),
                "CurrentCost": st.column_config.NumberColumn("Curr Cost", format="$%.4f"),
            },
        )

# ── CSV Export ────────────────────────────────────────────────────────────────
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download CSV",
    data=csv,
    file_name=f"on_hand_counts_{location}.csv",
    mime="text/csv",
)
