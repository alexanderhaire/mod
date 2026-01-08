import streamlit as st
import pandas as pd
import pydeck as pdk
from freight_analytics import (
    load_freight_data, load_all_freight_data, analyze_routes, predict_best_broker, 
    get_coordinates, PLANT_CITY_COORDS, KNOWN_RATES, estimate_tonnage, calculate_per_ton_rate
)

def render_freight_analytics_map():
    st.title("Freight Route Analytics & Map")
    st.markdown("Analyze historical freight data to identify the best brokers per route.")

    # 1. Load Data (Combined ERP + Broker Quotes)
    df = load_all_freight_data()
    
    if df.empty:
        st.warning("No freight history found. Please submit some quotes in the Broker Portal first.")
        return

    # 2. Analyze
    route_stats = analyze_routes(df)
    
    # 3. Visualization (Map)
    st.subheader("Route Map")
    
    # Prepare Map Data
    map_data = []
    
    # We need to fetch coordinates for each origin
    # This might be slow if there are many unique zips. 
    # In a real app, we'd batch geocode or pre-calculate.
    
    unique_origins = route_stats["origin_zip"].unique()
    
    with st.spinner("Geocoding routes..."):
        for origin in unique_origins:
            coords = get_coordinates(origin)
            if coords:
                # Find stats for this origin
                row = route_stats[route_stats["origin_zip"] == origin].iloc[0]
                
                # Color based on data source:
                # Green = ERP (actually paid/verified)
                # Red = Manual quote entry
                # Yellow = Mixed (both sources)
                source = row.get("source", "Unknown")
                if source == "ERP":
                    color = [0, 200, 0]  # Green - verified
                elif source == "Quote":
                    color = [255, 80, 80]  # Red - manual entry
                else:
                    color = [255, 200, 0]  # Yellow/Orange - mixed
                
                map_data.append({
                    "origin_zip": origin,
                    "count": int(row["sample_size"]),
                    "best_broker": row["best_broker"],
                    "avg_price": float(row["avg_market_price"]),
                    "best_price": float(row["best_price"]),
                    "src_lat": coords[0],
                    "src_lon": coords[1],
                    "dst_lat": PLANT_CITY_COORDS[0],
                    "dst_lon": PLANT_CITY_COORDS[1],
                    "color": color,
                    "source": source
                })
    
    if map_data:
        df_map = pd.DataFrame(map_data)

        # PyDeck Arc Layer
        layer = pdk.Layer(
            "ArcLayer",
            data=df_map,
            get_source_position=["src_lon", "src_lat"],
            get_target_position=["dst_lon", "dst_lat"],
            get_source_color="color",
            get_target_color=[255, 255, 0], # Yellow landing
            get_width="1 + (count * 0.5)",
            pickable=True,
            auto_highlight=True,
        )
        
        # Tooltip
        tooltip = {
            "html": "<b>Origin:</b> {origin_zip}<br/>"
                    "<b>Best Broker:</b> {best_broker}<br/>"
                    "<b>Best Price:</b> ${best_price}<br/>"
                    "<b>Avg Price:</b> ${avg_price}<br/>"
                    "<b>Quotes:</b> {count}<br/>"
                    "<b>Source:</b> {source}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        # Legend
        st.markdown("""
        **Legend:** 🟢 ERP (Verified/Paid) | 🔴 Manual Quote | 🟡 Mixed Sources
        """)

        view_state = pdk.ViewState(
            latitude=35.0,
            longitude=-95.0,
            zoom=3,
            pitch=40,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="light", # Streamlit default, no token needed usually
        )
        
        st.pydeck_chart(r)
        
        # --- Time Series for Selected Route ---
        st.markdown("##### Route Price History")
        
        # Route Selector
        route_options = df_map["origin_zip"].tolist()
        if route_options:
            selected_route = st.selectbox(
                "Select a route to view price history:",
                options=route_options,
                format_func=lambda x: f"{x} → Plant City (33563)"
            )
            
            if selected_route:
                # Filter quotes for this route
                route_df = df[df["origin_zip"] == selected_route].copy()
                
                if not route_df.empty and "submitted_at" in route_df.columns:
                    route_df = route_df.sort_values("submitted_at")
                    
                    # Show metrics
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Lowest Rate", f"${route_df['freight_price'].min():.2f}")
                    with col_m2:
                        st.metric("Highest Rate", f"${route_df['freight_price'].max():.2f}")
                    with col_m3:
                        st.metric("Avg Rate", f"${route_df['freight_price'].mean():.2f}")
                    
                    # Time-series line chart
                    st.line_chart(
                        route_df.set_index("submitted_at")[["freight_price"]],
                        use_container_width=True
                    )
                    
                    # Detailed table
                    with st.expander("View All Quotes for This Route"):
                        display_cols = ["submitted_at", "broker_id", "freight_price", "day_of_week"]
                        available_cols = [c for c in display_cols if c in route_df.columns]
                        st.dataframe(route_df[available_cols].sort_values("submitted_at", ascending=False), hide_index=True)
                else:
                    st.info("No time-series data available for this route yet.")
    else:
        st.info("Could not map any routes (Geocoding failed or no data).")

    # 4. Route Predictor
    st.divider()
    
    tab_predict, tab_manual, tab_insights = st.tabs(["🔮 Route Predictor", "📝 Manual Quote Entry", "📈 Time Insights"])
    
    with tab_predict:
        st.subheader("Predict Best Broker")
        col1, col2 = st.columns(2)
        with col1:
            target_zip = st.text_input("Enter Origin Zip Code", placeholder="e.g. 33602")
            if st.button("Find Best Broker"):
                if target_zip:
                    prediction = predict_best_broker(target_zip, df_history=df)
                    
                    if prediction:
                        st.success(f"Best Choice: **{prediction['broker_id']}**")
                        st.metric("Expected Rate", f"${prediction['predicted_price']:.2f}")
                        st.caption(f"Based on {prediction['confidence_samples']} past quotes.")
                        
                        try:
                            # Show alternatives
                            with st.expander("See all candidates"):
                                st.dataframe(prediction["all_candidates"])
                        except Exception:
                            pass
                    else:
                        st.warning("No historical data found for this zip code.")
                else:
                    st.error("Please enter a zip code.")
                    
        with col2:
            st.markdown("#### Top Routes (Lowest Cost)")
            if not route_stats.empty:
                # Add estimated $/ton based on known rates
                display_df = route_stats[["origin_zip", "best_broker", "best_price", "sample_size"]].copy()
                display_df["est_$/ton"] = display_df["origin_zip"].apply(
                    lambda z: KNOWN_RATES.get(z, KNOWN_RATES.get("DEFAULT_REGIONAL", 20.0))
                )
                st.dataframe(
                    display_df.sort_values("best_price"),
                    hide_index=True,
                    column_config={
                        "origin_zip": "Origin",
                        "best_broker": "Best Carrier",
                        "best_price": st.column_config.NumberColumn("Best Price", format="$%.2f"),
                        "sample_size": "# Quotes",
                        "est_$/ton": st.column_config.NumberColumn("Est. $/Ton", format="$%.2f")
                    }
                )

    with tab_manual:
        st.subheader("Log Historical/Manual Quote")
        st.info("Log a quote you received via email or phone to improve the model.")
        
        with st.form("manual_quote_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                # Get unique brokers from history + common ones
                known_brokers = sorted(df["broker_id"].unique().tolist()) if not df.empty else ["Broker A", "Broker B"]
                broker_id = st.text_input("Broker Name", placeholder="e.g. David Cole")
                origin_zip_manual = st.text_input("Origin Zip", placeholder="32920")
            
            with c2:
                freight_price_manual = st.number_input("Total Price ($)", min_value=0.0, step=10.0, value=0.0)
                tonnage_manual = st.number_input("Tonnage (tons)", min_value=0.0, step=1.0, value=0.0, 
                                                  help="If you know the load weight, enter it here")
            
            with c3:
                manual_date = st.date_input("Date Quoted", value=pd.to_datetime("today"))
                equipment_manual = st.selectbox("Equipment", ["Liquid Tanker (Stainless)", "Dry Van", "Reefer", "Flatbed", "Other"])
            
            # Show calculated per-ton rate
            if freight_price_manual > 0 and tonnage_manual > 0:
                per_ton_calc = calculate_per_ton_rate(freight_price_manual, tonnage_manual)
                st.info(f"📊 Calculated Rate: **${per_ton_calc:.2f}/ton**")
            
            submitted = st.form_submit_button("Log Quote")
            
            if submitted:
                if broker_id and origin_zip_manual and freight_price_manual > 0:
                    try:
                        import datetime
                        import hashlib
                        from broker_portal import save_freight_quote
                        
                        # Use noon as default time
                        dt_combined = datetime.datetime.combine(manual_date, datetime.time(12, 0))
                        
                        # Calculate per-ton rate if tonnage provided
                        per_ton = calculate_per_ton_rate(freight_price_manual, tonnage_manual) if tonnage_manual > 0 else None
                        
                        # Summary with tonnage
                        summary = {
                            "origin": f"Manual Entry Zip {origin_zip_manual}",
                            "manual_entry": True,
                            "tonnage": tonnage_manual if tonnage_manual > 0 else None,
                            "per_ton_rate": per_ton
                        }
                        
                        qid = hashlib.md5(f"{broker_id}{dt_combined}".encode()).hexdigest()
                        
                        save_freight_quote(
                            broker_id=broker_id,
                            vendor_quote_id=qid,
                            vendor_quote_summary=summary,
                            freight_price=freight_price_manual,
                            valid_until=None,
                            notes=f"Manual entry - {tonnage_manual:.1f} tons @ ${per_ton:.2f}/ton" if per_ton else "Manual entry",
                            equipment_type=equipment_manual,
                            submitted_at=dt_combined
                        )
                        st.success(f"Quote logged! {f'({tonnage_manual:.1f} tons @ ${per_ton:.2f}/ton)' if per_ton else ''}")
                    except Exception as e:
                        st.error(f"Error saving: {e}")
                else:
                    st.error("Please fill in Broker, Zip, and Price.")

    with tab_insights:
        st.subheader("Price Trends & Timing")
        
        if "submitted_at" in df.columns and "freight_price" in df.columns:
            # 1. Price Over Time Chart
            st.markdown("##### Price History")
            
            # Simple Scatter
            chart_data = df.copy()
            chart_data = chart_data.sort_values("submitted_at")
            
            st.scatter_chart(
                chart_data,
                x="submitted_at",
                y="freight_price",
                color="broker_id",
                size=50,
            )
            
            # 2. Weekday Analysis
            st.divider()
            st.markdown("##### Average Price by Day of Week")
            
            if "day_of_week" in df.columns:
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily_stats = df.groupby("day_of_week")["freight_price"].mean().reindex(day_order).dropna()
                
                st.bar_chart(daily_stats)
                
                # Insight
                best_day = daily_stats.idxmin()
                if isinstance(best_day, str):
                    st.info(f"💡 **Tip:** Historically, **{best_day}** has the lowest average freight rates.")
        else:
            st.warning("Insufficient time data for insights.")


if __name__ == "__main__":
    render_freight_analytics_map()
