import streamlit as st
import pandas as pd
import altair as alt
from vega_datasets import data
import datetime
import os, sys
import json
from prophet import Prophet
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text

# ------------------------------------------------------------------------------
# SET PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Weekly Performance Dashboard", layout="wide")
engine = create_engine("postgresql://postgres:1999%40Johannes@localhost:5432/bwb_data")

# Step 1: Inject custom CSS at the top
st.markdown("""
<style>
/* Your entire custom CSS */
body {
    background-color: #f2f2f2;
}
.login-container {
    display: flex; 
    justify-content: center; 
    align-items: center; 
    height: 90vh; 
    flex-direction: column;
}
.login-box {
    background-color: #fff; 
    padding: 2rem 3rem; 
    border-radius: 10px; 
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    max-width: 350px; 
    width: 100%;
}
.login-box h2 {
    text-align: center;
    margin-bottom: 1.5rem;
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-weight: 600;
}
.login-box input {
    width: 100%;
    padding: 0.7rem;
    margin-bottom: 1rem;
    border-radius: 5px;
    border: 1px solid #ccc;
    font-size: 1rem;
}
.login-box button {
    width: 100%;
    padding: 0.7rem;
    font-size: 1rem;
    color: #fff;
    background-color: #007bff;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}
.login-box button:hover {
    background-color: #0069d9;
}
</style>
""", unsafe_allow_html=True)

# Step 2: Manage session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

# Step 3: Credentials
ADMIN_USERNAME = "Administrator"
ADMIN_PASSWORD = "BWB2025"

# Step 4: If not logged in, show login container
if not st.session_state.logged_in:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("<h2>Login</h2>", unsafe_allow_html=True)

    # Fields
    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", placeholder="Enter password", type="password")

    if st.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Please try again.")
    
    st.markdown("</div>", unsafe_allow_html=True)  # close .login-box
    st.markdown("</div>", unsafe_allow_html=True)  # close .login-container
    st.stop()

# Step 5: Main dashboard code after login
st.sidebar.write(f"Welcome, **{st.session_state.username}**!")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

    
# ----------------------------------------------------------------
# 1) Page Config & Minimal CSS for "Boxes"
# ----------------------------------------------------------------

st.markdown("""
<style>
/* Basic resets */
body {
    background-color: #fdf6e3;
    font-family: 'Georgia', serif;
    color: #333;
}


/* Heading styling */
h1, h2, h3, h4 {
    font-family: 'Georgia', serif;
    color: #444;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 0.3em;
}

/* Box styling for each platform */
.platform-box {
    background-color: #f5f5f5;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
}
.platform-box h4 {
    margin-top: 0;
    font-weight: bold;
    color: #333;
}

/* Tabs styling */
[data-testid="stTabs"] > div > button[aria-selected="true"] {
    border-bottom: 3px solid #000;
    color: #000;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

###############################################################################
# 2) Define a Helper Function to Load Data (Cached)
###############################################################################
@st.cache_data(ttl=3600)
def load_table(table_name: str) -> pd.DataFrame:
    df = pd.read_sql(f'SELECT * FROM "{table_name}"', engine)
    if "Week" in df.columns:
        # Attempt to convert using pd.to_datetime.
        # If conversion fails due to the value being a date range (e.g., "2021-05-24/2021-05-30"),
        # then split on '/' and take the first date.
        try:
            df["Week"] = pd.to_datetime(df["Week"])
        except Exception:
            df["Week"] = df["Week"].apply(
                lambda x: pd.to_datetime(x.split("/")[0]) if isinstance(x, str) and "/" in x
                else pd.to_datetime(x, errors="coerce")
            )
    return df

###############################################################################
# 3) Load Precomputed Tables (Using Python-Aggregated Tables)
###############################################################################
# Google Analytics (Python version)
ga_overall = load_table("ga_weekly_aggregation_python")
ga_top = load_table("ga_weekly_top_sources_python")

# Website Sales (Python version)
website_overall = load_table("website_weekly_overall_python")
top10_campaigns = load_table("website_weekly_top_campaign_python")
promotion_analysis = load_table("website_promotion_analysis_python")
prom_combined = load_table("website_promotion_strategy_analysis_python")
promo_top10= load_table("website_promotion_top10_weekly_python")
promo_uplift = load_table("website_promotion_daily_uplift_top10_tidy_python")
pricing_strategy = load_table("website_pricing_strategy_analysis_python")
website_websource = load_table("website_websource_analysis_python")
combined_pp = load_table("website_promotion_strategy_analysis_python")
geo_source_analysis = load_table("website_geo_source_analysis_python")

# Google Ads (Python version)
google_ads_overall = load_table("google_ads_weekly_overall_python")
google_ads_top = load_table("google_ads_weekly_top_campaign_python")

# Facebook Ads (Python version)
fb_overall = load_table("facebook_ads_weekly_campaign_python")
fb_top10 = load_table("facebook_ads_top10_campaigns_python")  # Updated table name


# Bing Ads (Python version)
bing_overall = load_table("bing_ads_weekly_overall_python")
bing_campaign = load_table("bing_ads_weekly_campaign_python")
bing_top = load_table("bing_ads_weekly_top_campaign_python")

# Affiliate Sales (Python version)
affiliate_overall = load_table("affiliate_sales_weekly_overall_python")
affiliate_pub = load_table("affiliate_sales_weekly_publisher_python")
affiliate_top = load_table("affiliate_sales_weekly_top_publisher_python")

# Author Analysis (Python version)
top10_authors_weekly = load_table("website_top10_authors_by_profit_weekly")

###############################################################################
# 4) Sidebar Configuration: Logo, Navigation & Date Range
###############################################################################
st.sidebar.image("C:/Users/machi/Downloads/MSBA Capstone Data/Project/bwb.png", width=200)
st.sidebar.header(f"Welcome, {st.session_state.username}!")
if st.sidebar.button("Logout", key="logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.warning("You have been logged out. Please log in again.")
    st.stop()
st.sidebar.header("Navigation")
pages = ["Overview", "Website", "Author Analysis", "Google Ads", "Google Analytics", "Facebook", "Bing", "Affiliate","Analytics"]
selected_page = st.sidebar.radio("Go to page:", pages)

st.sidebar.header("Date Range")
date_options = {
    "Last 7 Days": 7,
    "Last 28 Days": 28,
    "Last 90 Days": 90,
    "Last 365 Days": 365,
    "Lifetime": None,
    "Custom Range": "custom"
}
selected_range_label = st.sidebar.selectbox("Select Range", list(date_options.keys()))
min_week = website_overall["Week"].min()
max_week = website_overall["Week"].max()
if date_options[selected_range_label] == "custom":
    start_date, end_date = st.sidebar.date_input("Select Custom Range", value=(min_week.date(), max_week.date()), key="global_date")
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
elif date_options[selected_range_label] is None:
    start_ts = min_week
    end_ts = max_week
else:
    days_offset = date_options[selected_range_label]
    end_ts = max_week
    start_ts = max_week - pd.Timedelta(days=days_offset)

def filter_df_by_week(df: pd.DataFrame, start_ts, end_ts) -> pd.DataFrame:
    if "Week" in df.columns:
        return df[(df["Week"] >= start_ts) & (df["Week"] <= end_ts)]
    else:
        return df

# Apply filtering to all tables.
website_overall = filter_df_by_week(website_overall, start_ts, end_ts)
top10_campaigns = filter_df_by_week(top10_campaigns, start_ts, end_ts)
promotion_analysis = filter_df_by_week(promotion_analysis, start_ts, end_ts)
top10_authors_weekly = filter_df_by_week(top10_authors_weekly, start_ts, end_ts)
promo_uplift = filter_df_by_week(promo_uplift, start_ts, end_ts)    
promo_top10 = filter_df_by_week(promo_top10 , start_ts, end_ts)
pricing_strategy = filter_df_by_week(pricing_strategy, start_ts, end_ts)
website_websource = filter_df_by_week(website_websource, start_ts, end_ts)
google_ads_overall = filter_df_by_week(google_ads_overall, start_ts, end_ts)
google_ads_top = filter_df_by_week(google_ads_top, start_ts, end_ts)
combined_pp = filter_df_by_week(combined_pp, start_ts, end_ts)
prom_combined = filter_df_by_week(prom_combined, start_ts, end_ts)
ga_overall = filter_df_by_week(ga_overall, start_ts, end_ts)
ga_top = filter_df_by_week(ga_top, start_ts, end_ts)
fb_overall = filter_df_by_week(fb_overall, start_ts, end_ts)
fb_top10 = filter_df_by_week(fb_top10, start_ts, end_ts)
geo_source_analysis = filter_df_by_week(geo_source_analysis, start_ts, end_ts)


bing_overall = filter_df_by_week(bing_overall, start_ts, end_ts) if not bing_overall.empty else bing_overall
bing_campaign = filter_df_by_week(bing_campaign, start_ts, end_ts) if not bing_campaign.empty else bing_campaign
bing_top = filter_df_by_week(bing_top, start_ts, end_ts) if not bing_top.empty else bing_top
affiliate_overall = filter_df_by_week(affiliate_overall, start_ts, end_ts) if affiliate_overall is not None else affiliate_overall
affiliate_pub = filter_df_by_week(affiliate_pub, start_ts, end_ts) if affiliate_pub is not None else affiliate_pub
affiliate_top = filter_df_by_week(affiliate_top, start_ts, end_ts) if affiliate_top is not None else affiliate_top

# Precompute overall metrics for the Overview page (from website_overall, etc.)
if not website_overall.empty:
    total_revenue = website_overall["Total_Revenue"].sum()
    total_profit = website_overall["Total_Profit"].sum()
    avg_margin = website_overall["Profit_Margin"].mean()
    avg_AOV = website_overall["AOV"].mean()
else:
    total_revenue = total_profit = avg_margin = avg_AOV = 0

if not google_ads_overall.empty:
    total_profit_ads = google_ads_overall["Profit"].sum()
    avg_roi_ads = google_ads_overall["ROI"].mean() * 100
else:
    total_profit_ads = 0
    avg_roi_ads = 0


# 5) Displaying Content Based on the Selected Page

st.markdown('<div class="page-container">', unsafe_allow_html=True)

if selected_page == "Overview":
    st.title("Executive Overview")
    
    st.markdown("""
    This dashboard provides an integrated view of Better World Books’ multi‐platform performance.
    It aggregates key metrics from Website Sales, Affiliate Marketing, Facebook Ads, Google Ads, 
    Bing Ads, and Google Analytics, enabling you to gauge overall health and make informed decisions.
    """)

    # ─────────────────────────────────────────────────────────────
    # Row 1: Website, Facebook, Google Analytics
    # ─────────────────────────────────────────────────────────────
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    # Website Metrics Box
    with row1_col1:
        website_html = f"""
        <p><strong>Revenue:</strong> ${total_revenue:,.2f}</p>
        <p><strong>Profit:</strong> ${total_profit:,.2f}</p>
        <p><strong>Avg Margin:</strong> {avg_margin:.2f}%</p>
        """
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Website</h4>{website_html}</div>', unsafe_allow_html=True)

    # Facebook Metrics Box
    with row1_col2:
        if not fb_overall.empty:
            fb_rev = fb_overall["revenue"].sum()
            fb_cost = fb_overall["cost"].sum()
            fb_profit = fb_rev - fb_cost
            fb_roi = (fb_rev / fb_cost * 100) if fb_cost > 0 else 0
            fb_ctr = (fb_overall["clicks"].sum() / fb_overall["impressions"].sum() * 100) if fb_overall["impressions"].sum() > 0 else 0
            fb_cpc = (fb_cost / fb_overall["clicks"].sum()) if fb_overall["clicks"].sum() > 0 else 0
            fb_html = f"""
            <p><strong>Revenue:</strong> ${fb_rev:,.2f}</p>
            <p><strong>Cost:</strong> ${fb_cost:,.2f}</p>
            <p><strong>Profit:</strong> ${fb_profit:,.2f}</p>
            <p><strong>ROI:</strong> {fb_roi:.2f}%</p>
            <p><strong>CTR:</strong> {fb_ctr:.2f}%</p>
            <p><strong>CPC:</strong> ${fb_cpc:.2f}</p>
            <p><strong>Impressions:</strong> {fb_overall["impressions"].sum():,}</p>
            <p><strong>Clicks:</strong> {fb_overall["clicks"].sum():,}</p>
            """
        else:
            fb_html = "<p>No Data Available</p>"
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Facebook Ads</h4>{fb_html}</div>', unsafe_allow_html=True)

    # Google Analytics Metrics Box
    with row1_col3:
        if not ga_overall.empty:
            ga_sess = ga_overall["Total_Sessions"].sum()
            ga_rev = ga_overall["Total_Purchase_Revenue"].sum()
            ga_bounce = ga_overall["Avg_Bounce_Rate"].mean()
            ga_html = f"""
            <p><strong>Sessions:</strong> {ga_sess:,}</p>
            <p><strong>Revenue:</strong> ${ga_rev:,.2f}</p>
            <p><strong>Bounce Rate:</strong> {ga_bounce:.2f}%</p>
            """
        else:
            ga_html = "<p>No Data Available</p>"
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Google Analytics</h4>{ga_html}</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────
    # Row 2: Google Ads, Bing Ads, Affiliate
    # ─────────────────────────────────────────────────────────────
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    # Google Ads Metrics Box
    with row2_col1:
        if not google_ads_overall.empty:
            ga_ads_cost = google_ads_overall["cost"].sum()
            ga_ads_rev = google_ads_overall["conversions_value"].sum()
            ga_ads_profit = ga_ads_rev - ga_ads_cost
            ga_ads_roi = (ga_ads_profit / ga_ads_cost * 100) if ga_ads_cost > 0 else 0
            ga_ads_ctr = (google_ads_overall["clicks"].sum() / google_ads_overall["impressions"].sum() * 100) if google_ads_overall["impressions"].sum() > 0 else 0
            ga_ads_cpc = (ga_ads_cost / google_ads_overall["clicks"].sum()) if google_ads_overall["clicks"].sum() > 0 else 0
            ads_html = f"""
            <p><strong>Profit:</strong> ${ga_ads_profit:,.2f}</p>
            <p><strong>ROI:</strong> {ga_ads_roi:.2f}%</p>
            <p><strong>CTR:</strong> {ga_ads_ctr:.2f}%</p>
            <p><strong>CPC:</strong> ${ga_ads_cpc:.2f}</p>
            <p><strong>Cost:</strong> ${ga_ads_cost:,.2f}</p>
            <p><strong>Revenue:</strong> ${ga_ads_rev:,.2f}</p>
            """
        else:
            ads_html = "<p>No Data Available</p>"
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Google Ads</h4>{ads_html}</div>', unsafe_allow_html=True)

    # Bing Ads Metrics Box
    with row2_col2:
        if bing_overall is not None and not bing_overall.empty:
            bing_rev = bing_overall["conversions_value"].sum()
            bing_cost = bing_overall["cost"].sum()
            bing_profit = bing_rev - bing_cost
            bing_roi = (bing_rev / bing_cost * 100) if bing_cost > 0 else 0
            bing_ctr = (bing_overall["clicks"].sum() / bing_overall["impressions"].sum() * 100) if bing_overall["impressions"].sum() > 0 else 0
            bing_cpc = (bing_cost / bing_overall["clicks"].sum()) if bing_overall["clicks"].sum() > 0 else 0
            bing_html = f"""
            <p><strong>Revenue:</strong> ${bing_rev:,.2f}</p>
            <p><strong>Profit:</strong> ${bing_profit:,.2f}</p>
            <p><strong>ROI:</strong> {bing_roi:.2f}%</p>
            <p><strong>CTR:</strong> {bing_ctr:.2f}%</p>
            <p><strong>CPC:</strong> ${bing_cpc:.2f}</p>
            <p><strong>Cost:</strong> ${bing_cost:,.2f}</p>
            <p><strong>Conversions:</strong> {bing_overall["conversions"].sum():,}</p>
            <p><strong>Impressions:</strong> {bing_overall["impressions"].sum():,}</p>
            <p><strong>Clicks:</strong> {bing_overall["clicks"].sum():,}</p>
            """
        else:
            bing_html = "<p>No Data Available</p>"
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Bing Ads</h4>{bing_html}</div>', unsafe_allow_html=True)

    # Affiliate Metrics Box
    with row2_col3:
        if affiliate_overall is not None and not affiliate_overall.empty:
            aff_sales = affiliate_overall["Total_Sales"].sum()
            aff_comm = affiliate_overall["Total_Commission"].sum()
            aff_net = affiliate_overall["Net_After_Commission"].sum()
            aff_orders = affiliate_overall["Order_Count"].sum()
            aff_html = f"""
            <p><strong>Total Sales:</strong> ${aff_sales:,.2f}</p>
            <p><strong>Net Revenue:</strong> ${aff_net:,.2f}</p>
            <p><strong>Orders:</strong> {aff_orders:,}</p>
            <p><strong>Commission:</strong> ${aff_comm:,.2f}</p>
            <p><strong>Avg Commission:</strong> ${aff_comm / aff_orders:,.2f}</p>
            <p><strong>Avg Sales:</strong> ${aff_sales / aff_orders:,.2f}</p>
            <p><strong>Avg Net Revenue:</strong> ${aff_net / aff_orders:,.2f}</p>
            <p><strong>Avg Order Value:</strong> ${aff_sales / aff_orders:,.2f}</p>
            """
        else:
            aff_html = "<p>No Data Available</p>"
        st.markdown(f'<div class="platform-box" style="border:1px solid #ccc;padding:10px;margin:5px;border-radius:5px;"><h4>Affiliate</h4>{aff_html}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    

    
    st.markdown("---")
    # --- Comparison Chart: Profit from Bing, Facebook, Google Ads ---
    st.subheader("Profit Comparison: Bing vs. Facebook vs. Google Ads")
    # Create a DataFrame from computed values:
    profit_data = pd.DataFrame({
        "Platform": ["Bing Ads", "Facebook Ads", "Google Ads"],
        "Profit": [
            bing_profit if bing_overall is not None and not bing_overall.empty else 0,
            fb_profit if not fb_overall.empty else 0,
            ga_ads_profit if not google_ads_overall.empty else 0
        ]
    })
    profit_chart = alt.Chart(profit_data).mark_bar().encode(
        x=alt.X("Platform:N", title="Platform"),
        y=alt.Y("Profit:Q", title="Profit (USD)", scale=alt.Scale(zero=False)),
        color=alt.Color("Platform:N", legend=None),
        tooltip=["Platform", "Profit"]
    ).properties(width=500, height=400)
    st.altair_chart(profit_chart, use_container_width=True)
    
    st.markdown("---")
    # --- Affiliate: Top Affiliate Websites by % Contribution ---
    st.subheader("Top Affiliate Websites by % Contribution to Revenue")
    if affiliate_pub is not None and not affiliate_pub.empty:
        # Calculate percentage dynamically
        sorted_aff = affiliate_pub.sort_values("Total_Sales", ascending=False).reset_index(drop=True)
        top_affiliates = sorted_aff.head(10)
        total_aff_sales = sorted_aff["Total_Sales"].sum()
        top_affiliates["Percent_of_Total"] = (top_affiliates["Total_Sales"] / total_aff_sales) * 100
    
        bar_chart = alt.Chart(top_affiliates).mark_bar().encode(
            x=alt.X("publishername:N", sort='-y', title="Affiliate Website"),
            y=alt.Y("Total_Sales:Q", title="Total Sales (USD)"),
            tooltip=[
                alt.Tooltip("publishername:N", title="Affiliate"),
                alt.Tooltip("Total_Sales:Q", title="Total Sales", format=",.2f"),
                alt.Tooltip("Percent_of_Total:Q", title="Contribution (%)", format=".2f")
            ],
            color=alt.Color("publishername:N", legend=alt.Legend(title="Affiliate"))
            ).properties(width=500, height=400)  
        
        st.altair_chart(bar_chart, use_container_width=True)
        
    
    else:
        st.info("No affiliate publisher data available.")
    
    st.markdown("---")
    # --- Top 10 Website Campaigns ---
    st.subheader("Top 10 Website Campaigns")
    grouped_campaigns = top10_campaigns.groupby("campaign", as_index=False).agg({
    "Total_Revenue": "sum",
    "Total_Profit": "sum",
    # Add any other columns you want, e.g. "Order_Count": "sum", ...
    })

# 3) Sort and pick top 10 by revenue (or by profit if you prefer)
    top10 = grouped_campaigns.sort_values("Total_Revenue", ascending=False).head(10)

# 4) Build your Altair chart from `top10`:
    chart = (
    alt.Chart(top10)
    .mark_bar()
    .encode(
        x=alt.X("campaign:N", sort="-y", title="Campaign"),
        y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
        color=alt.Color("campaign:N", legend=None),
        tooltip=["campaign", "Total_Revenue", "Total_Profit"]
    )
    .properties(width=600, height=400)
    )

    st.altair_chart(chart, use_container_width=True)

    
    # ----------------------
    # 2) Websource Analysis
    # ----------------------
elif selected_page == "Website":
    st.title("Detailed Website Sales Performance")
    website_tabs = st.tabs([
        "Overview",
        "Websource Analysis",
        "Promotion Analysis",
        "Pricing Strategy Analysis",
        "Combined Pricing & Promotion Analysis",
        "Overall Top Campaigns"
    ])
    # ----------------------
    # 1) Overview Tab
    # ----------------------
    with website_tabs[0]:
        st.subheader("Key Metrics & Trends")
        website_sorted = website_overall.sort_values("Week")
        if not website_sorted.empty:
            website_sorted["Prev_Revenue"] = website_sorted["Total_Revenue"].shift(1)
            website_sorted["Revenue_Change"] = website_sorted["Total_Revenue"] - website_sorted["Prev_Revenue"]
            latest_week_data = website_sorted.iloc[-1]
            if pd.notnull(latest_week_data["Prev_Revenue"]) and latest_week_data["Prev_Revenue"] != 0:
                wow_diff = latest_week_data["Revenue_Change"]
                wow_pct = (wow_diff / latest_week_data["Prev_Revenue"] * 100)
            else:
                wow_diff = 0
                wow_pct = 0
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("This Week's Revenue", f"${latest_week_data['Total_Revenue']:,.2f}",
                          f"{wow_diff:+.2f} ({wow_pct:+.1f}%) WoW")
            with colB:
                st.metric("Total Profit (Range)", f"${total_profit:,.2f}")
            with colC:
                st.metric("Avg Profit Margin", f"{avg_margin:.2f}%")
        st.markdown("---")
        st.subheader("Revenue & Profit Trends")
        if not website_overall.empty:
            line_revenue = alt.Chart(website_overall).mark_line(point=True, color="#1f77b4").encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                tooltip=["Week", "Total_Revenue"]
            )
            line_profit = alt.Chart(website_overall).mark_line(point=True, color="#2ca02c").encode(
                x=alt.X("Week:T"),
                y=alt.Y("Total_Profit:Q", title="Profit (USD)", scale=alt.Scale(zero=False)),
                tooltip=["Week", "Total_Profit"]
            )
            layered_web = alt.layer(line_revenue, line_profit).resolve_scale(y="independent").properties(
                width=700, height=400, title="Revenue (Blue) & Profit (Green) Trends"
            )
            st.altair_chart(layered_web, use_container_width=True)
        st.markdown("---")
        st.subheader("Interactive Bubble Chart: Orders vs Revenue")
        if not website_overall.empty:
            bubble_chart = alt.Chart(website_overall).mark_circle().encode(
                x=alt.X("Order_Count:Q", title="Order Count"),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                size=alt.Size("AOV:Q", title="Average Order Value", scale=alt.Scale(range=[50, 500])),
                color=alt.Color("Profit_Margin:Q", title="Profit Margin (%)", scale=alt.Scale(scheme="blues")),
                tooltip=["Week", "Order_Count", "Total_Revenue", "AOV", "Profit_Margin"]
            ).properties(width=700, height=400, title="Orders vs Revenue (Bubble size = AOV)")
            st.altair_chart(bubble_chart, use_container_width=True)
    # ----------------------
    # 2) Websource Analysis
    # ----------------------
    
    # ----------------------
# 2) Websource Analysis
# ----------------------
    with website_tabs[1]:
        st.subheader("Weekly Websource Revenue & Profit")
        
        if not website_websource.empty:
            # Aggregate revenue by each web source over the selected date range.
            revenue_by_source = (
                website_websource.groupby("cleaned_websource", as_index=False)
                .agg({"Total_Revenue": "sum"})
            )
            # Sort by Total_Revenue in descending order and take the top 10 web sources.
            top_sources = revenue_by_source.sort_values("Total_Revenue", ascending=False).head(10)
            top_sources_list = top_sources["cleaned_websource"].tolist()
            
            # Filter the original DataFrame to include only rows with the top 10 web sources.
            top_websource_df = website_websource[
                website_websource["cleaned_websource"].isin(top_sources_list)
            ].copy()
            
            # Bar chart for revenue by web source (Top 10)
            bar_chart = alt.Chart(top_websource_df).mark_bar().encode(
                x=alt.X(
                    "cleaned_websource:N",
                    title="Web Source",
                    sort=alt.SortField(field="Total_Revenue", order="descending")
                ),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                tooltip=[
                    "Week",
                    alt.Tooltip("cleaned_websource:N", title="Web Source"),
                    alt.Tooltip("Total_Revenue:Q", title="Revenue (USD)", format=",.2f"),
                    alt.Tooltip("Total_Total:Q", title="Total Profit", format=",.2f"),
                    alt.Tooltip("Profit_Margin:Q", title="Profit Margin (%)", format=".2f")
                ]
            ).properties(width=700, height=400, title="Revenue by Web Source (Top 10)")
            st.altair_chart(bar_chart, use_container_width=True)
            
            # Line chart for weekly revenue trend for the top 10 web sources.
            line_chart = alt.Chart(top_websource_df).mark_line(point=True).encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                color=alt.Color("cleaned_websource:N", title="Web Source"),
                tooltip=[
                    "Week",
                    alt.Tooltip("cleaned_websource:N", title="Web Source"),
                    alt.Tooltip("Total_Revenue:Q", title="Revenue (USD)", format=",.2f"),
                    alt.Tooltip("Total_Total:Q", title="Total Profit", format=",.2f"),
                    alt.Tooltip("Profit_Margin:Q", title="Profit Margin (%)", format=".2f")
                ]
            ).properties(width=700, height=400, title="Weekly Revenue Trend by Top 10 Web Sources")
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.info("No websource data available.")

    
    # ----------------------
    # 3) Promotion Analysis
    # ----------------------
    # ----------------------
# 3) Promotion Analysis
# ----------------------
    with website_tabs[2]:
        st.subheader("Weekly Promotion Performance")

        # Display a raw table preview if desired (optional)
        st.markdown("### Raw Promotion Data (Preview)")
        if not promo_top10.empty:
            st.dataframe(promo_top10.head(10))
        else:
            st.info("No promotion data available for the selected date range.")

        # --- Global Aggregation for the Entire Selected Period ---
        # Group the existing weekly promotion aggregates (from promo_top10) by promotion (using "promo_clean")
        global_promo = promo_top10.groupby("promo_clean", as_index=False).agg({
            "total_profit": "sum",
            "total_revenue": "sum",
            "total_orders": "sum"
        })

        # Sort the aggregated data by total_profit (descending) and take the top 10 promotions.
        global_top10 = global_promo.sort_values("total_profit", ascending=False).head(10)

        st.markdown("### Top 10 Promotions for the Selected Period")
        st.dataframe(global_top10)

        # --- Bar Chart for Global Top 10 Promotions by Total Profit ---
        profit_chart = alt.Chart(global_top10).mark_bar().encode(
            x=alt.X("promo_clean:N", 
                    title="Promotion", 
                    sort=alt.SortField(field="total_profit", order="descending")),
            y=alt.Y("total_profit:Q", title="Total Profit (USD)", scale=alt.Scale(zero=True)),
            tooltip=[
                alt.Tooltip("promo_clean:N", title="Promotion"),
                alt.Tooltip("total_profit:Q", title="Total Profit", format=",.2f"),
                alt.Tooltip("total_revenue:Q", title="Total Revenue", format=",.2f"),
                alt.Tooltip("total_orders:Q", title="Orders", format=",.0f")
            ],
            color=alt.Color("promo_clean:N", legend=None)
        ).properties(width=700, height=400, title="Global Top 10 Promotions by Total Profit")
        st.altair_chart(profit_chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Promotion Uplift Metrics (Daily Averages)")

        if promo_uplift.empty:
            st.info("No promotion uplift data available for the selected date range.")
        else:
            # Optionally show a preview of the uplift table
            st.dataframe(promo_uplift.head(10))

            # --- Global Aggregation for Uplift Metrics ---
            # Aggregate average uplift metrics per promotion across the period.
            uplift_agg = promo_uplift.groupby("promo_clean", as_index=False).agg({
                "revenue_lift": "mean",
                "profit_lift": "mean",
                "orders_lift": "mean"
            })

            # Sort by revenue_lift (descending) and take the top 10 promotions
            top10_uplift = uplift_agg.sort_values("revenue_lift", ascending=False).head(10)

            st.markdown("### Top 10 Promotions by Average Revenue Lift (Global)")
            st.dataframe(top10_uplift)

            # --- Create Bar Charts for Each Lift Metric ---
            revenue_lift_chart = alt.Chart(top10_uplift).mark_bar().encode(
                x=alt.X("promo_clean:N", title="Promotion", sort=alt.SortField(field="revenue_lift", order="descending")),
                y=alt.Y("revenue_lift:Q", title="Average Revenue Lift (%)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("promo_clean:N", title="Promotion"),
                    alt.Tooltip("revenue_lift:Q", title="Avg Revenue Lift (%)", format=".2f")
                ]
            ).properties(width=700, height=400, title="Revenue Lift (%) by Promotion (Top 10)")

            profit_lift_chart = alt.Chart(top10_uplift).mark_bar().encode(
                x=alt.X("promo_clean:N", title="Promotion", sort=alt.SortField(field="profit_lift", order="descending")),
                y=alt.Y("profit_lift:Q", title="Average Profit Lift (%)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("promo_clean:N", title="Promotion"),
                    alt.Tooltip("profit_lift:Q", title="Avg Profit Lift (%)", format=".2f")
                ]
            ).properties(width=700, height=400, title="Profit Lift (%) by Promotion (Top 10)")

            orders_lift_chart = alt.Chart(top10_uplift).mark_bar().encode(
                x=alt.X("promo_clean:N", title="Promotion", sort=alt.SortField(field="orders_lift", order="descending")),
                y=alt.Y("orders_lift:Q", title="Average Orders Lift (%)", scale=alt.Scale(zero=True)),
                tooltip=[
                    alt.Tooltip("promo_clean:N", title="Promotion"),
                    alt.Tooltip("orders_lift:Q", title="Avg Orders Lift (%)", format=".2f")
                ]
            ).properties(width=700, height=400, title="Orders Lift (%) by Promotion (Top 10)")

            st.altair_chart(revenue_lift_chart, use_container_width=True)
            st.altair_chart(profit_lift_chart, use_container_width=True)
            st.altair_chart(orders_lift_chart, use_container_width=True)
    # ----------------------
    # 4) Pricing Strategy Analysis
    # ----------------------
    with website_tabs[3]:
        st.subheader("Weekly Pricing Strategy Performance")

        if pricing_strategy.empty:
            st.info("No pricing strategy data available for the selected date range.")
        else:
            # 1) Optionally display the raw data after filtering.
            st.markdown("#### Data Preview (Filtered)")
            st.dataframe(pricing_strategy.head(10))

            # 2) Aggregate only on the filtered DataFrame:
            agg_pricing = pricing_strategy.groupby("strategy_base", as_index=False).agg({
                "Total_Revenue": "sum",
                "Total_Profit": "sum",
                "Order_Count": "sum"
            })

            # 3) Sort and pick top 10 by total revenue (or by profit, whichever you prefer).
            top10_strats = agg_pricing.sort_values("Total_Revenue", ascending=False).head(10)["strategy_base"].tolist()

            # 4) Filter out only rows from those top 10 strategies in the already-filtered data.
            filtered_pricing = pricing_strategy[pricing_strategy["strategy_base"].isin(top10_strats)]

            # --- Bar Chart ---
            bar_chart = alt.Chart(filtered_pricing).mark_bar().encode(
                x=alt.X(
                    "strategy_base:N",
                    title="Pricing Strategy",
                    sort=alt.SortField(field="Total_Revenue", order="descending")
                ),
                y=alt.Y("Total_Revenue:Q", title="Total Revenue (USD)", scale=alt.Scale(zero=True)),
                color=alt.Color("is_test:N", title="Test Mode", scale=alt.Scale(scheme="category10")),
                tooltip=[
                    alt.Tooltip("Week:T", title="Week"),
                    alt.Tooltip("strategy_base:N", title="Strategy"),
                    alt.Tooltip("Total_Revenue:Q", title="Total Revenue", format=",.2f"),
                    alt.Tooltip("Total_Profit:Q", title="Total Profit", format=",.2f"),
                    alt.Tooltip("Order_Count:Q", title="Orders")
                ]
            ).properties(width=700, height=400, title="Weekly Revenue by Pricing Strategy (Top 10)")
            st.altair_chart(bar_chart, use_container_width=True)

            st.markdown("---")

            # --- Line Chart ---
            line_chart = alt.Chart(filtered_pricing).mark_line(point=True).encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                color=alt.Color("strategy_base:N", title="Pricing Strategy"),
                tooltip=[
                    alt.Tooltip("Week:T", title="Week"),
                    alt.Tooltip("strategy_base:N", title="Strategy"),
                    alt.Tooltip("Total_Revenue:Q", title="Total Revenue", format=",.2f"),
                    alt.Tooltip("Total_Profit:Q", title="Total Profit", format=",.2f"),
                    alt.Tooltip("Order_Count:Q", title="Orders")
                ]
            ).properties(width=700, height=400, title="Weekly Revenue Trend by Pricing Strategy (Top 10)")
            st.altair_chart(line_chart, use_container_width=True)
        

    # ----------------------
    # 5) Combined Pricing & Promotion Analysis
    # ----------------------
    # --- (H) Combined Pricing & Promotion Analysis using website_promotion_strategy_analysis_python ---
    with website_tabs[4]:
        st.subheader("Combined Pricing & Promotion Analysis")
        
        # Check if the combined table is not empty
        if not prom_combined.empty:
            # Sort the data by Total_Profit and Profit_Margin (both descending)
            combined_df = prom_combined.sort_values(
                by=["Total_Profit", "Profit_Margin"],
                ascending=[False, False]
            )
            
            # Get the best performing combination (top row)
            best_combo = combined_df.iloc[0]
            st.markdown(
                f"**Recommendation:** The best performing combination is: Pricing Strategy **{best_combo['pricingstrategy']}**, "
                f"Promotion **{best_combo['promo_clean']}**, Campaign **{best_combo['campaign']}** (Category: **{best_combo['category']}**) "
                f"with Total Profit **${best_combo['Total_Profit']:,.2f}** and Profit Margin **{best_combo['Profit_Margin']:.2f}%**."
            )
            
            # Get the top 5 combinations for visualization
            top5 = combined_df.head(5)
            
            # Create a scatter chart to display Total Profit versus Profit Margin
            scatter_chart = alt.Chart(top5).mark_circle(size=100).encode(
                x=alt.X("Total_Profit:Q", title="Total Profit (USD)"),
                y=alt.Y("Profit_Margin:Q", title="Profit Margin (%)"),
                color=alt.Color("pricingstrategy:N", title="Pricing Strategy"),
                shape=alt.Shape("promo_clean:N", title="Promotion"),
                tooltip=[
                    "pricingstrategy",
                    "promo_clean",
                    "campaign",
                    "category",
                    alt.Tooltip("Total_Revenue:Q", title="Total Revenue", format=",.2f"),
                    alt.Tooltip("Total_Profit:Q", title="Total Profit", format=",.2f"),
                    alt.Tooltip("Order_Count:Q", title="Orders", format=",.0f"),
                    alt.Tooltip("Profit_Margin:Q", title="Profit Margin (%)", format=".2f")
                ]
            ).properties(
                width=700, height=400, title="Top 5 Combined Strategies: Profit vs Profit Margin"
            )
            
            st.altair_chart(scatter_chart, use_container_width=True)
        else:
            st.info("No combined pricing & promotion data available.")


    # ----------------------
    # 6) Overall Top Campaigns
    # ----------------------
    with website_tabs[5]:
        st.subheader("Overall Top Campaigns")
        if not top10_campaigns .empty:
            overall_campaign = top10_campaigns.groupby("campaign").agg({
                "Total_Profit": "sum",
                "Total_Revenue": "sum",
                "Order_Count": "sum",
                "Profit_Margin": "mean"
            }).reset_index()
            overall_campaign = overall_campaign.sort_values("Total_Profit", ascending=False).head(10)
            metric_options = {
                "Profit (USD)": "Total_Profit",
                "Revenue (USD)": "Total_Revenue",
                "Orders": "Order_Count"
            }
            selected_metric_label = st.selectbox("Select Metric to Display:", list(metric_options.keys()), index=0)
            selected_metric_col = metric_options[selected_metric_label]
            top_campaigns_chart = alt.Chart(overall_campaign).mark_bar().encode(
                x=alt.X("campaign:N", title="Campaign", sort=alt.SortField(field="Total_Profit", order="descending")),
                y=alt.Y(f"{selected_metric_col}:Q", title=selected_metric_label),
                tooltip=["campaign", "Total_Profit", "Total_Revenue", "Order_Count", "Profit_Margin"]
            ).properties(width=700, height=400, title=f"Top 10 Campaigns by {selected_metric_label}")
            st.altair_chart(top_campaigns_chart, use_container_width=True)
        else:
            st.info("No campaign data available.")

elif selected_page == "Author Analysis":
    st.title("Author Performance Report")
    
    # Display a bar chart of top 10 authors by Total Revenue
    if not top10_authors_weekly.empty:
        top_authors_overall = top10_authors_weekly.groupby("author").agg(
            Total_Revenue=("Total_Revenue", "sum")
        ).reset_index().sort_values("Total_Revenue", ascending=False).head(10)
    
        st.subheader("Top 10 Authors by Total Revenue")
        bar_chart = alt.Chart(top_authors_overall).mark_bar().encode(
            x=alt.X("author:N", title="Author", sort="-y"),
            y=alt.Y("Total_Revenue:Q", title="Total Revenue (USD)"),
            tooltip=["author", "Total_Revenue"]
        ).properties(width=700, height=400)
    
        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("No author data available for graphing.")
    
    # Display a table of top authors
    st.subheader("Search for an Author")
    
    if not top10_authors_weekly.empty:
        available_authors = sorted(top10_authors_weekly["author"].unique())
        selected_author = st.selectbox("Select an Author", available_authors)
        filtered_author = top10_authors_weekly[top10_authors_weekly["author"] == selected_author]
        if not filtered_author.empty:
            st.markdown(f"### Report for Author: {selected_author}")
            total_books = filtered_author["Total_Books"].sum()
            author_revenue = filtered_author["Total_Revenue"].sum()
            author_profit = filtered_author["Total_Profit"].sum()
            st.metric("Total Books Sold", total_books)
            st.metric("Total Revenue", f"${author_revenue:,.2f}")
            st.metric("Total Profit", f"${author_profit:,.2f}")
            author_trend = alt.Chart(filtered_author).mark_line(point=True).encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Total_Revenue:Q", title="Revenue (USD)"),
                tooltip=["Week", "Total_Revenue", "Total_Profit", "Total_Books"]
            ).properties(width=700, height=400, title="Weekly Revenue Trend")
            st.altair_chart(author_trend, use_container_width=True)
    


elif selected_page == "Google Ads":
    st.title("Weekly Google Ads Performance")
    if not google_ads_overall.empty:
        google_sorted = google_ads_overall.sort_values("Week")
        google_sorted["Prev_Profit"] = google_sorted["Profit"].shift(1)
        google_sorted["Profit_Change"] = google_sorted["Profit"] - google_sorted["Prev_Profit"]
        if not google_sorted.empty:
            latest_google = google_sorted.iloc[-1]
            profit_diff = latest_google["Profit_Change"]
            profit_pct = (profit_diff / latest_google["Prev_Profit"] * 100) if latest_google["Prev_Profit"] else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("This Week's Profit", f"${latest_google['Profit']:,.2f}",
                          f"{profit_diff:+.2f} ({profit_pct:+.1f}%) WoW")
            with col2:
                st.metric("Total Profit (Range)", f"${total_profit_ads:,.2f}")
            with col3:
                st.metric("Avg ROI", f"{avg_roi_ads:.2f}%")
        st.markdown("---")
        st.subheader("Profit Trends")
        line_profit_ads = alt.Chart(google_ads_overall).mark_line(point=True, color="red").encode(
            x=alt.X("Week:T", title="Week"),
            y=alt.Y("Profit:Q", title="Profit (USD)", scale=alt.Scale(zero=False)),
            tooltip=["Week", "Profit", "ROI"]
        ).properties(width=700, height=400)
        st.altair_chart(line_profit_ads, use_container_width=True)
        st.markdown("---")
        st.subheader("Top Weekly Google Ads Campaigns")
        if not google_ads_top.empty:
            ads_campaign_chart = alt.Chart(google_ads_top).mark_bar().encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("Profit:Q", title="Profit (USD)"),
                color=alt.Color("campaign_name:N", legend=alt.Legend(title="Campaign")),
                tooltip=["Week", "campaign_name", "Profit", "ROI"]
            ).properties(width=700, height=400).interactive()
            st.altair_chart(ads_campaign_chart, use_container_width=True)
        else:
            st.info("No top-campaign data available for the selected date range for Google Ads.")
    else:
        st.info("No Google Ads data available for the selected range.")

elif selected_page == "Google Analytics":
    st.title("Google Analytics Detailed View")
    if not ga_overall.empty:
        ga_total_sessions = ga_overall["Total_Sessions"].sum()
        ga_total_revenue = ga_overall["Total_Purchase_Revenue"].sum()
        ga_avg_bounce = ga_overall["Avg_Bounce_Rate"].mean()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sessions", f"{ga_total_sessions:,.0f}")
        with col2:
            st.metric("Total Purchase Revenue", f"${ga_total_revenue:,.2f}")
        with col3:
            st.metric("Avg Bounce Rate", f"{ga_avg_bounce:.2f}%")
        st.markdown("---")
        st.markdown("### Weekly GA Revenue Trend")
        ga_revenue_line = alt.Chart(ga_overall).mark_line(point=True, color="orange").encode(
            x=alt.X("Week:T", title="Week"),
            y=alt.Y("Total_Purchase_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
            tooltip=["Week", "Total_Purchase_Revenue", "Total_Sessions"]
        ).properties(width=700, height=400)
        st.altair_chart(ga_revenue_line, use_container_width=True)
        st.markdown("---")
        
        # --- Top GA Sources by Revenue (Aggregated over the Selected Period) ---
        st.markdown("### Top GA Sources by Revenue")
        if not ga_top.empty:
            # Group by 'source' and sum over Total_Purchase_Revenue and Total_Sessions.
            ga_source_agg = ga_top.groupby("source", as_index=False).agg({
                "Total_Purchase_Revenue": "sum",
                "Total_Sessions": "sum"
            })
            # Sort the aggregated data in descending order by revenue and take the top 10.
            top10_ga_sources = ga_source_agg.sort_values("Total_Purchase_Revenue", ascending=False).head(10)
            
            ga_top_bar = alt.Chart(top10_ga_sources).mark_bar().encode(
                x=alt.X("source:N", title="Source", sort=alt.SortField(field="Total_Purchase_Revenue", order="descending")),
                y=alt.Y("Total_Purchase_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                tooltip=["source", alt.Tooltip("Total_Purchase_Revenue:Q", title="Revenue (USD)", format=",.2f"),
                         alt.Tooltip("Total_Sessions:Q", title="Sessions", format=",.0f")]
            ).properties(width=700, height=400)
            st.altair_chart(ga_top_bar, use_container_width=True)
        else:
            st.info("No top GA sources data available.")
        
        st.markdown("---")
        st.markdown("### GA Sessions vs Revenue Scatter Plot")
        ga_scatter = alt.Chart(ga_overall).mark_circle(size=100).encode(
            x=alt.X("Total_Sessions:Q", title="Total Sessions"),
            y=alt.Y("Total_Purchase_Revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
            color=alt.Color("Avg_Bounce_Rate:Q", title="Avg Bounce Rate", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["Week", "Total_Sessions", "Total_Purchase_Revenue", "Avg_Bounce_Rate"]
        ).properties(width=700, height=400, title="Sessions vs Revenue (Color = Bounce Rate)")
        st.altair_chart(ga_scatter, use_container_width=True)
    else:
        st.info("No Google Analytics data available for the selected range.")


elif selected_page == "Facebook":
    st.title("Weekly Facebook Ads Performance")

    if not fb_overall.empty:
        fb_data = fb_overall.copy()
        fb_data["Profit"] = fb_data["revenue"] - fb_data["cost"]
        fb_data["ROI"] = fb_data.apply(
            lambda row: (row["revenue"] / row["cost"]) if row["cost"] > 0 else 0, axis=1
        )
        fb_data["CTR"] = fb_data.apply(
            lambda row: (row["clicks"] / row["impressions"] * 100) if row["impressions"] > 0 else 0, axis=1
        )
        fb_data["CPC"] = fb_data.apply(
            lambda row: (row["cost"] / row["clicks"]) if row["clicks"] > 0 else 0, axis=1
        )
        fb_data["Avg_Revenue_Per_Click"] = fb_data.apply(
            lambda row: (row["revenue"] / row["clicks"]) if row["clicks"] > 0 else 0, axis=1
        )

        # --- KEY FIX: Weighted (Overall) ROI ---
        total_revenue = fb_data["revenue"].sum()
        total_cost = fb_data["cost"].sum()
        overall_roi_ratio = total_revenue / total_cost if total_cost > 0 else 0

        # Overall Metrics
        total_profit_fb = total_revenue - total_cost

        st.markdown("---")
        st.subheader("Overall Facebook Metrics (Weekly)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        with col2:
            st.metric("Total Cost", f"${total_cost:,.2f}")
        with col3:
            st.metric("Total Profit", f"${total_profit_fb:,.2f}")
        with col4:
            st.metric("Overall ROI", f"{overall_roi_ratio*100:.2f}%")

        st.markdown("---")
        st.subheader("Weekly Facebook Revenue Trend")
        fb_line = (
            alt.Chart(fb_data)
            .mark_line(point=True, color="purple")
            .encode(
                x=alt.X("Week:T", title="Week"),
                y=alt.Y("revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                tooltip=["Week", "revenue", "cost", "Profit", "ROI", "CTR", "CPC", "Avg_Revenue_Per_Click"]
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(fb_line, use_container_width=True)

        st.markdown("---")
        st.subheader("Cost vs Revenue Scatter (Facebook)")
        fb_scatter = (
            alt.Chart(fb_data)
            .mark_circle(size=100)
            .encode(
                x=alt.X("cost:Q", title="Cost (USD)"),
                y=alt.Y("revenue:Q", title="Revenue (USD)", scale=alt.Scale(zero=False)),
                color=alt.Color("ROI:Q", title="ROI", scale=alt.Scale(scheme="greens")),
                tooltip=["Week", "revenue", "cost", "Profit", "ROI", "CTR", "CPC", "Avg_Revenue_Per_Click"]
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(fb_scatter, use_container_width=True)
        
        # --- New Section: Top Facebook Campaigns ---
        if not fb_top10.empty:
            st.markdown("---")
            st.subheader("Top 10 Facebook Campaigns for the Selected Period")
            # Aggregate the fb_top10 table by campaign.
            # Note: Adjust the group-by column if necessary (e.g., if your column is named "campaign" instead).
            aggregated_fb = fb_top10.groupby("campaign_name", as_index=False).agg({
                "revenue": "sum",
                "cost": "sum"
            })
            # Compute profit
            aggregated_fb["Profit"] = aggregated_fb["revenue"] - aggregated_fb["cost"]
            # Sort by Profit descending and take top 10
            top10_fb_campaigns = aggregated_fb.sort_values("Profit", ascending=False).head(10)
            
            fb_campaign_chart = alt.Chart(top10_fb_campaigns).mark_bar().encode(
                x=alt.X("campaign_name:N", sort="-y", title="Facebook Campaign"),
                y=alt.Y("Profit:Q", title="Profit (USD)", scale=alt.Scale(zero=False)),
                tooltip=["campaign_name", "Profit", "revenue", "cost"]
            ).properties(width=700, height=400, title="Top 10 Facebook Campaigns by Profit")
            
            st.altair_chart(fb_campaign_chart, use_container_width=True)
        else:
            st.info("No top campaign data available for Facebook Ads in the selected period.")
    else:
        st.info("No Facebook Ads data available for the selected range.")

elif selected_page == "Bing":
    st.title("Weekly Bing Ads Performance")
    if not bing_overall.empty:
        st.subheader("Overall Bing Ads Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Impressions", f"{bing_overall['impressions'].sum():,.0f}")
        with col2:
            st.metric("Total Clicks", f"{bing_overall['clicks'].sum():,.0f}")
        with col3:
            st.metric("Total Cost", f"${bing_overall['cost'].sum():,.2f}")
        with col4:
            st.metric("Total Profit", f"${bing_overall['Profit'].sum():,.2f}")
        with col5:
            st.metric("Total Revenue", f"${bing_overall['conversions_value'].sum():,.2f}")
        st.markdown("---")
        st.subheader("Bing ROI & CTR")
        col1, col2 = st.columns(2)
        with col1:
            avg_roi_bing = bing_overall["ROI"].mean() * 100
            st.metric("Avg ROI", f"{avg_roi_bing:.2f}%")
        with col2:
            avg_ctr_bing = bing_overall["CTR"].mean()
            st.metric("Avg CTR", f"{avg_ctr_bing:.2f}%")
        st.markdown("---")
        st.subheader("Bing Profit Trend")
        line_bing_profit = alt.Chart(bing_overall).mark_line(point=True, color="teal").encode(
            x=alt.X("Week:T", title="Week"),
            y=alt.Y("Profit:Q", title="Profit (USD)", scale=alt.Scale(zero=False)),
            tooltip=["Week", "Profit", "ROI", "CTR"]
        ).properties(width=700, height=400)
        st.altair_chart(line_bing_profit, use_container_width=True)
        st.markdown("---")
        st.subheader("Clicks vs Cost Scatter (Bing)")
        bing_scatter = alt.Chart(bing_overall).mark_circle(size=100).encode(
            x=alt.X("clicks:Q", title="Total Clicks"),
            y=alt.Y("cost:Q", title="Total Cost (USD)", scale=alt.Scale(zero=False)),
            color=alt.Color("ROI:Q", title="ROI", scale=alt.Scale(scheme="viridis")),
            tooltip=["Week", "clicks", "cost", "Profit", "ROI", "CTR"]
        ).properties(width=700, height=400)
        st.altair_chart(bing_scatter, use_container_width=True)
        st.markdown("---")
        st.subheader("Top 10 Bing Campaigns by Profit (Selected Period)")
        
        # IMPORTANT: Adjust the grouping column name if needed.
        # Here we use "campaign" assuming your data contains this column.
        aggregated_bing = bing_campaign.groupby("campaign_name", as_index=False).agg({
            "Profit": "sum",            # Total Profit
            "conversions_value": "sum",  # Total Revenue (for context)
            "clicks": "sum",
            "impressions": "sum",
            "ROI": "mean"               # Average ROI across weeks
        })
        
        # Sort by Profit descending and take the top 10
        top10_bing = aggregated_bing.sort_values("Profit", ascending=False).head(10)
        
        # Create a bar chart from the aggregated data
        chart = alt.Chart(top10_bing).mark_bar().encode(
            x=alt.X("campaign_name:N", sort="-y", title="Campaign"),
            y=alt.Y("Profit:Q", title="Total Profit (USD)", scale=alt.Scale(zero=False)),
            tooltip=["campaign_name", "Profit", "ROI", "conversions_value"]
        ).properties(width=700, height=400, title="Top 10 Bing Campaigns by Profit (Selected Period)")
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No Bing Ads data available for the selected range.")

        
elif selected_page == "Affiliate":
    st.title("Weekly Affiliate Sales Performance")
    
    # 1) Check if affiliate_overall is empty
    if affiliate_overall.empty:
        st.info("No Affiliate Sales data available for the selected range.")
    else:
        # Overall Affiliate Metrics
        total_sales = affiliate_overall["Total_Sales"].sum()
        total_commission = affiliate_overall["Total_Commission"].sum()
        total_fees = affiliate_overall["Total_Fees"].sum()
        total_discount = affiliate_overall["Total_Discount"].sum()
        net_after_commission = affiliate_overall["Net_After_Commission"].sum()
        order_count = affiliate_overall["Order_Count"].sum()
        
        st.markdown("---")
        st.subheader("Overall Affiliate Metrics (Weekly)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Sales", f"${total_sales:,.2f}")
        with col2:
            st.metric("Total Commission", f"${total_commission:,.2f}")
        with col3:
            st.metric("Total Fees", f"${total_fees:,.2f}")
        with col4:
            st.metric("Net Revenue", f"${net_after_commission:,.2f}")
        with col5:
            st.metric("Total Orders", f"{order_count:,}")
            
        # 2) Create a daily or weekly aggregated DataFrame
        # Here we create a daily aggregator as an example:
        affiliate_daily = (
            affiliate_overall
            .copy()
            .assign(date_only=lambda df: df["Week"].dt.date)   # extract date portion
            .groupby("date_only", as_index=False)
            .agg({"Total_Sales": "sum"})
            .rename(columns={"date_only": "Date"})
        )
        
        st.markdown("---")
        st.subheader("Affiliate Sales Trend (Aggregated by Day)")
        
        # 3) Build a line chart using altair
        aff_line = alt.Chart(affiliate_daily).mark_line(point=True, color="blue").encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Total_Sales:Q", title="Total Sales (USD)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Total_Sales:Q", title="Sales (USD)", format=",.2f")
            ]
        ).properties(width=700, height=400)
        st.altair_chart(aff_line, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Top Affiliate Websites by Sales & Profit")
        
        # 4) Show a simple bar chart for top affiliate websites
        agg_affiliate = affiliate_pub.groupby("publishername", as_index=False).agg({
        "Total_Sales": "sum",
        "Total_Commission": "sum",
        "Net_After_Commission": "sum",
        "Order_Count": "sum"
        })
                # Sort the aggregated data by Total_Sales in descending order and select the top 10 publishers.
        top_affiliate = agg_affiliate.sort_values("Total_Sales", ascending=False).head(10)
                
                # Create a bar chart based on the aggregated top 10 data.
        bar_chart = alt.Chart(top_affiliate).mark_bar().encode(
                    x=alt.X("publishername:N", sort='-y', title="Affiliate Website"),
                    y=alt.Y("Total_Sales:Q", title="Total Sales (USD)"),
                    tooltip=[
                        alt.Tooltip("publishername:N", title="Affiliate"),
                        alt.Tooltip("Total_Sales:Q", title="Total Sales", format=",.2f"),
                        alt.Tooltip("Total_Commission:Q", title="Total Commission", format=",.2f"),
                        alt.Tooltip("Net_After_Commission:Q", title="Net Revenue", format=",.2f"),
                        alt.Tooltip("Order_Count:Q", title="Orders", format=",.0f")
                    ]
                ).properties(
                    width=700,
                    height=400,
                    title="Top 10 Affiliate Websites by Sales (Selected Period)"
                )
            
            # Scatter chart for Sales vs Net Revenue
        scatter_chart = (
                alt.Chart(top_affiliate)
                .mark_circle(size=100)
                .encode(
                    x=alt.X("Total_Sales:Q", title="Total Sales (USD)", scale=alt.Scale(zero=False)),
                    y=alt.Y("Net_After_Commission:Q", title="Net Revenue (USD)", scale=alt.Scale(zero=False)),
                    size=alt.Size("Order_Count:Q", title="Orders"),
                    color=alt.Color("publishername:N", title="Affiliate"),
                    tooltip=["publishername", "Total_Sales", "Net_After_Commission", "Order_Count"]
                )
                .properties(width=700, height=400, title="Sales vs Net Revenue by Affiliate")
            )
        st.altair_chart(scatter_chart, use_container_width=True)
        
if selected_page == "Analytics":
    st.title("Interactive Data Analytics")
    st.markdown("### Explore Data, Build Your Own Plots, and Perform Statistical Analysis")
    
    ## -- Dataset Selection --
    dataset_option = st.selectbox("Choose Dataset:", ["Website Sales Overall", "Google Analytics", "Affiliate Sales"])
    if dataset_option == "Website Sales Overall":
        df = website_overall.copy()
    elif dataset_option == "Google Analytics":
        df = ga_overall.copy()
    elif dataset_option == "Affiliate Sales":
        df = affiliate_overall.copy()
    else:
        df = website_overall.copy()
    
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10))
    
    ## -- Summary Statistics --
    if st.checkbox("Show Summary Statistics"):
        st.write(df.describe())
    
    ## -- Custom Plotting --
    st.markdown("#### Create a Custom Plot")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    default_tooltip = ["Week"] if "Week" in df.columns else []
    
    if numeric_cols:
        col_x, col_y = st.columns(2)
        with col_x:
            x_var = st.selectbox("Select X-axis Variable", numeric_cols, index=0)
        with col_y:
            y_var = st.selectbox("Select Y-axis Variable", numeric_cols, index=1)
        plot_type = st.radio("Select Plot Type", ["Scatter Plot", "Line Plot", "Bar Chart"])
        
        if plot_type == "Scatter Plot":
            custom_chart = alt.Chart(df).mark_circle().encode(
                x=alt.X(f"{x_var}:Q", title=x_var),
                y=alt.Y(f"{y_var}:Q", title=y_var),
                tooltip=default_tooltip + [x_var, y_var]
            ).properties(width=700, height=400)
        elif plot_type == "Line Plot":
            custom_chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X(f"{x_var}:Q", title=x_var),
                y=alt.Y(f"{y_var}:Q", title=y_var),
                tooltip=default_tooltip + [x_var, y_var]
            ).properties(width=700, height=400)
        elif plot_type == "Bar Chart":
            custom_chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(f"{x_var}:N", title=x_var),
                y=alt.Y(f"{y_var}:Q", title=y_var),
                tooltip=default_tooltip + [x_var, y_var]
            ).properties(width=700, height=400)
        st.altair_chart(custom_chart, use_container_width=True)
    else:
        st.info("No numeric columns available in the dataset.")
    
    ## -- Correlation Analysis --
    st.markdown("#### Correlation Analysis")
    selected_vars = st.multiselect("Select variables for correlation analysis", numeric_cols, default=numeric_cols[:5])
    if selected_vars:
        corr_method = st.selectbox("Select Correlation Method", ["pearson", "spearman", "kendall"], index=0)
        corr_matrix = df[selected_vars].corr(method=corr_method)
        st.write("Correlation Matrix:")
        st.dataframe(corr_matrix)
        corr_long = corr_matrix.reset_index().melt('index')
        heatmap = alt.Chart(corr_long).mark_rect().encode(
            x=alt.X("variable:N", title="Variable"),
            y=alt.Y("index:N", title="Variable"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redyellowblue"), title="Correlation"),
            tooltip=["index", "variable", alt.Tooltip("value:Q", format=".2f")]
        ).properties(width=700, height=400, title="Correlation Heatmap")
        st.altair_chart(heatmap, use_container_width=True)
    
    ## -- Prophet Forecast for Profit --
    st.markdown("#### Prophet Forecasting of Profit")
    # Select variable to forecast. Default to 'Total_Profit' if available.
    if "Total_Profit" in numeric_cols:
        forecast_var = st.selectbox("Select Variable to Forecast", numeric_cols, index=numeric_cols.index("Total_Profit"))
    else:
        forecast_var = st.selectbox("Select Variable to Forecast", numeric_cols, index=0)
    if "Week" in df.columns:
        df_forecast = df[["Week", forecast_var]].dropna().rename(columns={"Week": "ds", forecast_var: "y"})
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
        horizon = st.number_input("Forecast Horizon (days):", min_value=1, value=30)
        if st.button("Run Forecast"):
            m = Prophet()
            m.fit(df_forecast)
            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)
            st.markdown("#### Forecast Data (Tail)")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            fig_prophet = m.plot(forecast)
            st.pyplot(fig_prophet)
            st.markdown("#### Forecast Components")
            fig_components = m.plot_components(forecast)
            st.pyplot(fig_components)
    
    ## -- Advanced Data Exploration: Filtering and Pivot Table --
    st.markdown("#### Advanced Data Exploration")
    st.write("Use the options below to filter the data and view pivot summaries.")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        filter_column = st.selectbox("Select a category to filter", categorical_cols)
        filter_val = st.text_input(f"Enter value for {filter_column} to filter by (or leave empty for all)")
        df_filtered = df[df[filter_column].str.contains(filter_val, case=False, na=False)] if filter_val else df.copy()
        st.markdown("#### Filtered Data Preview")
        st.dataframe(df_filtered.head(10))
        pivot_row = st.selectbox("Select row for pivot table", categorical_cols, index=0)
        pivot_col = st.selectbox("Select column for pivot table", categorical_cols, index=0)
        pivot_val = st.selectbox("Select value to aggregate", numeric_cols, index=0)
        agg_func = st.selectbox("Select Aggregation", ["sum", "mean", "max", "min"], index=0)
        if st.button("Show Pivot Table"):
            pivot_table = pd.pivot_table(df_filtered, values=pivot_val, index=pivot_row, columns=pivot_col, aggfunc=agg_func)
            st.write(pivot_table)
    
    ## -- Scatter Matrix --
    st.markdown("#### Scatter Matrix")
    if st.checkbox("Show Scatter Matrix for Selected Variables"):
        selected_pair_vars = st.multiselect("Select variables for scatter matrix", numeric_cols, default=numeric_cols[:4])
        if len(selected_pair_vars) >= 2:
            scatter_matrix = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X(alt.repeat("column"), type="quantitative"),
                y=alt.Y(alt.repeat("row"), type="quantitative"),
                color=alt.value("steelblue"),
                tooltip=selected_pair_vars
            ).properties(width=150, height=150).repeat(
                row=selected_pair_vars,
                column=selected_pair_vars
            )
            st.altair_chart(scatter_matrix, use_container_width=True)
        else:
            st.info("Please select at least 2 variables for the scatter matrix.")
    
    st.markdown("---")
    st.write("### End of Analytics Tab. Enjoy exploring the data!")
    
st.markdown("</div>", unsafe_allow_html=True)