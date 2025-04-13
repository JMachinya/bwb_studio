import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from urllib.parse import quote_plus

# =============================================================================
# Helper Functions
# =============================================================================
def get_week_start(dt):
    """
    Convert a pandas Timestamp to the start of the week (Monday).
    """
    if pd.isnull(dt):
        return None
    return dt - timedelta(days=dt.weekday())

def unify_source(src: str) -> str:
    """
    Cleans and standardizes the 'websource' string.
    """
    try:
        if src is None or src.strip() == "":
            return "unspecified"
        s = src.lower().strip()
        if '?' in s:
            s = s.split('?')[0]
        s = s.rstrip('/')
        if any(x in s for x in ["betterworldbooks.com", "betterworldbooks.zendesk.com", "(direct)"]):
            return "BetterWorldBooks"
        elif any(x in s for x in ["facebook.com", "m.facebook.com", "l.facebook.com", "lm.facebook.com", "business.facebook.com"]):
            return "Facebook"
        elif any(x in s for x in ["google", "g.doubleclick.net", "mail.google.com", "docs.google.com", "google_feed"]):
            return "Google"
        elif "bing" in s:
            return "Bing"
        elif "yahoo" in s:
            return "Yahoo"
        elif "duckduckgo" in s:
            return "DuckDuckGo"
        elif "affiliate" in s:
            return "Affiliate"
        elif any(x in s for x in ["klaviyo", "attentive", "sendinblue", "convertkit", "cj_feed"]):
            if "klaviyo" in s:
                return "Klaviyo"
            elif "attentive" in s:
                return "Attentive"
            elif "sendinblue" in s:
                return "Sendinblue"
            elif "convertkit" in s:
                return "ConvertKit"
            elif "cj_feed" in s:
                return "CJ"
        elif "substack" in s:
            return "Substack"
        elif "chatgpt" in s:
            return "ChatGPT"
        elif "trustpilot" in s:
            return "Trustpilot"
        elif "linkin.bio" in s or "later-linkinbio" in s:
            return "Linkin.bio"
        elif "nps" in s:
            return "NPS"
        else:
            return s
    except Exception:
        return "unspecified"

def clean_campaign(campaign):
    """
    Cleans a campaign string.
    """
    if pd.isna(campaign):
        return "unspecified"
    
    s = str(campaign).lower().strip()
    s = re.sub(r'\b(gclid|www\.google\.com|wwwgooglecom)\b', 'google campaign', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else "unspecified"

def derive_promo_label(row):
    """
    Creates a unified promotion label. If the 'promotion' column equals
    "NoPromotionApplied" (or is missing), returns "NoPromotion".
    Otherwise, returns the cleaned string from 'promotionid'.
    """
    if pd.isna(row['promotion']) or row['promotion'].strip() == "NoPromotionApplied":
        return "NoPromotion"
    else:
        return str(row['promotionid']).strip()

def calculate_uplift(pivot_df, metric):
    """
    Computes the uplift for each promotion relative to the "NoPromotion" baseline.
    Uplift = (Value - Baseline) / Baseline.
    """
    if 'NoPromotion' not in pivot_df.columns:
        raise ValueError("NoPromotion baseline column is missing in the pivot.")
    uplift_df = pivot_df.copy()
    baseline = pivot_df['NoPromotion']
    for col in pivot_df.columns:
        if col != 'NoPromotion':
            uplift_df[col + '_uplift'] = np.where(
                baseline != 0,
                (pivot_df[col] - baseline) / baseline,
                np.nan
            )
    return uplift_df

# =============================================================================
# Database Connection Setup
# =============================================================================
db_user   = "postgres"
db_pass   = "1999@Johannes"  # Password with special character '@'
db_host   = "localhost"
db_port   = "5432"
db_name   = "bwb_data"

encoded_db_pass = quote_plus(db_pass)
connection_string = f"postgresql://{db_user}:{encoded_db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

# =============================================================================
# Read the Website Sales Table and Clean Data
# =============================================================================
print("=== Reading Website Sales Data ===")
website_sales_df = pd.read_sql("SELECT * FROM website_sales", engine)

# Convert orderdate column to datetime and create a Week column (starting Monday)
website_sales_df['order_ts'] = pd.to_datetime(website_sales_df['orderdate'], format='%Y-%m-%d')
website_sales_df['Week'] = website_sales_df['order_ts'].apply(get_week_start)

# Clean the "websource" column and campaign column.
website_sales_df['cleaned_websource'] = website_sales_df['websource'].apply(unify_source)
website_sales_df['campaign'] = website_sales_df['campaign'].apply(clean_campaign)

print("Sample of campaign cleaning:")
print(website_sales_df[['campaign']].head(), "\n")


# =============================================================================
# (A) Weekly Aggregation by Campaign
# =============================================================================
print("=== (A) Weekly Aggregation by Campaign ===")
website_weekly_campaign = website_sales_df.groupby(['Week', 'campaign']).agg(
    Total_Revenue      = ('novusd', 'sum'),
    Total_Profit       = ('contributionmargin', 'sum'),
    Order_Count        = ('marketordernumber', 'count'),
    Total_Shipping     = ('shippingchargesusd', 'sum'),
    Total_Discount     = ('promodiscountsusd', 'sum'),
    Gross_Order_Value  = ('grossprodpaymentorder', 'sum'),
    Total_Books        = ('booksinorder', 'sum'),
    Returning_Count    = ('isreturningcustomer', 'sum')
).reset_index()

website_weekly_campaign['AOV'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    website_weekly_campaign['Total_Revenue'] / website_weekly_campaign['Order_Count'],
    0
)
website_weekly_campaign['Profit_Per_Order'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    website_weekly_campaign['Total_Profit'] / website_weekly_campaign['Order_Count'],
    0
)
website_weekly_campaign['Profit_Margin'] = np.where(
    website_weekly_campaign['Total_Revenue'] > 0,
    (website_weekly_campaign['Total_Profit'] / website_weekly_campaign['Total_Revenue']) * 100,
    0
)
website_weekly_campaign['Average_Discount_Percentage'] = np.where(
    website_weekly_campaign['Gross_Order_Value'] > 0,
    (website_weekly_campaign['Total_Discount'] / website_weekly_campaign['Gross_Order_Value']) * 100,
    0
)
website_weekly_campaign['Average_Discount_Amount'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    website_weekly_campaign['Total_Discount'] / website_weekly_campaign['Order_Count'],
    0
)
website_weekly_campaign['Gross_vs_Net_Difference'] = website_weekly_campaign['Gross_Order_Value'] - website_weekly_campaign['Total_Revenue']
website_weekly_campaign['Returning_Ratio'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    (website_weekly_campaign['Returning_Count'] / website_weekly_campaign['Order_Count']) * 100,
    0
)
website_weekly_campaign['Average_Order_Size'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    website_weekly_campaign['Total_Books'] / website_weekly_campaign['Order_Count'],
    0
)
website_weekly_campaign['Average_Shipping_Cost'] = np.where(
    website_weekly_campaign['Order_Count'] > 0,
    website_weekly_campaign['Total_Shipping'] / website_weekly_campaign['Order_Count'],
    0
)

print("Sample of website_weekly_campaign:")
print(website_weekly_campaign.head(), "\n")
website_weekly_campaign.to_sql("website_weekly_campaign_python", engine, if_exists="replace", index=False)
print("Created table: website_weekly_campaign_python")

# =============================================================================
# (B) Top Campaign per Week (by Total Profit)
# =============================================================================
print("=== (B) Top Campaign per Week ===")
website_weekly_campaign = website_weekly_campaign.sort_values(['Week', 'Total_Profit'], ascending=[True, False])
website_weekly_campaign['rank'] = website_weekly_campaign.groupby('Week').cumcount() + 1
website_weekly_top = website_weekly_campaign[website_weekly_campaign['rank'] <= 10].drop(columns='rank')
website_weekly_top.to_sql("website_weekly_top_campaign_python", engine, if_exists="replace", index=False)
print("Sample of website_weekly_top_campaign:")
print(website_weekly_top.head(), "\n")
print("Created table: website_weekly_top_campaign_python")

# =============================================================================
# (C) Extended Campaign Analysis (Overall Campaign Performance)
# =============================================================================
print("=== (C) Extended Campaign Analysis ===")
campaign_analysis = website_sales_df.groupby(['Week', 'campaign']).agg(
    Total_Revenue      = ('novusd', 'sum'),
    Total_Profit       = ('contributionmargin', 'sum'),
    Order_Count        = ('marketordernumber', 'count'),
    Total_Shipping     = ('shippingchargesusd', 'sum'),
    Total_Discount     = ('promodiscountsusd', 'sum'),
    Gross_Order_Value  = ('grossprodpaymentorder', 'sum'),
    Total_Books        = ('booksinorder', 'sum'),
    Returning_Count    = ('isreturningcustomer', 'sum')
).reset_index()

campaign_analysis['AOV'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    campaign_analysis['Total_Revenue'] / campaign_analysis['Order_Count'],
    0
)
campaign_analysis['Profit_Per_Order'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    campaign_analysis['Total_Profit'] / campaign_analysis['Order_Count'],
    0
)
campaign_analysis['Profit_Margin'] = np.where(
    campaign_analysis['Total_Revenue'] > 0,
    (campaign_analysis['Total_Profit'] / campaign_analysis['Total_Revenue']) * 100,
    0
)
campaign_analysis['Average_Discount_Percentage'] = np.where(
    campaign_analysis['Gross_Order_Value'] > 0,
    (campaign_analysis['Total_Discount'] / campaign_analysis['Gross_Order_Value']) * 100,
    0
)
campaign_analysis['Average_Discount_Amount'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    campaign_analysis['Total_Discount'] / campaign_analysis['Order_Count'],
    0
)
campaign_analysis['Gross_vs_Net_Difference'] = campaign_analysis['Gross_Order_Value'] - campaign_analysis['Total_Revenue']
campaign_analysis['Returning_Ratio'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    (campaign_analysis['Returning_Count'] / campaign_analysis['Order_Count']) * 100,
    0
)
campaign_analysis['Average_Order_Size'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    campaign_analysis['Total_Books'] / campaign_analysis['Order_Count'],
    0
)
campaign_analysis['Average_Shipping_Cost'] = np.where(
    campaign_analysis['Order_Count'] > 0,
    campaign_analysis['Total_Shipping'] / campaign_analysis['Order_Count'],
    0
)

campaign_analysis.to_sql("website_campaign_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_campaign_analysis_python")

# =============================================================================
# (C2) Segmented Campaign Analysis
# =============================================================================
print("=== (C2) Segmented Campaign Analysis ===")
campaign_segmentation_analysis = website_sales_df.groupby(
    ['Week', 'campaign', 'cleaned_websource', 'category', 'isreturningcustomer']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

campaign_segmentation_analysis['AOV'] = np.where(
    campaign_segmentation_analysis['Order_Count'] > 0,
    campaign_segmentation_analysis['Total_Revenue'] / campaign_segmentation_analysis['Order_Count'],
    0
)
campaign_segmentation_analysis['Profit_Margin'] = np.where(
    campaign_segmentation_analysis['Total_Revenue'] > 0,
    (campaign_segmentation_analysis['Total_Profit'] / campaign_segmentation_analysis['Total_Revenue']) * 100,
    0
)
campaign_segmentation_analysis.to_sql("website_campaign_segmentation_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_campaign_segmentation_analysis_python")

# =============================================================================
# (D) Promotion Analysis
# =============================================================================
# (D) Promotion Analysis: Top 10 Promotions per Week (Excluding 'NoPromotion')
print("=== (D) Promotion Analysis: Top 10 Promotions per Week (Excluding 'NoPromotion') ===")

# Create unified promotion labels.
website_sales_df['promo_clean'] = website_sales_df.apply(derive_promo_label, axis=1)


# Part D1: Compute Weekly Aggregation by Promotion

# Aggregate weekly metrics by promotion.
agg_metrics = website_sales_df.groupby(['Week', 'promo_clean']).agg(
    total_revenue = ('novusd', 'sum'),
    total_profit  = ('contributionmargin', 'sum'),
    total_orders  = ('marketordernumber', 'count')
).reset_index()


# Part D2: Select Top 10 Promotions (Excluding 'NoPromotion') per Week

# Exclude baseline rows.
agg_metrics_promos = agg_metrics[agg_metrics['promo_clean'] != "NoPromotion"]

# For each week, select the top 10 promotions by total profit.
top10_list = []
for week, grp in agg_metrics_promos.groupby('Week'):
    grp_top10 = grp.sort_values('total_profit', ascending=False).head(10).copy()
    top10_list.append(grp_top10)
top10_weekly_promos = pd.concat(top10_list).reset_index(drop=True)

print("\nTop 10 promotions by weekly profit (excluding NoPromotion):")
print(top10_weekly_promos)

# Write the table to PostgreSQL
top10_weekly_promos.to_sql("website_promotion_top10_weekly_python", engine, if_exists="replace", index=False)
print("Created table: website_promotion_top10_weekly_python")
# Display the top 10 promotions for each week   
print("Sample of top 10 promotions per week:")
print(top10_weekly_promos.head(20)) 
# =============================================================================
# Part D3: Compute Daily Averages for Each Promotion
# =============================================================================
# Compute the number of active days for each promotion per week.
daily_counts = website_sales_df.groupby(['Week', 'promo_clean']).agg(
    days_active = ('order_ts', 'nunique')
).reset_index()

# Merge the daily counts into the aggregated weekly metrics.
agg_metrics = pd.merge(agg_metrics, daily_counts, on=['Week', 'promo_clean'], how='left')
agg_metrics['avg_daily_revenue'] = agg_metrics['total_revenue'] / agg_metrics['days_active']
agg_metrics['avg_daily_profit'] = agg_metrics['total_profit'] / agg_metrics['days_active']
agg_metrics['avg_daily_orders'] = agg_metrics['total_orders'] / agg_metrics['days_active']

# =============================================================================
# Part D4: Create a Tidy Table for Top 10 Promotions with Uplift Metrics
# =============================================================================
# Restrict to only the top 10 promotions per week
top10_keys = top10_weekly_promos[['Week', 'promo_clean']].drop_duplicates()

# Merge to retrieve the daily averages for these top 10 promotions.
top10_daily = pd.merge(top10_keys, agg_metrics, on=['Week', 'promo_clean'], how='left')

# Retrieve the baseline ("NoPromotion") daily averages for each week.
baseline_data = agg_metrics[agg_metrics['promo_clean'] == "NoPromotion"][
    ['Week', 'avg_daily_revenue', 'avg_daily_profit', 'avg_daily_orders']
].rename(columns={
    'avg_daily_revenue': 'baseline_avg_daily_revenue',
    'avg_daily_profit': 'baseline_avg_daily_profit',
    'avg_daily_orders': 'baseline_avg_daily_orders'
})

# Merge baseline averages with the top 10 promotions (by Week).
top10_daily = pd.merge(top10_daily, baseline_data, on="Week", how="left")

# Compute uplift metrics relative to the baseline daily averages.
top10_daily["revenue_lift"] = np.where(
    top10_daily["baseline_avg_daily_revenue"] > 0,
    ((top10_daily["avg_daily_revenue"] - top10_daily["baseline_avg_daily_revenue"]) / top10_daily["baseline_avg_daily_revenue"]) * 100,
    np.nan
)
top10_daily["profit_lift"] = np.where(
    top10_daily["baseline_avg_daily_profit"] > 0,
    ((top10_daily["avg_daily_profit"] - top10_daily["baseline_avg_daily_profit"]) / top10_daily["baseline_avg_daily_profit"]) * 100,
    np.nan
)
top10_daily["orders_lift"] = np.where(
    top10_daily["baseline_avg_daily_orders"] > 0,
    ((top10_daily["avg_daily_orders"] - top10_daily["baseline_avg_daily_orders"]) / top10_daily["baseline_avg_daily_orders"]) * 100,
    np.nan
)

# Create the final tidy table containing only desired columns.
weekly_uplift_tidy = top10_daily[['Week', 'promo_clean', 'revenue_lift', 'profit_lift', 'orders_lift']]

print("\nFinal Tidy Weekly Uplift Table for Top 10 Promotions:")
print(weekly_uplift_tidy.head(20))

# =============================================================================
#Write the Tidy Table to PostgreSQL
# =============================================================================
weekly_uplift_tidy.to_sql("website_promotion_daily_uplift_top10_python", engine, if_exists="replace", index=False)
print("Created table: website_promotion_daily_uplift_top10_python")


#D3

promotion_analysis = website_sales_df.groupby(
    ['Week', 'promo_clean', 'cleaned_websource', 'category', 'isreturningcustomer']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count'),
    Total_Discount = ('promodiscountsusd', 'sum')
).reset_index()

promotion_analysis['AOV'] = np.where(
    promotion_analysis['Order_Count'] > 0,
    promotion_analysis['Total_Revenue'] / promotion_analysis['Order_Count'],
    0
)
promotion_analysis['Profit_Margin'] = np.where(
    promotion_analysis['Total_Revenue'] > 0,
    (promotion_analysis['Total_Profit'] / promotion_analysis['Total_Revenue']) * 100,
    0
)
promotion_analysis['Average_Discount_Amount'] = np.where(
    promotion_analysis['Order_Count'] > 0,
    promotion_analysis['Total_Discount'] / promotion_analysis['Order_Count'],
    0
)
promotion_analysis = promotion_analysis.sort_values(['Week', 'Total_Profit'], ascending=[True, False])
promotion_analysis['rank'] = promotion_analysis.groupby('Week').cumcount() + 1
promotion_analysis_top10 = promotion_analysis[promotion_analysis['rank'] <= 10].drop(columns='rank')

promotion_analysis_top10.to_sql("website_promotion_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_promotion_analysis_python")



# =============================================================================
# (E) Customer Analysis (New vs. Returning)
# =============================================================================
print("=== (E) Customer Analysis ===")
customer_analysis = website_sales_df.groupby(
    ['Week', 'isreturningcustomer']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

customer_analysis.to_sql("website_customer_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_customer_analysis_python")

# =============================================================================
# (F) Geographical & Source Analysis
# =============================================================================
print("=== (F) Geographical & Source Analysis ===")
geo_source_analysis = website_sales_df.groupby(
    ['Week', 'country', 'orderdestinationtype', 'cleaned_websource', 'medium']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

geo_source_analysis.to_sql("website_geo_source_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_geo_source_analysis_python")

# =============================================================================
# (G) Websource Analysis (Cleaning and Aggregation)
# =============================================================================
print("=== (G) Websource Analysis ===")
websource_analysis = website_sales_df.groupby(
    ['Week', 'cleaned_websource']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Total   = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

websource_analysis['AOV'] = np.where(
    websource_analysis['Order_Count'] > 0,
    websource_analysis['Total_Revenue'] / websource_analysis['Order_Count'],
    0
)
websource_analysis['Profit_Margin'] = np.where(
    websource_analysis['Total_Revenue'] > 0,
    (websource_analysis['Total_Total'] / websource_analysis['Total_Revenue']) * 100,
    0
)
websource_analysis.to_sql("website_websource_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_websource_analysis_python")

# =============================================================================
# (H) Promotion Strategy Analysis (Combined Pricing & Promotion)
# =============================================================================
print("=== (H) Promotion Strategy Analysis ===")
promotion_strategy_analysis = website_sales_df.groupby(
    ['Week', 'promo_clean',
     'campaign', 'pricingstrategy',
     'category']
).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count'),
    Total_Discount = ('promodiscountsusd', 'sum')
).reset_index()


promotion_strategy_analysis['Profit_Margin'] = np.where(
    promotion_strategy_analysis['Total_Revenue'] > 0,
    (promotion_strategy_analysis['Total_Profit'] / promotion_strategy_analysis['Total_Revenue']) * 100,
    0
)
promotion_strategy_analysis['Average_Discount_Amount'] = np.where(
    promotion_strategy_analysis['Order_Count'] > 0,
    promotion_strategy_analysis['Total_Discount'] / promotion_strategy_analysis['Order_Count'],
    0
)
promotion_strategy_analysis = promotion_strategy_analysis.sort_values(['Week', 'Total_Profit'], ascending=[True, False])
promotion_strategy_analysis['rank'] = promotion_strategy_analysis.groupby('Week').cumcount() + 1
promotion_strategy_analysis_top10 = promotion_strategy_analysis[promotion_strategy_analysis['rank'] <= 10].drop(columns='rank')

promotion_strategy_analysis_top10.to_sql("website_promotion_strategy_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_promotion_strategy_analysis_python")

# =============================================================================
# (I) Author Analysis
# =============================================================================
print("=== (I) Author Analysis ===")

author_analysis = website_sales_df.groupby(['Week', 'author']).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count'),
    Total_Books   = ('booksinorder', 'sum')
).reset_index()

author_analysis['AOV'] = np.where(
    author_analysis['Order_Count'] > 0,
    author_analysis['Total_Revenue'] / author_analysis['Order_Count'],
    0
)
author_analysis['Profit_Margin'] = np.where(
    author_analysis['Total_Revenue'] > 0,
    (author_analysis['Total_Profit'] / author_analysis['Total_Revenue']) * 100,
    0
)

# Save the detailed weekly author analysis to the database.
author_analysis.to_sql("website_author_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_author_analysis_python")

# =============================================================================
# (I3) Top 10 Authors by Profit Per Week
# =============================================================================
print("=== Top 10 Authors by Profit Per Week ===")
# First, sort the detailed weekly author metrics by Week and then by Total_Profit (descending).
author_weekly_ranked = author_analysis.sort_values(['Week', 'Total_Profit'], ascending=[True, False]).copy()

# Compute a rank per week.
author_weekly_ranked['rank'] = author_weekly_ranked.groupby('Week').cumcount() + 1

# Filter to keep only the top 10 authors for each week.
top10_authors_weekly = author_weekly_ranked[author_weekly_ranked['rank'] <= 10].drop(columns='rank')

print("Sample of top 10 weekly authors by profit:")
print(top10_authors_weekly.head())

# Send the top 10 weekly authors table to the database.
top10_authors_weekly.to_sql("website_top10_authors_by_profit_weekly", engine, if_exists="replace", index=False)
print("Created table: website_top10_authors_by_profit_weekly")

print("=== All Author Analysis Steps Completed Successfully ===")

# =============================================================================
# (J) Category Analysis
# =============================================================================
print("=== (J) Category Analysis ===")
category_analysis = website_sales_df.groupby(['Week', 'majorcategory']).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

category_analysis['AOV'] = np.where(
    category_analysis['Order_Count'] > 0,
    category_analysis['Total_Revenue'] / category_analysis['Order_Count'],
    0
)
category_analysis['Profit_Margin'] = np.where(
    category_analysis['Total_Revenue'] > 0,
    (category_analysis['Total_Profit'] / category_analysis['Total_Revenue']) * 100,
    0
)
category_analysis.to_sql("website_category_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_category_analysis_python")

# =============================================================================
# (K) Pricing Strategy Analysis (Including Test Metrics)
# =============================================================================
print("=== (K) Pricing Strategy Analysis: Including Test Metrics ===")

# For consistency, make a copy of website_sales_df
website_sales = website_sales_df.copy()

# Clean pricingstrategy to create strategy_base and flag is_test
website_sales['strategy_base'] = website_sales['pricingstrategy'].str.replace(r'Test$', '', regex=True)
website_sales['is_test'] = website_sales['pricingstrategy'].str.endswith('Test')

# Display the first few rows to check the new columns
print(website_sales.head())

# Optionally, view aggregated counts by strategy_base and is_test
agg_df = website_sales.groupby(['strategy_base', 'is_test']).size().reset_index(name='count')
print(agg_df)

# Ensure orderdate is datetime and create a Week column
website_sales['orderdate'] = pd.to_datetime(website_sales['orderdate'])
website_sales['Week'] = website_sales['orderdate'].dt.to_period('W').astype(str)

# Group by Week, cleaned pricing strategy (strategy_base), and test flag (is_test)
pricing_strategy_analysis = website_sales.groupby(['Week', 'strategy_base', 'is_test']).agg(
    Total_Revenue = ('novusd', 'sum'),
    Total_Profit  = ('contributionmargin', 'sum'),
    Order_Count   = ('marketordernumber', 'count')
).reset_index()

# Calculate additional metrics: Average Order Value (AOV) and Profit Margin
pricing_strategy_analysis['AOV'] = np.where(
    pricing_strategy_analysis['Order_Count'] > 0,
    pricing_strategy_analysis['Total_Revenue'] / pricing_strategy_analysis['Order_Count'],
    0
)
pricing_strategy_analysis['Profit_Margin'] = np.where(
    pricing_strategy_analysis['Total_Revenue'] > 0,
    (pricing_strategy_analysis['Total_Profit'] / pricing_strategy_analysis['Total_Revenue']) * 100,
    0
)

# Display the final DataFrame and save it
print(pricing_strategy_analysis)
pricing_strategy_analysis.to_sql("website_pricing_strategy_analysis_python", engine, if_exists="replace", index=False)
print("Created table: website_pricing_strategy_analysis_python")

# =============================================================================
# (C) Overall Website Weekly Aggregation (Ignoring Campaign)
# =============================================================================
print("=== Overall Website Weekly Aggregation ===")
website_weekly_overall = website_sales_df.groupby('Week').agg(
    Total_Revenue      = ('novusd', 'sum'),
    Total_Profit       = ('contributionmargin', 'sum'),
    Order_Count        = ('marketordernumber', 'count'),
    Total_Shipping     = ('shippingchargesusd', 'sum'),
    Total_Discount     = ('promodiscountsusd', 'sum'),
    Gross_Order_Value  = ('grossprodpaymentorder', 'sum'),
    Total_Books        = ('booksinorder', 'sum'),
    Returning_Count    = ('isreturningcustomer', 'sum')
).reset_index()

website_weekly_overall['AOV'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    website_weekly_overall['Total_Revenue'] / website_weekly_overall['Order_Count'],
    0
)
website_weekly_overall['Profit_Per_Order'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    website_weekly_overall['Total_Profit'] / website_weekly_overall['Order_Count'],
    0
)
website_weekly_overall['Profit_Margin'] = np.where(
    website_weekly_overall['Total_Revenue'] > 0,
    (website_weekly_overall['Total_Profit'] / website_weekly_overall['Total_Revenue']) * 100,
    0
)
website_weekly_overall['Average_Discount_Percentage'] = np.where(
    website_weekly_overall['Gross_Order_Value'] > 0,
    (website_weekly_overall['Total_Discount'] / website_weekly_overall['Gross_Order_Value']) * 100,
    0
)
website_weekly_overall['Average_Discount_Amount'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    website_weekly_overall['Total_Discount'] / website_weekly_overall['Order_Count'],
    0
)
website_weekly_overall['Gross_vs_Net_Difference'] = website_weekly_overall['Gross_Order_Value'] - website_weekly_overall['Total_Revenue']
website_weekly_overall['Returning_Ratio'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    (website_weekly_overall['Returning_Count'] / website_weekly_overall['Order_Count']) * 100,
    0
)
website_weekly_overall['Average_Order_Size'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    website_weekly_overall['Total_Books'] / website_weekly_overall['Order_Count'],
    0
)
website_weekly_overall['Average_Shipping_Cost'] = np.where(
    website_weekly_overall['Order_Count'] > 0,
    website_weekly_overall['Total_Shipping'] / website_weekly_overall['Order_Count'],
    0
)

website_weekly_overall.to_sql("website_weekly_overall_python", engine, if_exists="replace", index=False)
print("Created table: website_weekly_overall_python")

print("=== All Website Sales Aggregations and Analysis Completed Successfully ===")

# =============================================================================
# PART D: GOOGLE ANALYTICS WEEKLY AGGREGATIONS (Converted to pandas)
# =============================================================================
print("=== Starting Google Analytics Weekly Aggregation ===")


ga_df = pd.read_sql("SELECT * FROM ga_sessions_revenue", engine)

# -----------------------------------------------------------------------------
# Step 1: Convert date to Timestamp and create a Week column

ga_df["date_ts"] = pd.to_datetime(ga_df["date"], format="%m/%d/%Y")

# Create the Week column.
# We use the "W-MON" frequency so that each period starts on Monday. 
# Then, we convert each period to its start time (a Timestamp representing Monday).
ga_df["Week"] = ga_df["date_ts"].dt.to_period("W-MON").apply(lambda r: r.start_time)

# -----------------------------------------------------------------------------
# Step 2: Clean the landing_page column
# -----------------------------------------------------------------------------
# Replace null/missing landing_page values with "unspecified".
ga_df["landing_page"] = ga_df["landing_page"].fillna("unspecified")

# -----------------------------------------------------------------------------
# Step 3: Overall GA Weekly Aggregation
# -----------------------------------------------------------------------------
weekly_agg = ga_df.groupby("Week").agg(
    Total_Sessions=("sessions", "sum"),
    Avg_Bounce_Rate=("bounce_rate", "mean"),
    Total_Active_Users=("active_users", "sum"),
    Total_Purchase_Revenue=("purchase_revenue", "sum")
).reset_index()

# -----------------------------------------------------------------------------
# Step 4: Weekly Top Dimensions Aggregation for GA
# -----------------------------------------------------------------------------
weekly_dim_agg = ga_df.groupby(
    ["Week", "campaign_name", "source", "device_category", "landing_page", "medium"]
).agg(
    Total_Sessions=("sessions", "sum"),
    Avg_Bounce_Rate=("bounce_rate", "mean"),
    Total_Active_Users=("active_users", "sum"),
    Total_Purchase_Revenue=("purchase_revenue", "sum")
).reset_index()

# -----------------------------------------------------------------------------
# Step 5: Rank and Select the Top 10 Dimensions per Week
# -----------------------------------------------------------------------------
# Sort by Week and descending purchase revenue, then pick the top 10 rows for each week.
weekly_top_dimensions = (weekly_dim_agg
                         .sort_values(["Week", "Total_Purchase_Revenue"], ascending=[True, False])
                         .groupby("Week")
                         .head(10)
                         .reset_index(drop=True))

# -----------------------------------------------------------------------------
# Step 6: Write the Results to PostgreSQL
# -----------------------------------------------------------------------------
weekly_agg.to_sql("ga_weekly_aggregation_python", engine, if_exists="replace", index=False)
print("Created table: ga_weekly_aggregation_python")

weekly_top_dimensions.to_sql("ga_weekly_top_sources_python", engine, if_exists="replace", index=False)
print("Created table: ga_weekly_top_sources_python")


# =============================================================================
# PART C: GOOGLE ADS WEEKLY AGGREGATIONS (Pandas Version)
# =============================================================================
print("=== Starting Google Ads Weekly Aggregations (pandas version) ===")

# Read the Google Ads data from PostgreSQL.
# Replace "google_ads_stats" with your actual table name if needed.
google_ads_df = pd.read_sql("SELECT * FROM google_ads_stats", engine)

# -----------------------------------------------------------------------------
# Step 1: Convert date to Timestamp and Create Week Column
# -----------------------------------------------------------------------------
# Convert "date" column to datetime. The format is assumed to be "yyyy-MM-dd".
google_ads_df["date_ts"] = pd.to_datetime(google_ads_df["date"], format="%Y-%m-%d")

# Create the "Week" column by truncating to the week (Monday as the start)
# One approach is to convert the timestamp to a weekly period and then extract its start time.
google_ads_df["Week"] = google_ads_df["date_ts"].dt.to_period("W-MON").apply(lambda r: r.start_time)

# -----------------------------------------------------------------------------
# Step 2: Ensure the "cost" Column Exists
# -----------------------------------------------------------------------------
if "cost" not in google_ads_df.columns:
    # Compute cost from cost_micros if cost column is missing.
    google_ads_df["cost"] = google_ads_df["cost_micros"] / 1_000_000

# -----------------------------------------------------------------------------
# Step 3: Overall Google Ads Weekly Aggregation
# -----------------------------------------------------------------------------
# Group by Week and aggregate key metrics.
ga_overall = google_ads_df.groupby("Week").agg(
    conversions_value=pd.NamedAgg(column="conversions_value", aggfunc="sum"),
    cost=pd.NamedAgg(column="cost", aggfunc="sum"),
    impressions=pd.NamedAgg(column="impressions", aggfunc="sum"),
    clicks=pd.NamedAgg(column="clicks", aggfunc="sum"),
    conversions=pd.NamedAgg(column="conversions", aggfunc="sum")
).reset_index()

# Compute derived metrics.
ga_overall["Profit"] = ga_overall["conversions_value"] - ga_overall["cost"]
ga_overall["ROI"] = np.where(ga_overall["cost"] > 0,
                             ga_overall["conversions_value"] / ga_overall["cost"],
                             0)
ga_overall["CTR"] = np.where(ga_overall["impressions"] > 0,
                             (ga_overall["clicks"] / ga_overall["impressions"]) * 100,
                             0)
ga_overall["Conversion_Rate"] = np.where(ga_overall["clicks"] > 0,
                                         (ga_overall["conversions"] / ga_overall["clicks"]) * 100,
                                         0)
ga_overall["CPC"] = np.where(ga_overall["clicks"] > 0,
                             ga_overall["cost"] / ga_overall["clicks"],
                             0)
ga_overall["Cost_Per_Conversion"] = np.where(ga_overall["conversions"] > 0,
                                             ga_overall["cost"] / ga_overall["conversions"],
                                             0)
ga_overall["Avg_Revenue_Per_Conversion"] = np.where(ga_overall["conversions"] > 0,
                                                    ga_overall["conversions_value"] / ga_overall["conversions"],
                                                    0)
ga_overall["Avg_Revenue_Per_Click"] = np.where(ga_overall["clicks"] > 0,
                                               ga_overall["conversions_value"] / ga_overall["clicks"],
                                               0)

# -----------------------------------------------------------------------------
# Step 4: Write the Overall Google Ads Weekly Aggregation to PostgreSQL
# -----------------------------------------------------------------------------
ga_overall.to_sql("google_ads_weekly_overall_python", engine, if_exists="replace", index=False)
print("Created table: google_ads_weekly_overall_python")

# -----------------------------------------------------------------------------
# Step 5: Weekly Campaign-Level Aggregation for Google Ads
# -----------------------------------------------------------------------------
ga_campaign = google_ads_df.groupby(["Week", "campaign_name"]).agg(
    conversions_value=pd.NamedAgg(column="conversions_value", aggfunc="sum"),
    cost=pd.NamedAgg(column="cost", aggfunc="sum"),
    impressions=pd.NamedAgg(column="impressions", aggfunc="sum"),
    clicks=pd.NamedAgg(column="clicks", aggfunc="sum"),
    conversions=pd.NamedAgg(column="conversions", aggfunc="sum")
).reset_index()

# Compute derived metrics at the campaign level.
ga_campaign["Profit"] = ga_campaign["conversions_value"] - ga_campaign["cost"]
ga_campaign["ROI"] = np.where(ga_campaign["cost"] > 0,
                              ga_campaign["conversions_value"] / ga_campaign["cost"],
                              0)
ga_campaign["CTR"] = np.where(ga_campaign["impressions"] > 0,
                              (ga_campaign["clicks"] / ga_campaign["impressions"]) * 100,
                              0)
ga_campaign["Conversion_Rate"] = np.where(ga_campaign["clicks"] > 0,
                                          (ga_campaign["conversions"] / ga_campaign["clicks"]) * 100,
                                          0)
ga_campaign["CPC"] = np.where(ga_campaign["clicks"] > 0,
                              ga_campaign["cost"] / ga_campaign["clicks"],
                              0)
ga_campaign["Cost_Per_Conversion"] = np.where(ga_campaign["conversions"] > 0,
                                              ga_campaign["cost"] / ga_campaign["conversions"],
                                              0)
ga_campaign["Avg_Revenue_Per_Conversion"] = np.where(ga_campaign["conversions"] > 0,
                                                     ga_campaign["conversions_value"] / ga_campaign["conversions"],
                                                     0)
ga_campaign["Avg_Revenue_Per_Click"] = np.where(ga_campaign["clicks"] > 0,
                                                ga_campaign["conversions_value"] / ga_campaign["clicks"],
                                                0)

# Write the campaign-level aggregation to PostgreSQL.
ga_campaign.to_sql("google_ads_weekly_campaign_python", engine, if_exists="replace", index=False)
print("Created table: google_ads_weekly_campaign_python")

# -----------------------------------------------------------------------------
# Step 6: Rank and Select the Top 10 Campaigns per Week by Profit
# -----------------------------------------------------------------------------
# For each week, sort by Profit descending and take the top 10 rows.
ga_campaign_ranked = (
    ga_campaign.sort_values(["Week", "Profit"], ascending=[True, False])
    .groupby("Week")
    .head(10)
    .reset_index(drop=True)
)

# Write the top campaigns per week to PostgreSQL.
ga_campaign_ranked.to_sql("google_ads_weekly_top_campaign_python", engine, if_exists="replace", index=False)
print("Created table: google_ads_weekly_top_campaign_python")


# Read Data
fb_insights = pd.read_sql("SELECT * FROM facebook_ads_insights", engine)
fb_actions = pd.read_sql("SELECT * FROM facebook_ads_actions", engine)

# Define purchase action types
purchase_actions = [
    "purchase",
    "omni_purchase",
    "onsite_web_purchase",
    "offsite_conversion.fb_pixel_purchase",
    "onsite_web_app_purchase",
    "web_in_store_purchase"
]

# Pre-filter actions table: keep only rows where action_type is in purchase_actions
fb_actions = fb_actions[fb_actions["action_type"].isin(purchase_actions)].copy()

# Create a synthetic primary key in both tables based on: date, campaign_id, adset_id, ad_id
cols_str = ["date", "campaign_id", "adset_id", "ad_id"]
for col in cols_str:
    fb_insights[col] = fb_insights[col].astype(str)
    fb_actions[col] = fb_actions[col].astype(str)

fb_insights["primary_key"] = fb_insights["date"] + "_" + fb_insights["campaign_id"] + "_" + fb_insights["adset_id"] + "_" + fb_insights["ad_id"]
fb_actions["primary_key"] = fb_actions["date"] + "_" + fb_actions["campaign_id"] + "_" + fb_actions["adset_id"] + "_" + fb_actions["ad_id"]

# Merge insights and pre-filtered actions
fb_combined = pd.merge(
    fb_insights, fb_actions,
    on="primary_key",
    how="left",
    suffixes=("", "_act")
)

# Convert date to datetime, and derive Weekly column (Monday as start)
fb_combined["date"] = pd.to_datetime(fb_combined["date"], errors="coerce")
fb_combined["Week"] = fb_combined["date"] - pd.to_timedelta(fb_combined["date"].dt.weekday, unit="D")

# For the purchase actions, 'value' column from actions is now the revenue
# Calculate ROI, e.g., ROI = purchase_value / spend (handle division by zero)
fb_combined["ROI"] = fb_combined["value"].div(fb_combined["spend"]).replace([np.inf, -np.inf, np.nan], 0)

# Weekly Campaign-Level Aggregation
weekly_campaigns_fb = (
    fb_combined.groupby(["Week", "campaign_name"], dropna=False)
    .agg(
        revenue=("value", "sum"),
        cost=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        ROI=("ROI", "mean")
    )
    .reset_index()
)

# Select Top 10 Campaigns per Week by revenue
weekly_campaigns_fb = weekly_campaigns_fb.sort_values(["Week", "revenue"], ascending=[True, False])
top10_campaigns_fb = weekly_campaigns_fb.groupby("Week").head(10).reset_index(drop=True)

# Write Results to PostgreSQL
top10_campaigns_fb.to_sql("facebook_ads_top10_campaigns_python", con=engine, if_exists="replace", index=False)
print("Created table: facebook_ads_top10_campaigns_python")


# Aggregate the Facebook metrics by week:
weekly_facebook_overall = fb_combined.groupby("Week").agg(
    revenue = pd.NamedAgg(column="value", aggfunc="sum"),
    cost = pd.NamedAgg(column="spend", aggfunc="sum"),
    impressions = pd.NamedAgg(column="impressions", aggfunc="sum"),
    clicks = pd.NamedAgg(column="clicks", aggfunc="sum")
).reset_index()

# Compute additional metrics:
weekly_facebook_overall["Profit"] = weekly_facebook_overall["revenue"] - weekly_facebook_overall["cost"]
weekly_facebook_overall["ROI"] = np.where(
    weekly_facebook_overall["cost"] > 0,
    weekly_facebook_overall["revenue"] / weekly_facebook_overall["cost"],
    0
)
weekly_facebook_overall["CTR"] = np.where(
    weekly_facebook_overall["impressions"] > 0,
    (weekly_facebook_overall["clicks"] / weekly_facebook_overall["impressions"]) * 100,
    0
)
weekly_facebook_overall["CPC"] = np.where(
    weekly_facebook_overall["clicks"] > 0,
    weekly_facebook_overall["cost"] / weekly_facebook_overall["clicks"],
    0
)
weekly_facebook_overall["Avg_Revenue_Per_Click"] = np.where(
    weekly_facebook_overall["clicks"] > 0,
    weekly_facebook_overall["revenue"] / weekly_facebook_overall["clicks"],
    0
)

# Write the weekly campaign-level aggregation to PostgreSQL:
weekly_facebook_overall.to_sql(
    "facebook_ads_weekly_campaign_python",
    con=engine,
    if_exists="replace",
    index=False
)
print("Created table: facebook_ads_weekly_campaign_python")

# -----------------------------------------------------------------------------
# 5) Another Overall Weekly Aggregated Table (Additional Metrics)
# -----------------------------------------------------------------------------
# For an additional aggregation, you might compute overall totals along with average ROI.
# Here we compute total values and also compute the average ROI from the fb_combined data.
# (Note: If ROI wasn’t available per row, you can compute it on the weekly level as above.)
fb_weekly = fb_combined.groupby("Week").agg(
    Total_Value = pd.NamedAgg(column="value", aggfunc="sum"),
    Total_Spend = pd.NamedAgg(column="spend", aggfunc="sum"),
    Total_Impressions = pd.NamedAgg(column="impressions", aggfunc="sum"),
    Total_Clicks = pd.NamedAgg(column="clicks", aggfunc="sum"),
    Avg_ROI = pd.NamedAgg(column="ROI", aggfunc="mean")  # If ROI exists per row; otherwise, recalc it.
).reset_index()

# Write the overall weekly aggregated metrics to PostgreSQL:
fb_weekly.to_sql(
    "facebook_ads_weekly_aggregated_python",
    con=engine,
    if_exists="replace",
    index=False
)
print("Created table: facebook_ads_weekly_aggregated_python")

bing_df = pd.read_sql("SELECT * FROM bing_ads_stats", engine)

# -----------------------------------------------------------------------------
# 3) Convert Date and Create a Week Column
# -----------------------------------------------------------------------------
# Convert the 'date' column to datetime.
bing_df["date_ts"] = pd.to_datetime(bing_df["date"], format="%Y-%m-%d", errors="coerce")
# Create a 'Week' column with the start of the week (Monday).
bing_df["Week"] = bing_df["date_ts"] - pd.to_timedelta(bing_df["date_ts"].dt.weekday, unit="D")

# -----------------------------------------------------------------------------
# 4) Overall Weekly Aggregation for Bing Ads
# -----------------------------------------------------------------------------
weekly_overall = bing_df.groupby("Week", as_index=False).agg({
    "conversions_value": "sum",
    "spend": "sum",
    "impressions": "sum",
    "clicks": "sum",
    "conversions": "sum"
})

# Rename 'spend' to 'cost' for consistency.
weekly_overall = weekly_overall.rename(columns={"spend": "cost"})

# Compute advanced metrics.
weekly_overall["Profit"] = weekly_overall["conversions_value"] - weekly_overall["cost"]
weekly_overall["ROI"] = np.where(
    weekly_overall["cost"] > 0,
    weekly_overall["conversions_value"] / weekly_overall["cost"],
    0
)
weekly_overall["CTR"] = np.where(
    weekly_overall["impressions"] > 0,
    (weekly_overall["clicks"] / weekly_overall["impressions"]) * 100,
    0
)
weekly_overall["Conversion_Rate"] = np.where(
    weekly_overall["clicks"] > 0,
    (weekly_overall["conversions"] / weekly_overall["clicks"]) * 100,
    0
)
weekly_overall["CPC"] = np.where(
    weekly_overall["clicks"] > 0,
    weekly_overall["cost"] / weekly_overall["clicks"],
    0
)
weekly_overall["Cost_Per_Conversion"] = np.where(
    weekly_overall["conversions"] > 0,
    weekly_overall["cost"] / weekly_overall["conversions"],
    0
)
weekly_overall["Avg_Revenue_Per_Conversion"] = np.where(
    weekly_overall["conversions"] > 0,
    weekly_overall["conversions_value"] / weekly_overall["conversions"],
    0
)
weekly_overall["Avg_Revenue_Per_Click"] = np.where(
    weekly_overall["clicks"] > 0,
    weekly_overall["conversions_value"] / weekly_overall["clicks"],
    0
)

# -----------------------------------------------------------------------------
# 5) Write Overall Weekly Aggregation to PostgreSQL
# -----------------------------------------------------------------------------
weekly_overall.to_sql(
    "bing_ads_weekly_overall_python", 
    con=engine, 
    if_exists="replace", 
    index=False
)
print("Created table: bing_ads_weekly_overall_python")

# -----------------------------------------------------------------------------
# 6) Weekly Aggregation by Campaign
# -----------------------------------------------------------------------------
weekly_campaign = bing_df.groupby(["Week", "campaign_name"], as_index=False).agg({
    "conversions_value": "sum",
    "spend": "sum",
    "impressions": "sum",
    "clicks": "sum",
    "conversions": "sum"
})
weekly_campaign = weekly_campaign.rename(columns={"spend": "cost"})

# Compute advanced metrics at the campaign level.
weekly_campaign["Profit"] = weekly_campaign["conversions_value"] - weekly_campaign["cost"]
weekly_campaign["ROI"] = np.where(
    weekly_campaign["cost"] > 0,
    weekly_campaign["conversions_value"] / weekly_campaign["cost"],
    0
)
weekly_campaign["CTR"] = np.where(
    weekly_campaign["impressions"] > 0,
    (weekly_campaign["clicks"] / weekly_campaign["impressions"]) * 100,
    0
)
weekly_campaign["Conversion_Rate"] = np.where(
    weekly_campaign["clicks"] > 0,
    (weekly_campaign["conversions"] / weekly_campaign["clicks"]) * 100,
    0
)
weekly_campaign["CPC"] = np.where(
    weekly_campaign["clicks"] > 0,
    weekly_campaign["cost"] / weekly_campaign["clicks"],
    0
)
weekly_campaign["Cost_Per_Conversion"] = np.where(
    weekly_campaign["conversions"] > 0,
    weekly_campaign["cost"] / weekly_campaign["conversions"],
    0
)
weekly_campaign["Avg_Revenue_Per_Conversion"] = np.where(
    weekly_campaign["conversions"] > 0,
    weekly_campaign["conversions_value"] / weekly_campaign["conversions"],
    0
)
weekly_campaign["Avg_Revenue_Per_Click"] = np.where(
    weekly_campaign["clicks"] > 0,
    weekly_campaign["conversions_value"] / weekly_campaign["clicks"],
    0
)

# -----------------------------------------------------------------------------
# 7) Write Weekly Campaign-Level Aggregation to PostgreSQL
# -----------------------------------------------------------------------------
weekly_campaign.to_sql(
    "bing_ads_weekly_campaign_python", 
    con=engine, 
    if_exists="replace", 
    index=False
)
print("Created table: bing_ads_weekly_campaign_python")

# -----------------------------------------------------------------------------
# 8) Rank Campaigns by Profit within Each Week and Select Top 10
# -----------------------------------------------------------------------------
# First, rank the campaigns by Profit descending within each week.
weekly_campaign["rank"] = weekly_campaign.groupby("Week")["Profit"].rank(method="first", ascending=False)

# Select top 10 campaigns per week (where rank <= 10) and drop the rank column.
top10_campaigns = weekly_campaign[weekly_campaign["rank"] <= 10].copy()
top10_campaigns = top10_campaigns.drop(columns="rank")

# -----------------------------------------------------------------------------
# 9) Write Top 10 Weekly Campaigns to PostgreSQL
# -----------------------------------------------------------------------------
top10_campaigns.to_sql(
    "bing_ads_weekly_top_campaign_python", 
    con=engine, 
    if_exists="replace", 
    index=False
)
print("Created table: bing_ads_weekly_top_campaign_python")


# -----------------------------------------------------------------------------
# 2) Read Affiliate Sales Data from PostgreSQL
# -----------------------------------------------------------------------------
affiliate_df = pd.read_sql("SELECT * FROM affiliate_sales", engine)

# -----------------------------------------------------------------------------
# 3) Convert Date Column and Create Week Column
# -----------------------------------------------------------------------------
# Convert 'eventdate' to datetime and create a timestamp column
affiliate_df["event_ts"] = pd.to_datetime(affiliate_df["eventdate"], errors="coerce")
# Create 'Week' column which is the Monday of the week of 'event_ts'
affiliate_df["Week"] = affiliate_df["event_ts"] - pd.to_timedelta(affiliate_df["event_ts"].dt.weekday, unit="D")

# -----------------------------------------------------------------------------
# 4) Overall Weekly Aggregation for Affiliate Sales
# -----------------------------------------------------------------------------
affiliate_weekly = affiliate_df.groupby("Week", as_index=False).agg({
    "marketordernumber": pd.Series.nunique,  # distinct count
    "saleamountusd": "sum",
    "publishercommissionusd": "sum",
    "cjfeeusd": "sum",
    "orderdiscountusd": "sum",
    "correctedamountusd": "sum"
})
# Rename the columns accordingly:
affiliate_weekly = affiliate_weekly.rename(columns={
    "marketordernumber": "Order_Count",
    "saleamountusd": "Total_Sales",
    "publishercommissionusd": "Total_Commission",
    "cjfeeusd": "Total_Fees",
    "orderdiscountusd": "Total_Discount",
    "correctedamountusd": "Corrected_Amount"
})

# Calculate additional metric: Net_After_Commission = Total_Sales - Total_Commission - Total_Fees
affiliate_weekly["Net_After_Commission"] = affiliate_weekly["Total_Sales"] - affiliate_weekly["Total_Commission"] - affiliate_weekly["Total_Fees"]

# -----------------------------------------------------------------------------
# 5) Write Overall Affiliate Weekly Aggregation to PostgreSQL
# -----------------------------------------------------------------------------
affiliate_weekly.to_sql("affiliate_sales_weekly_overall_python",
                          con=engine,
                          if_exists="replace",
                          index=False)
print("Created table: affiliate_sales_weekly_overall_python")

# -----------------------------------------------------------------------------
# 6) Weekly Aggregation by Publisher
# -----------------------------------------------------------------------------
affiliate_weekly_pub = affiliate_df.groupby(["Week", "publishername"], as_index=False).agg({
    "marketordernumber": pd.Series.nunique,
    "saleamountusd": "sum",
    "publishercommissionusd": "sum"
})
affiliate_weekly_pub = affiliate_weekly_pub.rename(columns={
    "marketordernumber": "Order_Count",
    "saleamountusd": "Total_Sales",
    "publishercommissionusd": "Total_Commission"
})
# Calculate Net_After_Commission = Total_Sales - Total_Commission
affiliate_weekly_pub["Net_After_Commission"] = affiliate_weekly_pub["Total_Sales"] - affiliate_weekly_pub["Total_Commission"]

# -----------------------------------------------------------------------------
# 7) Write Weekly Aggregation by Publisher to PostgreSQL
# -----------------------------------------------------------------------------
affiliate_weekly_pub.to_sql("affiliate_sales_weekly_publisher_python",
                              con=engine,
                              if_exists="replace",
                              index=False)
print("Created table: affiliate_sales_weekly_publisher_python")

# -----------------------------------------------------------------------------
# 8) Rank Publishers within Each Week and Select Top 10 by Net_After_Commission
# -----------------------------------------------------------------------------
def top_n_per_week(df, group_col, sort_col, n=10):
    # For each week, sort by sort_col descending and take the top n rows.
    return df.sort_values(by=sort_col, ascending=False).groupby(group_col).head(n)

top10_publishers_weekly = top_n_per_week(affiliate_weekly_pub, group_col="Week", sort_col="Net_After_Commission", n=10)

# -----------------------------------------------------------------------------
# 9) Write Top 10 Weekly Publishers to PostgreSQL
# -----------------------------------------------------------------------------
top10_publishers_weekly.to_sql("affiliate_sales_weekly_top_publisher_python",
                               con=engine,
                               if_exists="replace",
                               index=False)
print("Created table: affiliate_sales_weekly_top_publisher_python")

print("=== All Affiliate Sales Weekly Aggregations Completed Successfully ===")