import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict

random.seed(42)
np.random.seed(42)


# --- Helper Functions ---
def extract_size(name):
    try:
        return int([s for s in name.split() if 'g' in s][0].replace('g', ''))
    except:
        return np.nan

def generate_sales_data_with_campaigns(products_csv_path, start_date, end_date, stores_per_region=5, num_campaigns=8):
    products = pd.read_csv(products_csv_path)
    products["Size_g"] = products["Product_Name"].apply(extract_size)
    products["Popularity"] = products["Category"].map({
        "Wafers": 5, "Namkeen": 4, "Western-snacks": 3, "Noodles": 2, "Confectionery": 3
    })
    products["SKU"] = ["SKU" + str(i+1).zfill(4) for i in range(len(products))]

    regions = ['Gujarat', 'Maharashtra', 'Madhya Pradesh', 'Rajasthan', 'Goa']
    regional_multipliers = {
        "Gujarat": 1.2, "Maharashtra": 1.0, "Madhya Pradesh": 0.9, "Rajasthan": 0.8, "Goa": 0.6
    }
    store_ids = [f"{r[:3].upper()}_{i+1}" for r in regions for i in range(stores_per_region)]
    date_range = pd.date_range(start=start_date, end=end_date)

    # --- Generate campaigns ---
    def generate_marketing_data(sku_list, regions, num_campaigns):
        campaigns = []
        channels = ["Meta Ads", "TV", "Posters", "Instagram", "WhatsApp"]
        for i in range(num_campaigns):
            sku = random.choice(sku_list)
            region = random.choice(regions)
            start = datetime(2024, random.randint(4, 6), random.randint(1, 20))
            end = start + timedelta(days=random.choice([7, 10, 14]))
            campaigns.append([
                f"MKT{i+1:03}", start.date(), end.date(), sku,
                random.randint(30000, 80000), random.choice(channels), region
            ])
        return pd.DataFrame(campaigns, columns=[
            "Campaign_ID", "Start_Date", "End_Date", "Target_SKU", "Budget", "Channel", "Region"
        ])

    campaign_df = generate_marketing_data(products["SKU"].tolist(), regions, num_campaigns)
    channel_effectiveness = {"Meta Ads": 1.3, "TV": 1.4, "Posters": 1.1, "Instagram": 1.2, "WhatsApp": 1.0}

    # --- Campaign mappings ---
    campaign_map = {}
    channel_map = {}
    for _, row in campaign_df.iterrows():
        key = (row["Target_SKU"], row["Region"])
        start = pd.to_datetime(row["Start_Date"]).date()
        end = pd.to_datetime(row["End_Date"]).date()
        campaign_map.setdefault(key, []).append((start, end))
        channel_map[key] = row["Channel"]

    # --- External metadata ---
    weather_options = ["Sunny", "Cloudy", "Rainy", "Storm", "Clear"]
    holiday_calendar = {
        "2024-01-14": "Makar Sankranti", "2024-01-26": "Republic Day",
        "2024-03-25": "Holi", "2024-08-15": "Independence Day",
        "2024-10-02": "Gandhi Jayanti", "2024-10-31": "Diwali",
        "2024-11-01": "New Year (Gujarati)", "2025-01-01": "New Year's Day",
        "2025-03-14": "Maha Shivratri", "2025-03-20": "Holi"
    }
    holiday_calendar = {pd.to_datetime(k).date(): v for k, v in holiday_calendar.items()}
    date_metadata = {
        d.date(): {
            "Holiday": holiday_calendar.get(d.date(), "None"),
            "Weather": random.choice(weather_options)
        }
        for d in date_range
    }

    # --- Sales loop ---
    sales_data = []
    for date in date_range:
        for store in store_ids:
            region = [r for r in regions if r[:3].upper() in store][0]
            for _, row in products.iterrows():
                sku = row["SKU"]
                category = row["Category"]
                month = date.month

                base_sales = np.clip(100 - row["Price"] + row["Popularity"] * 5, 1, None)
                regional_factor = regional_multipliers[region]
                weekday_factor = 1.3 if date.weekday() >= 5 else 1.0
                noise = random.uniform(0.8, 1.2)

                # Promotion check
                key = (sku, region)
                date_obj = date.date()
                is_promo = any(start <= date_obj <= end for (start, end) in campaign_map.get(key, []))
                channel = channel_map.get(key, "WhatsApp")
                promo_boost = 1.5 * channel_effectiveness.get(channel, 1.0) if is_promo else 1.0

                # External effect modifiers
                holiday = date_metadata[date.date()]["Holiday"]
                weather = date_metadata[date.date()]["Weather"]
                holiday_boost = 1.2 if holiday != "None" else 1.0
                weather_effect = {
                    "Sunny": 1.1, "Cloudy": 1.0, "Rainy": 0.85, "Storm": 0.6, "Clear": 1.0
                }.get(weather, 1.0)
                competitor_price = round(random.uniform(0.85, 1.15) * row["Price"], 2)
                price_competition = 0.9 if competitor_price < row["Price"] else 1.1

                # Seasonal effect
                seasonal_multiplier = 1.0
                if month in [4, 5, 6] and category in ["Wafers", "Confectionery"]:
                    seasonal_multiplier = 1.2
                elif month in [7, 8, 9] and category in ["Wafers", "Namkeen"]:
                    seasonal_multiplier = 0.85
                elif month in [10, 11]:
                    seasonal_multiplier = 1.3
                elif month in [12, 1]:
                    if category in ["Noodles", "Namkeen"]:
                        seasonal_multiplier = 1.15
                    else:
                        seasonal_multiplier = 0.9
                elif month == 3:
                    seasonal_multiplier = 0.95

                # Final sales calculation
                units_sold = int(base_sales * regional_factor * weekday_factor *
                                 promo_boost * holiday_boost * weather_effect *
                                 price_competition * seasonal_multiplier * noise)

                revenue = units_sold * row["Price"]

                sales_data.append([
                    date.date(), store, region, sku, row["Product_Name"], row["Category"],
                    units_sold, row["Price"], revenue, is_promo
                ])

    sales_df = pd.DataFrame(sales_data, columns=[
        "Date", "Store_ID", "Region", "SKU", "Product_Name", "Category",
        "Units_Sold", "Price", "Revenue", "On_Promotion"
    ])

    return sales_df, campaign_df


def generate_inventory_data(sales_df):
    inventory_records = []
    initial_stock = defaultdict(lambda: random.randint(500, 1000))
    returns_rate = 0.01
    damage_rate = 0.005
    sku_replenishment = defaultdict(lambda: random.randint(400, 600))
    threshold = defaultdict(lambda: 200)

    grouped = sales_df.groupby(["Date", "Region", "SKU"])["Units_Sold"].sum().reset_index()

    for _, row in grouped.iterrows():
        date, region, sku, units_sold = row["Date"], row["Region"], row["SKU"], row["Units_Sold"]
        key = (region, sku)
        opening = initial_stock[key]
        returns = int(units_sold * returns_rate)
        damaged = int(units_sold * damage_rate)
        closing = opening - units_sold + returns - damaged

        if closing < threshold[sku]:
            closing += sku_replenishment[sku]

        inventory_records.append([
            date, region, sku, opening, units_sold, returns, damaged, closing
        ])
        initial_stock[key] = closing

    return pd.DataFrame(inventory_records, columns=[
        "Date", "Region", "SKU", "Opening_Stock", "Units_Sold", "Returns", "Damaged", "Closing_Stock"
    ])

def generate_crm_data(sales_df):
    store_ids = sales_df["Store_ID"].unique().tolist()
    retailer_map = {store: f"R{idx+1:03}" for idx, store in enumerate(store_ids)}
    sales_df["Retailer_ID"] = sales_df["Store_ID"].map(retailer_map)

    # 🛠 Fix: Convert Date column to Timestamp
    sales_df["Date"] = pd.to_datetime(sales_df["Date"])

    cutoff = pd.Timestamp(datetime.today() - timedelta(days=30))

    monthly_sales = (
        sales_df[sales_df["Date"] >= cutoff]
        .groupby("Retailer_ID")["Revenue"].sum().reset_index()
    )

    def generate_kpis(sales):
        if sales > 5000000:
            return 0, 5
        elif sales > 4000000:
            return 1, 4
        elif sales > 1000000:
            return 2, 3
        else:
            return 3, random.randint(1, 2)

    crm_rows = []
    for _, row in monthly_sales.iterrows():
        rid = row["Retailer_ID"]
        region = sales_df[sales_df["Retailer_ID"] == rid]["Region"].mode()[0]
        last_order = sales_df[sales_df["Retailer_ID"] == rid]["Date"].max()
        complaints, satisfaction = generate_kpis(row["Revenue"])
        crm_rows.append([rid, region, last_order, complaints, satisfaction, row["Revenue"]])

    return pd.DataFrame(crm_rows, columns=[
        "Retailer_ID", "Region", "Last_Order_Date", "Complaints", "Satisfaction", "Monthly_Sales"
    ])

def generate_warehouse_data(inventory_df):
    warehouse_records = []
    closing_stock = defaultdict(lambda: random.randint(500, 1000))

    grouped = inventory_df.groupby(["Date", "Region", "SKU"]).agg({
        "Units_Sold": "sum", "Returns": "sum", "Damaged": "sum"
    }).reset_index()

    for _, row in grouped.iterrows():
        date, region, sku = row["Date"], row["Region"], row["SKU"]
        sold = row["Units_Sold"]
        returned = row["Returns"]
        damaged = row["Damaged"]

        prev_stock = closing_stock[(region, sku)]
        stock_out = sold
        stock_in = random.randint(200, 500) if prev_stock < 600 else 0

        closing = prev_stock + stock_in - stock_out + returned - damaged
        closing_stock[(region, sku)] = closing

        warehouse_records.append([
            date, region, sku, stock_in, stock_out, damaged, returned, closing
        ])

    return pd.DataFrame(warehouse_records, columns=[
        "Date", "Region", "SKU", "Stock_In", "Stock_Out", "Damaged", "Returned", "Closing_Stock"
    ])


def generate_external_data(sales_df):
    weather_options = ["Sunny", "Cloudy", "Rainy", "Storm", "Clear"]
    holiday_calendar = {
        "2024-01-14": "Makar Sankranti", "2024-01-26": "Republic Day",
        "2024-03-25": "Holi", "2024-08-15": "Independence Day",
        "2024-10-02": "Gandhi Jayanti", "2024-10-31": "Diwali",
        "2024-11-01": "New Year (Gujarati)", "2025-01-01": "New Year's Day",
        "2025-03-14": "Maha Shivratri", "2025-03-20": "Holi"
    }
    holiday_calendar = {pd.to_datetime(k).date(): v for k, v in holiday_calendar.items()}

    sales_df["Date"] = pd.to_datetime(sales_df["Date"])
    avg_price_df = sales_df.groupby(["Date", "SKU"])["Price"].mean().reset_index()

    date_metadata = {
        date.date(): {
            "Holiday": holiday_calendar.get(date.date(), "No Holiday"),
            "Weather": random.choice(weather_options)
        }
        for date in avg_price_df["Date"].unique()
    }

    external_data = [
        [
            row["Date"].date(),
            date_metadata[row["Date"].date()]["Holiday"],
            date_metadata[row["Date"].date()]["Weather"],
            row["SKU"],
            round(random.uniform(0.85, 1.15) * row["Price"], 2)
        ]
        for _, row in avg_price_df.iterrows()
    ]

    return pd.DataFrame(external_data, columns=[
        "Date", "Holiday", "Weather", "SKU", "Competitor_Price"
    ])
