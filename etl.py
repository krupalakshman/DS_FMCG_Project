# data_pipeline.py

import pandas as pd

def run_pipeline():
    # --- File Paths ---
    BASE_DIR = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/synthetic"
    
    sales_df = pd.read_csv(f"{BASE_DIR}/synthetic_sales.csv", parse_dates=["Date"])
    campaign_df = pd.read_csv(f"{BASE_DIR}/synthetic_campaign.csv", parse_dates=["Start_Date", "End_Date"])
    inventory_df = pd.read_csv(f"{BASE_DIR}/synthetic_inventory.csv", parse_dates=["Date"])
    warehouse_df = pd.read_csv(f"{BASE_DIR}/synthetic_warehouse.csv", parse_dates=["Date"])
    crm_df = pd.read_csv(f"{BASE_DIR}/synthetic_crm.csv", parse_dates=["Last_Order_Date"])
    external_df = pd.read_csv(f"{BASE_DIR}/synthetic_external.csv", parse_dates=["Date"])

    # === Campaign Flatten ===
    campaign_flattened = []
    for _, row in campaign_df.iterrows():
        for day in pd.date_range(row["Start_Date"], row["End_Date"]):
            campaign_flattened.append([
                day.date(), row["Target_SKU"], row["Region"], row["Campaign_ID"]
            ])
    campaign_expanded_df = pd.DataFrame(campaign_flattened, columns=[
        "Date", "SKU", "Region", "Campaign_ID"
    ])
    campaign_expanded_df["Date"] = pd.to_datetime(campaign_expanded_df["Date"])

    # === Retailer Mapping ===
    store_ids = sales_df["Store_ID"].unique().tolist()
    retailer_map = {store: f"R{idx+1:03}" for idx, store in enumerate(store_ids)}
    sales_df["Retailer_ID"] = sales_df["Store_ID"].map(retailer_map)

    # === Merge Data Sources ===
    merged_df = pd.merge(sales_df, campaign_expanded_df, how="left", on=["Date", "SKU", "Region"])
    merged_df = pd.merge(merged_df, campaign_df[["Campaign_ID", "Budget", "Channel"]], how="left", on="Campaign_ID")
    merged_df = pd.merge(merged_df, inventory_df, how="left", on=["Date", "SKU", "Region"], suffixes=('', '_Inventory'))
    merged_df = pd.merge(merged_df, warehouse_df, how="left", on=["Date", "SKU", "Region"], suffixes=('', '_Warehouse'))
    merged_df = pd.merge(merged_df, external_df, how="left", on=["Date", "SKU"])
    merged_df = pd.merge(merged_df, crm_df, how="left", on=["Retailer_ID", "Region"])

    # === Clean Columns ===
    final_cols = [
        "Date", "Store_ID", "Retailer_ID", "Region", "SKU", "Product_Name", "Category",
        "Units_Sold", "Price", "Revenue", "On_Promotion",
        "Campaign_ID", "Budget", "Channel",
        "Opening_Stock", "Returns", "Damaged", "Closing_Stock",
        "Stock_In", "Stock_Out", "Damaged_Warehouse", "Returned", "Closing_Stock_Warehouse",
        "Holiday", "Weather", "Competitor_Price",
        "Last_Order_Date", "Complaints", "Satisfaction", "Monthly_Sales"
    ]
    final_cols = [col for col in final_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols]

    # === Clean campaign info ===
    merged_df["Campaign_ID"] = merged_df["Campaign_ID"].fillna("No_campaign")
    merged_df["Budget"] = 0.0
    mask = ~merged_df.duplicated(subset=["Campaign_ID"])
    merged_df.loc[mask, "Budget"] = merged_df.loc[mask, "Campaign_ID"].map(
        campaign_df.set_index("Campaign_ID")["Budget"]
    )
    merged_df["Channel"] = merged_df["Channel"].fillna("None")

    # === ROI Report ===
    roi_df = merged_df.groupby("Campaign_ID").agg({"Revenue": "sum"}).reset_index()
    roi_df = roi_df.merge(campaign_df[["Campaign_ID", "Budget"]], on="Campaign_ID", how="left")
    roi_df["ROI"] = (roi_df["Revenue"] - roi_df["Budget"]) / roi_df["Budget"]

    merged_df.to_csv(f"{BASE_DIR}/../processed/merged_dataset.csv", index=False)
    print("✅ Merged dataset created successfully.")

    # === Product Master ===
    products_raw = pd.read_csv("C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/raw/balaji_products.csv")
    sales = pd.read_csv(f"{BASE_DIR}/synthetic_sales.csv")
    campaign = pd.read_csv(f"{BASE_DIR}/synthetic_campaign.csv")
    sales_products = sales[["SKU", "Product_Name", "Category", "Price"]].drop_duplicates()
    campaign_products = campaign.merge(sales_products, left_on="Target_SKU", right_on="SKU", how="left")
    campaign_products = campaign_products[["SKU", "Product_Name", "Category", "Price"]].drop_duplicates()
    product_master = pd.concat([products_raw, sales_products, campaign_products], ignore_index=True).drop_duplicates(subset=["SKU"]).dropna()

    def extract_size(name):
        try:
            return int([s for s in name.split() if 'g' in s][0].replace('g', '').strip())
        except:
            return None

    def normalize_name(name):
        name = name.lower().replace("balaji", "").replace("-", "").strip()
        return " ".join(name.split())

    product_master["Size_g"] = product_master["Product_Name"].apply(extract_size)
    product_master["Normalized_Name"] = product_master["Product_Name"].apply(normalize_name)
    product_master["Brand"] = "Balaji"

    def get_subcategory(row):
        name = row["Normalized_Name"]
        if "twist" in name or "masti" in name or "chaska" in name:
            return "Flavored"
        elif "salted" in name:
            return "Plain"
        elif "noodle" in name:
            return "Instant"
        elif "chikki" in name:
            return "Sweet"
        else:
            return "Other"

    product_master["Subcategory"] = product_master.apply(get_subcategory, axis=1)

    product_master_final = product_master[[
        "SKU", "Product_Name", "Brand", "Category", "Size_g", "Price"
    ]].drop_duplicates().sort_values("SKU")

    product_master_final.to_csv("C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/products_master.csv", index=False)

    # === Timeseries Expansion ===
    df = merged_df.copy()
    store_sku_date = pd.MultiIndex.from_product([
        df["Store_ID"].unique(), df["SKU"].unique(),
        pd.date_range(df["Date"].min(), df["Date"].max(), freq='D')
    ], names=["Store_ID", "SKU", "Date"])
    
    ts_full = pd.DataFrame(index=store_sku_date).reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    ts_full = ts_full.merge(df, on=["Store_ID", "SKU", "Date"], how="left")

    default_fill = {
        'Units_Sold': 0,
        'Retailer_ID': 'UNKNOWN',
        'Region': 'UNKNOWN',
        'Product_Name': 'UNKNOWN',
        'Category': 'UNKNOWN',
        'Price': 0.0,
        'Revenue': 0.0,
        'On_Promotion': False,
        'Campaign_ID': 'NONE',
        'Budget': 0.0,
        'Channel': 'Offline',
        'Opening_Stock': 0,
        'Returns': 0,
        'Damaged': 0,
        'Closing_Stock': 0,
        'Stock_In': 0,
        'Stock_Out': 0,
        'Damaged_Warehouse': 0,
        'Returned': 0,
        'Closing_Stock_Warehouse': 0,
        'Holiday': 'No',
        'Weather': 'Clear',
        'Competitor_Price': df["Competitor_Price"].mean(),
        'Last_Order_Date': pd.NaT,
        'Complaints': 0,
        'Satisfaction': 5,
        'Monthly_Sales': 0.0
    }

    ts_full.fillna(value=default_fill, inplace=True)
    ts_full.sort_values(['Store_ID', 'SKU', 'Date'], inplace=True)
    ts_full[['Retailer_ID', 'Region', 'Product_Name', 'Category', 'Price']] = (
        ts_full.groupby(['Store_ID', 'SKU'])[['Retailer_ID', 'Region', 'Product_Name', 'Category', 'Price']]
        .ffill().bfill()
    )
    ts_full['Last_Order_Date'] = ts_full.groupby(['Store_ID', 'SKU'])['Last_Order_Date'].ffill().bfill()
    
    group_cols = ['Store_ID', 'SKU']
    ts_full['Lag_1D_Units_Sold'] = ts_full.groupby(group_cols)['Units_Sold'].shift(1)
    ts_full['Lag_7D_Units_Sold'] = ts_full.groupby(group_cols)['Units_Sold'].shift(7)
    ts_full['Rolling_3D_Units_Sold'] = ts_full.groupby(group_cols)['Units_Sold'].transform(lambda x: x.shift(1).rolling(3).mean())
    ts_full['Rolling_7D_Units_Sold'] = ts_full.groupby(group_cols)['Units_Sold'].transform(lambda x: x.shift(1).rolling(7).mean())
    ts_full['Rolling_14D_Units_Sold'] = ts_full.groupby(group_cols)['Units_Sold'].transform(lambda x: x.shift(1).rolling(14).mean())

    ts_final = ts_full.dropna(subset=[
        'Lag_1D_Units_Sold', 'Lag_7D_Units_Sold',
        'Rolling_3D_Units_Sold', 'Rolling_7D_Units_Sold', 'Rolling_14D_Units_Sold'
    ])
    ts_final.to_csv("C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/timeseries_data.csv", index=False)
    print("✅ Time series data with lag and rolling features saved.")

# === Optional direct call ===
if __name__ == "__main__":
    run_pipeline()
