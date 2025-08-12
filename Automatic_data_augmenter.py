import pandas as pd
from datetime import datetime, timedelta
from data_generator import (
    generate_sales_data_with_campaigns,
    generate_inventory_data,
    generate_warehouse_data,
    generate_crm_data,
    generate_external_data
)

# === CONFIGURATION ===
PRODUCTS_CSV = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/raw/balaji_products.csv"
DATA_DIR = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/synthetic"
SALES_CSV = f"{DATA_DIR}/synthetic_sales.csv"
CAMPAIGN_CSV = f"{DATA_DIR}/synthetic_campaign.csv"
INVENTORY_CSV = f"{DATA_DIR}/synthetic_inventory.csv"
WAREHOUSE_CSV = f"{DATA_DIR}/synthetic_warehouse.csv"
CRM_CSV = f"{DATA_DIR}/synthetic_crm.csv"
EXTERNAL_CSV = f"{DATA_DIR}/synthetic_external.csv"


# === Load latest sales date ===
def get_next_date_to_generate(sales_csv_path):
    sales_df = pd.read_csv(sales_csv_path, parse_dates=["Date"])
    last_date = sales_df["Date"].max().date()
    next_date = last_date + timedelta(days=1)
    return next_date


# === Generate all daily datasets ===
def generate_daily_data(date, products_csv, stores_per_region=5, num_campaigns=0):
    sales_df, campaign_df = generate_sales_data_with_campaigns(
        products_csv_path=products_csv,
        start_date=date,
        end_date=date,
        stores_per_region=stores_per_region,
        num_campaigns=num_campaigns
    )
    inventory_df = generate_inventory_data(sales_df)
    warehouse_df = generate_warehouse_data(inventory_df)
    crm_df = generate_crm_data(sales_df)
    external_df = generate_external_data(sales_df)

    return sales_df, campaign_df, inventory_df, warehouse_df, crm_df, external_df


# === Append or overwrite generated data ===
def update_datasets(sales, campaign, inventory, warehouse, crm, external):
    sales.to_csv(SALES_CSV, mode='a', header=False, index=False)
    inventory.to_csv(INVENTORY_CSV, mode='a', header=False, index=False)
    warehouse.to_csv(WAREHOUSE_CSV, mode='a', header=False, index=False)
    external.to_csv(EXTERNAL_CSV, mode='a', header=False, index=False)

    # Replace daily CRM summary
    crm.to_csv(CRM_CSV, index=False)

    # Optional: Save campaigns if needed
    # campaign.to_csv(CAMPAIGN_CSV, mode='a', header=False, index=False)


# === Main driver ===
def main():
    next_date = get_next_date_to_generate(SALES_CSV)
    today = datetime.today().date()

    if next_date > today:
        print("✅ Dataset is already up to date.")
        return

    print(f"📅 Generating data for {next_date}")
    sales, campaign, inventory, warehouse, crm, external = generate_daily_data(
        date=next_date,
        products_csv=PRODUCTS_CSV
    )
    update_datasets(sales, campaign, inventory, warehouse, crm, external)
    print(f"✅ Augmented data for {next_date}")


if __name__ == "__main__":
    main()
