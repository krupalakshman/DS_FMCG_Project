import pandas as pd
import numpy as np

# === CONFIGURABLE CONSTANTS ===
LEAD_TIME = 3
SERVICE_LEVEL_Z = 1.645  # 95% service level
HOLDING_COST_PER_UNIT = 0.5
STOCKOUT_COST_PER_UNIT = 2.0
ORDER_COST = 50
DEFAULT_INITIAL_STOCK = 500
ROLLING_STD_WINDOW = 14
REORDER_QTY_WINDOW = 7

MERGED_DATA_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/merged_dataset.csv"
FORECAST_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_90day_forecast_master.csv"
OUTPUT_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/inventory_simulation_powerbi.csv"


def load_data(merged_path, forecast_path):
    merged_df = pd.read_csv(merged_path, parse_dates=["Date"])
    forecast_df = pd.read_csv(forecast_path, parse_dates=["Date"])
    return merged_df, forecast_df


def prepare_forecast(merged_df, forecast_df):
    latest_stock_df = (
        merged_df.sort_values("Date")
        .groupby(["SKU", "Region"])
        .last()
        .reset_index()[["SKU", "Region", "Closing_Stock"]]
        .rename(columns={"Closing_Stock": "Initial_Stock"})
    )

    forecast_df = forecast_df.sort_values(["SKU", "Region", "Date"])
    forecast_df = forecast_df.merge(latest_stock_df, on=["SKU", "Region"], how="left")

    forecast_df["Rolling_STD"] = forecast_df.groupby(["SKU", "Region"])["Predicted_Units_Sold"].transform(
        lambda x: x.rolling(ROLLING_STD_WINDOW, min_periods=1).std()
    )

    forecast_df["Safety_Stock"] = SERVICE_LEVEL_Z * forecast_df["Rolling_STD"] * np.sqrt(LEAD_TIME)

    forecast_df["LeadTime_Demand"] = forecast_df.groupby(["SKU", "Region"])["Predicted_Units_Sold"].transform(
        lambda x: x.shift(1).rolling(LEAD_TIME).sum()
    )

    forecast_df["ROP"] = forecast_df["LeadTime_Demand"] + forecast_df["Safety_Stock"]

    forecast_df["Reorder_Qty"] = forecast_df.groupby(["SKU", "Region"])["Predicted_Units_Sold"].transform(
        lambda x: x.rolling(REORDER_QTY_WINDOW, min_periods=1).mean()
    ).round()

    return forecast_df


def simulate_inventory(forecast_df):
    results = []

    for (sku, region), group in forecast_df.groupby(["SKU", "Region"]):
        stock = group["Initial_Stock"].iloc[0]
        stock = DEFAULT_INITIAL_STOCK if pd.isna(stock) else stock

        for _, row in group.iterrows():
            demand = row["Predicted_Units_Sold"]
            rop = row["ROP"]
            reorder_qty = row["Reorder_Qty"]

            order_placed = int(stock <= rop)
            if order_placed:
                stock += reorder_qty

            stock_after = max(stock - demand, 0)

            results.append({
                "Date": row["Date"],
                "SKU": sku,
                "Region": region,
                "Predicted_Units_Sold": demand,
                "Stock_Before": stock,
                "ROP": rop,
                "Reorder_Qty": reorder_qty,
                "Stock_After": stock_after,
                "Stockout": max(demand - stock, 0),
                "Holding_Cost": stock_after * HOLDING_COST_PER_UNIT,
                "Stockout_Cost": max(demand - stock, 0) * STOCKOUT_COST_PER_UNIT,
                "Order_Placed": order_placed
            })

            stock = stock_after  # Update stock for next day

    return pd.DataFrame(results)


def summarize_costs(sim_df):
    total_holding_cost = sim_df["Holding_Cost"].sum()
    total_stockout_cost = sim_df["Stockout_Cost"].sum()
    total_orders = sim_df["Order_Placed"].sum()
    total_order_cost = total_orders * ORDER_COST
    total_inventory_cost = total_holding_cost + total_stockout_cost + total_order_cost

    print("\n📊 Inventory Cost Summary")
    print(f"📦 Holding Cost     : ₹{total_holding_cost:,.2f}")
    print(f"❌ Stockout Cost    : ₹{total_stockout_cost:,.2f}")
    print(f"🚚 Order Cost       : ₹{total_order_cost:,.2f} ({total_orders} orders)")
    print(f"💰 Total Inventory Cost: ₹{total_inventory_cost:,.2f}")


def export_results(sim_df, path):
    sim_df.to_csv(path, index=False)
    print(f"\n✅ Exported simulation results to: {path}")


def main():
    print("🚀 Starting Inventory Simulation...")
    merged_df, forecast_df = load_data(MERGED_DATA_PATH, FORECAST_PATH)
    forecast_df = prepare_forecast(merged_df, forecast_df)
    sim_df = simulate_inventory(forecast_df)
    summarize_costs(sim_df)
    export_results(sim_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
