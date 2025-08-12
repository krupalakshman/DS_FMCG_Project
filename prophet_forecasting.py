# forecast_prophet_allsku.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# === Paths ===
DATA_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/timeseries_data.csv"
PLOT_DIR = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_plots"
FORECAST_DIR = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_forecasts"
EVAL_METRICS_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_evaluation_metrics.csv"
MASTER_FORECAST_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_forecasts_master.csv"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)


def preprocess(df: pd.DataFrame, region: str, sku: str) -> pd.DataFrame:
    """Prepare data for Prophet with regressors."""
    df_filtered = df[(df["Region"] == region) & (df["SKU"] == sku)].copy()

    if len(df_filtered) < 180:
        return pd.DataFrame()

    df_prophet = df_filtered.rename(columns={"Date": "ds", "Units_Sold": "y"})
    df_prophet = df_prophet[["ds", "y", "Price", "On_Promotion", "Holiday", "Weather", "Competitor_Price"]]

    df_prophet["On_Promotion"] = df_prophet["On_Promotion"].astype(int)
    df_prophet["Holiday"] = df_prophet["Holiday"].apply(lambda x: 1 if x != "No Holiday" else 0)
    df_prophet = pd.get_dummies(df_prophet, columns=["Weather"])
    df_prophet[["Price", "Competitor_Price"]] = StandardScaler().fit_transform(df_prophet[["Price", "Competitor_Price"]])
    df_prophet.dropna(inplace=True)

    return df_prophet if len(df_prophet) >= 180 else pd.DataFrame()


def build_prophet_model(df_train: pd.DataFrame) -> Prophet:
    """Configure and fit Prophet model with regressors."""
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    for col in df_train.columns:
        if col not in ["ds", "y"]:
            model.add_regressor(col)

    model.fit(df_train)
    return model


def forecast_and_plot(model: Prophet, df_all: pd.DataFrame, test: pd.DataFrame,
                      region: str, sku: str) -> Dict:
    """Generate forecasts and evaluation metrics, save plots."""
    forecast = model.predict(df_all.drop(columns="y"))
    forecast_test = forecast.set_index("ds").loc[test["ds"]].reset_index()
    y_true = test.set_index("ds").loc[forecast_test["ds"], "y"].values
    y_pred = forecast_test["yhat"].values

    fig = model.plot(forecast)
    plt.title(f"{sku} @ {region}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{sku}_{region}.png")
    plt.close()

    return {
        "Region": region,
        "SKU": sku,
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4)
    }


def generate_future_forecast(model: Prophet, df_train: pd.DataFrame, region: str, sku: str):
    """Forecast next 90 days using recent avg values of regressors."""
    last_date = df_train["ds"].max()
    recent_avg = df_train.drop(columns=["ds", "y"]).tail(30).mean()
    future = pd.DataFrame({"ds": pd.date_range(last_date + pd.Timedelta(days=1), periods=90)})

    for col in recent_avg.index:
        future[col] = recent_avg[col]

    forecast_90 = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_90.columns = ["Date", "Forecast", "Lower_Bound", "Upper_Bound"]
    forecast_90["SKU"] = sku
    forecast_90["Region"] = region

    forecast_90.to_csv(f"{FORECAST_DIR}/{sku}_{region}_future_90.csv", index=False)


def consolidate_forecasts(forecast_dir: str, output_path: str):
    """Merge all forecast files into one master CSV."""
    all_forecasts = []

    for file in os.listdir(forecast_dir):
        if file.endswith("_future_90.csv"):
            sku, region = file.replace("_future_90.csv", "").split("_", 1)
            df_fc = pd.read_csv(os.path.join(forecast_dir, file), parse_dates=["Date"])
            df_fc["SKU"], df_fc["Region"] = sku, region
            all_forecasts.append(df_fc)

    if all_forecasts:
        pd.concat(all_forecasts, ignore_index=True).to_csv(output_path, index=False)


def main():
    logging.info("🚀 Loading data...")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    combinations = df[["Region", "SKU"]].drop_duplicates()
    metrics: List[Dict] = []

    for _, row in combinations.iterrows():
        region, sku = row["Region"], row["SKU"]
        logging.info(f"🔍 Processing: {sku} @ {region}")

        df_prophet = preprocess(df, region, sku)
        if df_prophet.empty:
            logging.warning(f"⏭️ Skipped: {sku} @ {region} (insufficient data)")
            continue

        train, test = df_prophet[:-90], df_prophet[-90:]

        try:
            model = build_prophet_model(train)
        except Exception as e:
            logging.error(f"❌ Failed to train model for {sku} @ {region}: {e}")
            continue

        metrics.append(forecast_and_plot(model, df_prophet, test, region, sku))
        generate_future_forecast(model, train, region, sku)

    pd.DataFrame(metrics).to_csv(EVAL_METRICS_PATH, index=False)
    consolidate_forecasts(FORECAST_DIR, MASTER_FORECAST_PATH)
    logging.info("✅ All SKU-region forecasts complete.")


if __name__ == "__main__":
    main()
