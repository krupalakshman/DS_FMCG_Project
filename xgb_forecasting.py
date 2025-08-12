import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO)

# === CONFIG ===
DATA_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/timeseries_data.csv"
PLOT_DIR = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_plots"
METRICS_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_metrics.csv"
FORECAST_MASTER_PATH = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_90day_forecast_master.csv"
os.makedirs(PLOT_DIR, exist_ok=True)

# === FEATURES ===
FEATURES = [
    'Price', 'On_Promotion', 'Budget', 'Channel', 'Opening_Stock', 'Returns',
    'Damaged', 'Closing_Stock', 'Stock_In', 'Stock_Out', 'Damaged_Warehouse',
    'Returned', 'Closing_Stock_Warehouse', 'Holiday', 'Weather', 'Competitor_Price',
    'Complaints', 'Satisfaction', 'Monthly_Sales',
    'Lag_1D_Units_Sold', 'Lag_7D_Units_Sold', 'Rolling_3D_Units_Sold',
    'Rolling_7D_Units_Sold', 'Rolling_14D_Units_Sold'
]
TARGET = 'Units_Sold'


# === FUNCTIONS ===
def preprocess_data(path):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["On_Promotion"] = df["On_Promotion"].astype(int)
    df["Holiday"] = df["Holiday"].apply(lambda x: 0 if x == "No Holiday" else 1)
    df["Weather"] = LabelEncoder().fit_transform(df["Weather"])
    df["Channel"] = LabelEncoder().fit_transform(df["Channel"])
    return df


def train_xgb_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4)
    }


def generate_forecast(model, last_row, features, region, sku):
    future_dates = pd.date_range(start=last_row["Date"] + timedelta(days=1), periods=90)
    preds = []

    for fdate in future_dates:
        row_input = last_row[features].copy()
        row_input["Holiday"] = 0  # assume no holiday
        pred = max(0, model.predict(pd.DataFrame([row_input]))[0])
        preds.append({
            "Date": fdate, "Region": region, "SKU": sku,
            "Predicted_Units_Sold": round(pred)
        })
        last_row["Units_Sold"] = pred  # roll forward

    return pd.DataFrame(preds)


def plot_predictions(dates, actual, predicted, title, path):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.title(title)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_forecast(forecast_df, title, path):
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_df["Date"], forecast_df["Predicted_Units_Sold"], label="Forecast (Next 90D)")
    plt.title(title)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(path)
    plt.close()


def forecast_all(df):
    combinations = df[["Region", "SKU"]].drop_duplicates()
    metrics = []
    
    for _, row in combinations.iterrows():
        region, sku = row["Region"], row["SKU"]
        df_filtered = df[(df["Region"] == region) & (df["SKU"] == sku)].copy()

        if df_filtered.shape[0] < 150:
            continue

        df_filtered.dropna(subset=FEATURES + [TARGET], inplace=True)
        df_filtered.sort_values("Date", inplace=True)

        X, y, dates = df_filtered[FEATURES], df_filtered[TARGET], df_filtered["Date"]
        X_train, X_test = X[:-90], X[-90:]
        y_train, y_test, test_dates = y[:-90], y[-90:], dates[-90:]

        model = train_xgb_model(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics.append({
            "Region": region,
            "SKU": sku,
            **evaluate_model(y_test, y_pred)
        })

        plot_predictions(
            test_dates, y_test.values, y_pred,
            title=f"XGBoost Forecast: {sku} @ {region}",
            path=f"{PLOT_DIR}/{sku}_{region}.png"
        )

        last_row = df_filtered.iloc[-1].copy()
        forecast_df = generate_forecast(model, last_row, FEATURES, region, sku)
        forecast_df.to_csv(f"{PLOT_DIR}/{sku}_{region}_future_forecast.csv", index=False)

        plot_forecast(
            forecast_df,
            title=f"Future Forecast: {sku} @ {region} (Next 90 Days)",
            path=f"{PLOT_DIR}/{sku}_{region}_future_forecast_plot.png"
        )

    return metrics


def combine_forecasts_and_save():
    files = glob.glob(f"{PLOT_DIR}/*_future_forecast.csv")
    all_forecasts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    all_forecasts.to_csv(FORECAST_MASTER_PATH, index=False)


# === MAIN ===
def main():
    logging.info("Starting XGBoost All-SKU Forecasting Pipeline...")
    df = preprocess_data(DATA_PATH)
    metrics = forecast_all(df)
    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    combine_forecasts_and_save()
    logging.info("Forecasting complete. Outputs saved.")

if __name__ == "__main__":
    main()
