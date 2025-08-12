import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# === Load Forecast Data ===
@st.cache_data
def load_forecast_data():
    xgb_df = pd.read_csv("C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_90day_forecast_master.csv", parse_dates=["Date"])
    prophet_df = pd.read_csv("C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_forecasts_master.csv", parse_dates=["Date"])
    return xgb_df, prophet_df

xgb_df, prophet_df = load_forecast_data()

# === Sidebar Filters ===
st.sidebar.title("🔍 Filter Forecasts")
view_type = st.sidebar.selectbox("Select Model", ["XGBoost", "Prophet"])
df = xgb_df if view_type == "XGBoost" else prophet_df

regions = sorted(df["Region"].unique())
region = st.sidebar.selectbox("Select Region", regions)

skus = sorted(df[df["Region"] == region]["SKU"].unique())
sku = st.sidebar.selectbox("Select SKU", skus)

# === Filtered Data ===
filtered_df = df[(df["Region"] == region) & (df["SKU"] == sku)]

st.title("📦 FMCG Demand Forecast Dashboard")
st.markdown(f"### Forecast for **{sku}** in **{region}** using **{view_type}**")

# === Plot Forecast ===
fig = px.line(
    filtered_df,
    x="Date",
    y="Forecast" if view_type == "Prophet" else "Predicted_Units_Sold",
    title=f"{sku} - {region} Forecast ({view_type})",
    labels={"Forecast": "Units Sold", "Predicted_Units_Sold": "Units Sold"},
    markers=True
)

# Add confidence interval for Prophet
if view_type == "Prophet":
    fig.add_traces([
        px.line(filtered_df, x="Date", y="Lower_Bound").data[0],
        px.line(filtered_df, x="Date", y="Upper_Bound").data[0]
    ])
    fig.update_traces(line=dict(dash='dot'), selector=dict(name="Lower_Bound"))
    fig.update_traces(line=dict(dash='dot'), selector=dict(name="Upper_Bound"))

fig.update_layout(legend_title_text="Forecast")
st.plotly_chart(fig, use_container_width=True)

# === Metrics Table (Optional) ===
if view_type == "XGBoost":
    metrics_path = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/xgb_allsku_metrics.csv"
else:
    metrics_path = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs/prophet_evaluation_metrics.csv"

metrics_df = pd.read_csv(metrics_path)
metric_row = metrics_df[(metrics_df["Region"] == region) & (metrics_df["SKU"] == sku)]

if not metric_row.empty:
    st.subheader("📊 Evaluation Metrics")
    st.dataframe(metric_row.reset_index(drop=True), use_container_width=True)
