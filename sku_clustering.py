import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# === Setup logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_data(filepath):
    logging.info("Loading data...")
    return pd.read_csv(filepath, parse_dates=["Date"])

def aggregate_features(df):
    logging.info("Aggregating features at SKU level...")
    sku_features = df.groupby("SKU").agg({
        "Units_Sold": ["mean", "std"],
        "Revenue": "mean",
        "Returns": "mean",
        "Damaged": "mean",
        "Price": "mean",
        "On_Promotion": "mean"
    })
    sku_features.columns = [
        'Units_Sold_Mean', 'Units_Sold_STD', 'Revenue_Mean',
        'Returns_Mean', 'Damaged_Mean', 'Price_Mean', 'Promo_Rate'
    ]
    sku_features = sku_features.reset_index()
    sku_features.fillna(0, inplace=True)
    return sku_features

def scale_features(df):
    logging.info("Scaling features...")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.drop("SKU", axis=1))
    return scaled

def elbow_method(X_scaled, output_dir):
    logging.info("Running Elbow method...")
    wcss = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    
    plt.figure()
    plt.plot(range(2, 11), wcss, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for K')
    plt.grid()
    elbow_plot_path = os.path.join(output_dir, "elbow_plot.png")
    plt.savefig(elbow_plot_path)
    plt.close()
    logging.info(f"Elbow plot saved to {elbow_plot_path}")

def apply_clustering(X_scaled, df, output_dir, n_clusters=4):
    logging.info("Applying KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, df["KMeans_Cluster"])
    logging.info(f"KMeans Silhouette Score: {score:.4f}")

    dbscan = DBSCAN(eps=1.5, min_samples=3)
    df["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)
    
    if len(set(dbscan.labels_)) > 1:
        db_score = silhouette_score(X_scaled, dbscan.labels_)
        logging.info(f"DBSCAN Silhouette Score: {db_score:.4f}")
    else:
        logging.info("DBSCAN found only one cluster or noise.")
    
    return df

def apply_pca_and_plot(X_scaled, df, output_dir):
    logging.info("Applying PCA and generating cluster plots...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]

    # KMeans Cluster Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="KMeans_Cluster", palette="tab10")
    plt.title("KMeans SKU Clusters (PCA)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "kmeans_pca_clusters.png"))
    plt.close()

    # DBSCAN Cluster Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="DBSCAN_Cluster", palette="tab10")
    plt.title("DBSCAN SKU Clusters (PCA)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dbscan_pca_clusters.png"))
    plt.close()

def generate_heatmap(df, output_dir):
    logging.info("Generating cluster heatmap...")
    features = [
        "Units_Sold_Mean", "Units_Sold_STD", "Revenue_Mean",
        "Returns_Mean", "Damaged_Mean", "Price_Mean", "Promo_Rate"
    ]
    cluster_summary = df.groupby("KMeans_Cluster")[features].mean()

    scaler = StandardScaler()
    cluster_summary_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_summary),
        columns=cluster_summary.columns,
        index=cluster_summary.index
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_summary_scaled, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("SKU Cluster Heatmap (KMeans Clusters)")
    plt.ylabel("Cluster")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "cluster_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    logging.info(f"Heatmap saved to {heatmap_path}")

def main():
    input_path = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/merged_dataset.csv"
    output_dir = "C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/outputs"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(input_path)
    sku_df = aggregate_features(df)
    X_scaled = scale_features(sku_df)

    elbow_method(X_scaled, output_dir)
    sku_df = apply_clustering(X_scaled, sku_df, output_dir)
    apply_pca_and_plot(X_scaled, sku_df, output_dir)
    generate_heatmap(sku_df, output_dir)

    output_csv = os.path.join(output_dir, "clustered_sku_features.csv")
    sku_df.to_csv(output_csv, index=False)
    logging.info(f"Clustered features saved to {output_csv}")

if __name__ == "__main__":
    main()
