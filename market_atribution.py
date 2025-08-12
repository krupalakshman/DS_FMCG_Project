# marketing_attribution.py

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report
)

def run_marketing_attribution_model(csv_path):
    # === Load Data ===
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    # === Basic Cleaning ===
    df['On_Promotion'] = df['On_Promotion'].fillna(0)
    df['Budget'] = df['Budget'].fillna(0)
    df['Channel'] = df['Channel'].fillna('Unknown')
    df['Revenue'] = df['Revenue'].fillna(0)

    # === One-Hot Encoding ===
    cat_cols = ['Channel', 'Region', 'Category']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    feature_cols = ['On_Promotion', 'Budget'] + \
                   [col for col in df.columns if col.startswith('Channel_') or col.startswith('Region_') or col.startswith('Category_')]

    # === Regression: Predict Revenue ===
    X_reg = df[feature_cols]
    y_reg = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    print("📊 XGBoost Regressor Results")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)
    plt.title("XGBoost Feature Attribution (Revenue)")
    plt.tight_layout()
    plt.show()

    # === Classification: High Revenue or Not ===
    df['High_Revenue'] = (df['Revenue'] > df['Revenue'].median()).astype(int)
    X_cls = df[feature_cols]
    y_cls = df['High_Revenue']
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_cls, y_train_cls)
    y_pred_cls = logreg.predict(X_test_cls)

    print("📊 Logistic Regression Results")
    print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
    print(classification_report(y_test_cls, y_pred_cls))

    feature_importance = pd.Series(logreg.coef_[0], index=X_train_cls.columns).sort_values(ascending=False)
    feature_importance.head(10).plot(kind='barh', title="Logistic Regression Feature Attribution")
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    run_marketing_attribution_model(
        csv_path="C:/Users/dakoj/OneDrive/Desktop/Workoopolis/FMCG_Project/data/processed/merged_dataset.csv"
    )
