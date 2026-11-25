import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import onnxmltools
import kagglehub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "data"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "feature_contract.json"
ARTIFACTS_DIR = ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    canonical = json.dumps(contract["features"], sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return contract, schema_hash


def make_synthetic_logistics_data(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    distance_km = rng.gamma(shape=2.2, scale=250.0, size=n).astype(np.float32)
    weight_grams = rng.lognormal(mean=7.0, sigma=1.1, size=n).astype(np.float32)
    seller_late_rate = rng.beta(a=1.5, b=12.0, size=n).astype(np.float32)
    is_weekend = rng.integers(0, 2, size=n).astype(np.float32)
    freight_ratio = rng.uniform(0.05, 0.65, size=n).astype(np.float32)

    # Inject realistic missing values (~8%) to test null <-> NaN parity
    weight_grams[rng.random(n) < 0.08] = np.nan
    seller_late_rate[rng.random(n) < 0.08] = np.nan

    # True latent SLA breach probability (~12% late deliveries)
    logit = (
        -3.4
        + 0.0022 * distance_km
        + 4.5 * np.nan_to_num(seller_late_rate, nan=0.18)
        + 0.45 * is_weekend
        + 1.2 * freight_ratio
        + 0.00008 * np.nan_to_num(weight_grams, nan=1500.0)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    is_late = (rng.random(n) < prob).astype(np.int32)

    return pd.DataFrame(
        {
            "distance_km": distance_km,
            "weight_grams": weight_grams,
            "seller_late_rate": seller_late_rate,
            "is_weekend": is_weekend,
            "freight_ratio": freight_ratio,
            "is_late": is_late,
        }
    )


def resolve_olist_dir() -> Path | None:
    if (DATA_DIR / "olist_orders_dataset.csv").exists():
        return DATA_DIR
    try:
        return Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
    except Exception as e:
        print(f"[warn] kagglehub download skipped ({e}); falling back to synthetic data")
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return (2 * r * np.arcsin(np.sqrt(a))).astype(np.float32)


def load_olist_dataset(olist_dir: Path) -> pd.DataFrame:
    orders = pd.read_csv(
        olist_dir / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"],
    )
    items = pd.read_csv(olist_dir / "olist_order_items_dataset.csv")
    products = pd.read_csv(olist_dir / "olist_products_dataset.csv")[["product_id", "product_weight_g"]]
    customers = pd.read_csv(olist_dir / "olist_customers_dataset.csv")[["customer_id", "customer_zip_code_prefix"]]
    sellers = pd.read_csv(olist_dir / "olist_sellers_dataset.csv")[["seller_id", "seller_zip_code_prefix"]]
    geo = (
        pd.read_csv(olist_dir / "olist_geolocation_dataset.csv")
        .groupby("geolocation_zip_code_prefix", as_index=False)[["geolocation_lat", "geolocation_lng"]]
        .mean()
    )

    # Keep delivered orders with valid timestamps
    orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()
    orders["is_late"] = (orders["order_delivered_customer_date"] > orders["order_estimated_delivery_date"]).astype(np.int32)
    orders["is_weekend"] = (orders["order_purchase_timestamp"].dt.day_of_week >= 5).astype(np.float32)

    # Aggregate order items to order grain (1 row per order)
    items = items.merge(products, on="product_id", how="left")
    order_agg = items.groupby("order_id", as_index=False).agg(
        seller_id=("seller_id", "first"),
        weight_grams=("product_weight_g", "sum"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
    )
    order_agg["freight_ratio"] = (
        order_agg["total_freight"] / (order_agg["total_price"] + order_agg["total_freight"] + 1e-6)
    ).astype(np.float32)

    df = (
        orders[["order_id", "customer_id", "is_weekend", "is_late"]]
        .merge(order_agg, on="order_id", how="inner")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix", "geolocation_lat": "c_lat", "geolocation_lng": "c_lng"}), on="customer_zip_code_prefix", how="left")
        .merge(geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix", "geolocation_lat": "s_lat", "geolocation_lng": "s_lng"}), on="seller_zip_code_prefix", how="left")
    )

    df["distance_km"] = haversine_km(df["s_lat"], df["s_lng"], df["c_lat"], df["c_lng"])
    # Drop rows missing non-nullable contract features
    df = df.dropna(subset=["distance_km", "freight_ratio", "is_weekend"]).reset_index(drop=True)
    return df[["seller_id", "distance_km", "weight_grams", "is_weekend", "freight_ratio", "is_late"]]


def evaluate(name: str, y_true, probs, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)

    # Exact peak F1 and its decision threshold
    prec_curve, rec_curve, ths = precision_recall_curve(y_true, probs)
    f1_curve = 2 * prec_curve[:-1] * rec_curve[:-1] / (prec_curve[:-1] + rec_curve[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_curve))

    return {
        "name": name,
        "pr_auc": average_precision_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "peak_f1": float(f1_curve[best_idx]),
        "peak_th": float(ths[best_idx]),
    }


def main():
    contract, schema_hash = load_contract()
    feature_cols = [f["name"] for f in contract["features"]]
    threshold = contract["threshold"]

    olist_dir = resolve_olist_dir()
    if olist_dir is not None:
        print(f"Loading real Olist dataset from {olist_dir}")
        df = load_olist_dataset(olist_dir)
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["is_late"], random_state=42
        )

        # Leak-free target encoding: compute seller_late_rate strictly on train_df
        seller_stats = train_df.groupby("seller_id")["is_late"].agg(["mean", "count"])
        valid_sellers = seller_stats[seller_stats["count"] >= 5]["mean"]

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df["seller_late_rate"] = train_df["seller_id"].map(valid_sellers).astype(np.float32)
        test_df["seller_late_rate"] = test_df["seller_id"].map(valid_sellers).astype(np.float32)

        X_train, y_train = train_df[feature_cols].astype(np.float32), train_df["is_late"].values
        X_test, y_test = test_df[feature_cols].astype(np.float32), test_df["is_late"].values
    else:
        print("Using synthetic logistics dataset")
        df = make_synthetic_logistics_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_cols].astype(np.float32),
            df["is_late"].values,
            test_size=0.2,
            stratify=df["is_late"].values,
            random_state=42,
        )

    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} | Late rate: {y_train.mean():.2%}\n")

    # 1. Baseline: Logistic Regression (requires manual NaN fill)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_train.fillna(0.0), y_train)

    # 2. Random Forest (parallel bagging; cap depth/leaf size to prevent overfitting on 76k rows)
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)

    # 3. CatBoost (symmetric oblivious trees, native NaN handling)
    cb_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.08, random_seed=42, verbose=0
    )
    cb_model.fit(X_train, y_train)

    # 4. XGBoost (native NaN handling)
    xgb_model = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    xgb_model.fit(X_train, y_train)

    # 5. LightGBM (native NaN handling — exported to ONNX)
    lgb_model = LGBMClassifier(
        n_estimators=120, num_leaves=31, learning_rate=0.08, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    # Collect, sort descending by F1 at contract threshold, and print leaderboard
    results = [
        evaluate("LogisticRegression", y_test, lr.predict_proba(X_test.fillna(0.0))[:, 1], threshold),
        evaluate("RandomForest", y_test, rf_model.predict_proba(X_test)[:, 1], threshold),
        evaluate("CatBoost", y_test, cb_model.predict_proba(X_test)[:, 1], threshold),
        evaluate("XGBoost", y_test, xgb_model.predict_proba(X_test)[:, 1], threshold),
        evaluate("LightGBM", y_test, lgb_model.predict_proba(X_test)[:, 1], threshold),
    ]
    results.sort(key=lambda r: r["f1"], reverse=True)

    print(f"{'Model':18s} | {'PR-AUC':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'F1@' + str(threshold):>7s} | {'Peak F1 (@th)':>15s}")
    print("-" * 74)
    for r in results:
        print(
            f"{r['name']:18s} | {r['pr_auc']:6.4f} | {r['precision']:6.4f} | {r['recall']:6.4f} "
            f"| {r['f1']:7.4f} | {r['peak_f1']:.4f} (@{r['peak_th']:.3f})"
        )

    # Export LightGBM -> ONNX with flat probability tensor (zipmap=False)
    initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        lgb_model,
        initial_types=initial_type,
        target_opset=15,
        zipmap=False,
    )
    model_path = ARTIFACTS_DIR / "model.onnx"
    onnxmltools.utils.save_model(onnx_model, str(model_path))

    # Emit 100 golden fixtures for 1:1 Go <-> Python parity testing
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    sample_df = X_test.iloc[:100].reset_index(drop=True)

    fixtures = []
    for _, row in sample_df.iterrows():
        raw_input = {
            col: (None if pd.isna(row[col]) else round(float(row[col]), 6))
            for col in feature_cols
        }
        vec = np.array(
            [[np.nan if raw_input[c] is None else raw_input[c] for c in feature_cols]],
            dtype=np.float32,
        )
        _, row_probs = sess.run(None, {"input": vec})
        late_prob = float(row_probs[0][1])
        fixtures.append(
            {
                "input": raw_input,
                "expected_prob": late_prob,
                "expected_label": int(late_prob >= threshold),
            }
        )

    out = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "threshold": threshold,
        "cases": fixtures,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(out, indent=2))
    print(f"\nExported {model_path} and {len(fixtures)} golden fixtures (schema_hash={schema_hash})")

if __name__ == "__main__":
    main()
