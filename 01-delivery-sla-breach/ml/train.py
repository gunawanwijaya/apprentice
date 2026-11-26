import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort
import onnxmltools
import kagglehub
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

    delivery_days = np.clip(
        2.5 + 0.008 * distance_km + 6.0 * np.nan_to_num(seller_late_rate, nan=0.18) + rng.normal(0, 1.5, size=n),
        1.0,
        60.0,
    ).astype(np.float32)

    return pd.DataFrame({
        "distance_km": distance_km,
        "weight_grams": weight_grams,
        "seller_late_rate": seller_late_rate,
        "is_weekend": is_weekend,
        "freight_ratio": freight_ratio,
        "is_late": is_late,
        "delivery_days": delivery_days,
    })


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
    orders["delivery_days"] = (
        (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"])
        .dt.total_seconds()
        .div(86400.0)
        .clip(lower=1.0, upper=60.0)
        .astype(np.float32)
    )

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
        orders[["order_id", "customer_id", "is_weekend", "is_late", "delivery_days"]]
        .merge(order_agg, on="order_id", how="inner")
        .merge(customers, on="customer_id", how="left")
        .merge(sellers, on="seller_id", how="left")
        .merge(geo.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix", "geolocation_lat": "c_lat", "geolocation_lng": "c_lng"}), on="customer_zip_code_prefix", how="left")
        .merge(geo.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix", "geolocation_lat": "s_lat", "geolocation_lng": "s_lng"}), on="seller_zip_code_prefix", how="left")
    )

    df["distance_km"] = haversine_km(df["s_lat"], df["s_lng"], df["c_lat"], df["c_lng"])
    # Drop rows missing non-nullable contract features
    df = df.dropna(subset=["distance_km", "freight_ratio", "is_weekend"]).reset_index(drop=True)

    return df[["seller_id", "distance_km", "weight_grams", "is_weekend", "freight_ratio", "is_late", "delivery_days"]]


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


def evaluate_regression(name: str, y_true, preds) -> dict:
    return {
        "name": name,
        "mae": float(mean_absolute_error(y_true, preds)),
        "rmse": float(root_mean_squared_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
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
    else:
        print("Using synthetic logistics dataset")
        df = make_synthetic_logistics_data()
        train_df, test_df = train_test_split(
            df, test_size=0.2, stratify=df["is_late"], random_state=42
        )

    X_train = train_df[feature_cols].astype(np.float32)
    X_test = test_df[feature_cols].astype(np.float32)
    y_train = train_df["is_late"].values
    y_test = test_df["is_late"].values
    y_reg_train = train_df["delivery_days"].values
    y_reg_test = test_df["delivery_days"].values

    print(
        f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,} "
        f"| Late rate: {y_train.mean():.2%} | Mean ETA: {y_reg_train.mean():.2f}d\n"
    )

    # --- 1. Classification Benchmark (is_late) ---
    clf_candidates = [
        ("LogisticRegression", LogisticRegression(max_iter=500), True),
        ("RandomForest", RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42), False),
        ("CatBoost", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=42, verbose=0, allow_writing_files=False), False),
        ("XGBoost", XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42), False),
        ("LightGBM", LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.08, random_state=42, verbose=-1), False),
    ]

    clf_results = []
    trained_clfs = {}
    for name, model, needs_fill in clf_candidates:
        model.fit(X_train.fillna(0.0) if needs_fill else X_train, y_train)
        probs = model.predict_proba(X_test.fillna(0.0) if needs_fill else X_test)[:, 1]
        trained_clfs[name] = model
        clf_results.append(evaluate(name, y_test, probs, threshold))

    clf_results.sort(key=lambda r: r["f1"], reverse=True)

    print(f"{'Model':18s} | {'PR-AUC':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'F1@' + str(threshold):>7s} | {'Peak F1 (@th)':>15s}")
    print("-" * 74)
    for r in clf_results:
        print(
            f"{r['name']:18s} | {r['pr_auc']:6.4f} | {r['precision']:6.4f} | {r['recall']:6.4f} "
            f"| {r['f1']:7.4f} | {r['peak_f1']:.4f} (@{r['peak_th']:.3f})"
        )

    # --- 2. Regression Benchmark (delivery_days ETA) ---
    reg_candidates = [
        ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), True),
        ("RandomForestReg", RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=42), False),
        ("CatBoostReg", CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08, random_seed=42, verbose=0, allow_writing_files=False), False),
        ("XGBRegressor", XGBRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42), False),
        ("LGBMRegressor", LGBMRegressor(n_estimators=120, num_leaves=31, learning_rate=0.08, random_state=42, verbose=-1), False),
    ]

    reg_results = []
    trained_regs = {}
    for name, model, needs_fill in reg_candidates:
        model.fit(X_train.fillna(0.0) if needs_fill else X_train, y_reg_train)
        preds = model.predict(X_test.fillna(0.0) if needs_fill else X_test)
        trained_regs[name] = model
        reg_results.append(evaluate_regression(name, y_reg_test, preds))

    # Sort ascending by MAE (lower error in days is better)
    reg_results.sort(key=lambda r: r["mae"])

    print(f"\n{'Regressor':18s} | {'MAE (days)':>10s} | {'RMSE (days)':>11s} | {'R2':>7s}")
    print("-" * 56)
    for r in reg_results:
        print(f"{r['name']:18s} | {r['mae']:10.4f} | {r['rmse']:11.4f} | {r['r2']:7.4f}")

    # --- 3. Conformalized Quantile Regression (CQR) Benchmark (p10 / p50 / p90) ---
    coverage_target = float(contract.get("eta_coverage_target", 0.80))
    alpha_low = (1.0 - coverage_target) / 2.0   # 0.10
    alpha_high = 1.0 - alpha_low                # 0.90

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_reg_train, test_size=0.2, random_state=42
    )

    cqr_builders = [
        (
            "LightGBM-CQR",
            lambda q: LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=120,
                num_leaves=31,
                learning_rate=0.08,
                random_state=42,
                verbose=-1,
            ),
        ),
        (
            "XGBoost-CQR",
            lambda q: XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
            ),
        ),
        (
            "CatBoost-CQR",
            lambda q: CatBoostRegressor(
                loss_function=f"Quantile:alpha={q}",
                iterations=300,
                depth=6,
                learning_rate=0.08,
                random_seed=42,
                verbose=0,
                allow_writing_files=False,
            ),
        ),
    ]

    cqr_results = []
    trained_cqr = {}

    for name, build_fn in cqr_builders:
        q_models = {}
        for label, q_alpha in [("p10", alpha_low), ("p50", 0.50), ("p90", alpha_high)]:
            m = build_fn(q_alpha)
            m.fit(X_fit, y_fit)
            q_models[label] = m

        # Calibrate CQR margin Q strictly on held-out calibration split
        cal_lo = q_models["p10"].predict(X_cal)
        cal_hi = q_models["p90"].predict(X_cal)
        conformity_scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
        q_level = min(1.0, coverage_target * (1.0 + 1.0 / len(conformity_scores)))
        cqr_margin = float(np.quantile(conformity_scores, q_level))

        # Audit Raw vs. CQR empirical coverage & sharpness on unseen X_test
        test_lo_raw = q_models["p10"].predict(X_test)
        test_p50_raw = q_models["p50"].predict(X_test)
        test_hi_raw = q_models["p90"].predict(X_test)

        raw_cov = float(((y_reg_test >= test_lo_raw) & (y_reg_test <= test_hi_raw)).mean())
        test_lo_cqr = np.maximum(1.0, test_lo_raw - cqr_margin)
        test_hi_cqr = np.maximum(test_lo_cqr, test_hi_raw + cqr_margin)
        cqr_cov = float(((y_reg_test >= test_lo_cqr) & (y_reg_test <= test_hi_cqr)).mean())
        avg_width = float((test_hi_cqr - test_lo_cqr).mean())
        p50_mae = float(mean_absolute_error(y_reg_test, test_p50_raw))

        trained_cqr[name] = {"models": q_models, "margin": cqr_margin, "raw_cov": raw_cov, "cqr_cov": cqr_cov, "avg_width": avg_width}
        cqr_results.append({
            "name": name,
            "raw_cov": raw_cov,
            "cqr_cov": cqr_cov,
            "q_margin": cqr_margin,
            "avg_width": avg_width,
            "p50_mae": p50_mae,
        })

    # Sort ascending by Avg Width (narrowest interval at ~80% coverage wins)
    cqr_results.sort(key=lambda r: r["avg_width"])

    print(f"\n{'CQR Model':18s} | {'Raw Cov':>7s} | {'CQR Cov':>7s} | {'Q Margin':>8s} | {'Avg Width':>9s} | {'p50 MAE':>7s}")
    print("-" * 70)
    for r in cqr_results:
        print(
            f"{r['name']:18s} | {r['raw_cov']:6.2%} | {r['cqr_cov']:6.2%} "
            f"| {r['q_margin']:+7.3f}d | {r['avg_width']:8.3f}d | {r['p50_mae']:6.3f}d"
        )

    # --- 4. Export LightGBM Classifier + CQR Quantile Models -> ONNX ---
    initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]

    clf_onnx = onnxmltools.convert_lightgbm(
        trained_clfs["LightGBM"],
        initial_types=initial_type,
        target_opset=15,
        zipmap=False,
    )
    model_path = ARTIFACTS_DIR / "model.onnx"
    onnxmltools.utils.save_model(clf_onnx, str(model_path))

    lgb_cqr = trained_cqr["LightGBM-CQR"]
    q_paths = {}
    for label in ("p10", "p50", "p90"):
        q_onnx = onnxmltools.convert_lightgbm(
            lgb_cqr["models"][label],
            initial_types=initial_type,
            target_opset=15,
        )
        p = ARTIFACTS_DIR / f"eta_{label}.onnx"
        onnxmltools.utils.save_model(q_onnx, str(p))
        q_paths[label] = p

    cqr_cal = {
        "model": "LightGBM-CQR",
        "coverage_target": coverage_target,
        "cqr_margin_days": round(lgb_cqr["margin"], 6),
        "raw_test_coverage": round(lgb_cqr["raw_cov"], 4),
        "cqr_test_coverage": round(lgb_cqr["cqr_cov"], 4),
        "avg_interval_width_days": round(lgb_cqr["avg_width"], 3),
    }
    (ARTIFACTS_DIR / "cqr_calibration.json").write_text(json.dumps(cqr_cal, indent=2))

    # --- 5. Emit 100 Multi-Output Golden Fixtures (Classifier + CQR Range) ---
    clf_sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    p10_sess = ort.InferenceSession(str(q_paths["p10"]), providers=["CPUExecutionProvider"])
    p50_sess = ort.InferenceSession(str(q_paths["p50"]), providers=["CPUExecutionProvider"])
    p90_sess = ort.InferenceSession(str(q_paths["p90"]), providers=["CPUExecutionProvider"])
    sample_df = X_test.iloc[:100].reset_index(drop=True)

    q_val = cqr_cal["cqr_margin_days"]
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
        _, row_probs = clf_sess.run(None, {"input": vec})
        late_prob = float(row_probs[0][1])

        raw_p10 = float(p10_sess.run(None, {"input": vec})[0][0][0])
        raw_p50 = float(p50_sess.run(None, {"input": vec})[0][0][0])
        raw_p90 = float(p90_sess.run(None, {"input": vec})[0][0][0])

        eta_min = max(1.0, raw_p10 - q_val)
        eta_max = max(eta_min, raw_p90 + q_val)
        eta_p50 = min(max(raw_p50, eta_min), eta_max)

        fixtures.append(
            {
                "input": raw_input,
                "expected_prob": late_prob,
                "expected_label": int(late_prob >= threshold),
                "expected_eta_min_days": round(eta_min, 5),
                "expected_eta_p50_days": round(eta_p50, 5),
                "expected_eta_max_days": round(eta_max, 5),
            }
        )

    out = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "threshold": threshold,
        "cases": fixtures,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(out, indent=2))
    print(f"\nExported model.onnx, eta_p10/p50/p90.onnx, cqr_calibration.json & {len(fixtures)} fixtures (schema_hash={schema_hash})")

if __name__ == "__main__":
    main()
