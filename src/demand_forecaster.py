"""
Talep tahminleme modülü (v6 - Competition).
Hibrit yaklaşım: STL + WMA + LightGBM + DOW_AVG ensemble.

v6 İyileştirmeler:
- [A] Rolling CV tabanlı DOW bias düzeltmesi
- [B] Multi-window rolling cross-validation
- [D] Post-holiday rebound tespiti ve düzeltmesi
- [E] LightGBM basit hiperparametre tuning (CV tabanlı)
- Önceki tüm düzeltmeler korunuyor (anomali, vectorized lag, vb.)
"""
import logging
import warnings
import numpy as np
import pandas as pd
from . import config

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Bilinen anomali tarihleri (tatil etkisi veya veri kesilmesi)
ANOMALY_DATES = {
    pd.Timestamp("2026-04-30"),  # 1 Mayıs öncesi - 23K desi (normal ~960K)
    pd.Timestamp("2026-05-01"),  # İşçi Bayramı - 102K (normal ~960K)
    pd.Timestamp("2026-05-10"),  # Veri kesilmesi - 25K (normal ~250K)
}


# ══════════════════════════════════════════════
# Rota Aktivite Analizi
# ══════════════════════════════════════════════

def analyze_route_activity(talep_df, routes):
    """Her rota × haftanın günü için aktiflik oranı hesaplar."""
    activity = {}
    talep_df = talep_df.copy()
    talep_df["dow"] = talep_df["tarih"].dt.dayofweek
    dow_total_days = talep_df.groupby("dow")["tarih"].nunique()

    for cikis, varis in routes:
        route_data = talep_df[(talep_df["cikis"] == cikis) & (talep_df["varis"] == varis)]
        for dow in range(7):
            dow_route = route_data[route_data["dow"] == dow]
            total_possible = dow_total_days.get(dow, 1)
            active_days = dow_route["tarih"].nunique()
            activity[(cikis, varis, dow)] = active_days / total_possible if total_possible > 0 else 0

    return activity


# ══════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════

def add_features(df):
    """Zaman serisi feature'larını ekler."""
    df = df.copy()
    df["dow"] = df["tarih"].dt.dayofweek
    df["day_of_month"] = df["tarih"].dt.day
    df["week"] = df["tarih"].dt.isocalendar().week.astype(int)
    df["month"] = df["tarih"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)

    holidays = set(pd.to_datetime(config.PUBLIC_HOLIDAYS_2026))
    df["is_holiday"] = df["tarih"].isin(holidays).astype(int)

    df["is_post_holiday"] = 0
    for h in holidays:
        next_day = h + pd.Timedelta(days=1)
        while next_day.dayofweek >= 5:
            next_day += pd.Timedelta(days=1)
        df.loc[df["tarih"] == next_day, "is_post_holiday"] = 1

    # [D] Post-holiday rebound feature
    df["is_rebound_week"] = 0
    for h in holidays:
        for day_offset in range(1, 8):
            rebound_date = h + pd.Timedelta(days=day_offset)
            df.loc[df["tarih"] == rebound_date, "is_rebound_week"] = 1

    return df


def add_lag_features(route_df, lags=[7, 14, 21, 28]):
    """Rota bazında lag feature'ları ekler. Vectorized anomali düzeltmesi."""
    df = route_df.copy().sort_values("tarih")
    if "dow" not in df.columns:
        df["dow"] = df["tarih"].dt.dayofweek

    # Temiz DOW ortalamaları — bir kez hesapla
    clean_dow_means = (
        df[~df["tarih"].isin(ANOMALY_DATES)]
        .groupby("dow")["desi"]
        .apply(lambda x: x.tail(4).mean())
        .to_dict()
    )

    for lag in lags:
        df[f"lag_{lag}"] = df["desi"].shift(lag)
        lookback_dates = df["tarih"] - pd.Timedelta(days=lag)
        anomaly_mask = lookback_dates.isin(ANOMALY_DATES)
        if anomaly_mask.any():
            df.loc[anomaly_mask, f"lag_{lag}"] = (
                df.loc[anomaly_mask, "dow"].map(clean_dow_means)
            )

    df["rolling_7"] = df["desi"].rolling(7, min_periods=1).mean()
    df["rolling_14"] = df["desi"].rolling(14, min_periods=1).mean()
    df["rolling_28"] = df["desi"].rolling(28, min_periods=1).mean()
    df["rolling_7_std"] = df["desi"].rolling(7, min_periods=1).std().fillna(0)
    df["same_dow_rolling_mean"] = df.groupby("dow")["desi"].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )

    return df


# ══════════════════════════════════════════════
# Yöntem 1: STL Decomposition
# ══════════════════════════════════════════════

def forecast_stl(route_series, forecast_dates):
    """STL-benzeri decomposition. Son 6 hafta mevsimsellik + trend."""
    df = route_series.copy().sort_values("tarih")
    df["dow"] = df["tarih"].dt.dayofweek

    if len(df) < 7:
        dow_means = df.groupby("dow")["desi"].mean()
        return {d: max(0, dow_means.get(d.dayofweek, df["desi"].mean())) for d in forecast_dates}

    recent = df[df["tarih"] >= df["tarih"].max() - pd.Timedelta(days=42)].copy()
    recent_clean = recent[~recent["tarih"].isin(ANOMALY_DATES)]
    if len(recent_clean) < 5:
        recent_clean = recent

    overall_mean = recent_clean["desi"].mean()
    if overall_mean == 0:
        return {d: 0 for d in forecast_dates}

    dow_means = recent_clean.groupby("dow")["desi"].mean()
    seasonal_factors = {dow: m / overall_mean for dow, m in dow_means.items()}

    last_3w = df[(df["tarih"] >= df["tarih"].max() - pd.Timedelta(days=21)) &
                  (~df["tarih"].isin(ANOMALY_DATES))]
    prev_3w = df[(df["tarih"] >= df["tarih"].max() - pd.Timedelta(days=42)) &
                  (df["tarih"] < df["tarih"].max() - pd.Timedelta(days=21)) &
                  (~df["tarih"].isin(ANOMALY_DATES))]

    def deseasonal_mean(subset):
        if len(subset) == 0:
            return 0
        sf = subset["dow"].map(lambda d: seasonal_factors.get(d, 1))
        sf = sf.replace(0, np.nan)
        adjusted = subset["desi"] / sf
        return adjusted.dropna().mean() if not adjusted.dropna().empty else 0

    last_level = deseasonal_mean(last_3w)
    prev_level = deseasonal_mean(prev_3w)

    if prev_level > 0 and last_level > 0:
        trend_ratio = np.clip(last_level / prev_level, 0.85, 1.15)
    else:
        trend_ratio = 1.0

    forecasts = {}
    for d in forecast_dates:
        sf = seasonal_factors.get(d.dayofweek, 1.0)
        forecasts[d] = max(0, last_level * trend_ratio * sf)

    return forecasts


# ══════════════════════════════════════════════
# Yöntem 2: Ağırlıklı Hareketli Ortalama
# ══════════════════════════════════════════════

def forecast_wma(route_series, forecast_dates):
    """WMA + tatil haftası atlama + trend adjustment."""
    df = route_series.copy().sort_values("tarih")
    df["dow"] = df["tarih"].dt.dayofweek
    holidays = set(pd.to_datetime(config.PUBLIC_HOLIDAYS_2026))
    holidays.update(ANOMALY_DATES)

    weights = config.WMA_WEIGHTS
    n_weeks = config.WMA_WEEKS

    forecasts = {}
    for d in forecast_dates:
        dow = d.dayofweek
        values = []
        w = 1
        max_lookback = 12

        while len(values) < n_weeks and w <= max_lookback:
            lookback_date = d - pd.Timedelta(weeks=w)
            w += 1

            if lookback_date in ANOMALY_DATES:
                continue

            week_start = lookback_date - pd.Timedelta(days=lookback_date.dayofweek)
            week_dates = [week_start + pd.Timedelta(days=i) for i in range(7)]
            is_holiday_week = any(wd in holidays for wd in week_dates)

            match = df[df["tarih"] == lookback_date]
            if len(match) > 0:
                val = match["desi"].values[0]
                dow_avg = df[df["dow"] == dow]["desi"].mean()
                if is_holiday_week and val < dow_avg * 0.4:
                    continue
                values.append(val)
            elif not is_holiday_week:
                dow_data = df[df["dow"] == dow]
                if len(dow_data) > 0:
                    closest_idx = (dow_data["tarih"] - lookback_date).abs().argsort().iloc[0]
                    values.append(dow_data.iloc[closest_idx]["desi"])

        if len(values) == 0:
            dow_data = df[df["dow"] == dow]
            forecasts[d] = dow_data["desi"].mean() if len(dow_data) > 0 else 0
        else:
            w_use = weights[:len(values)]
            w_sum = sum(w_use)
            w_normalized = [wi / w_sum for wi in w_use]
            forecasts[d] = max(0, sum(v * wt for v, wt in zip(values, w_normalized)))

    return forecasts


# ══════════════════════════════════════════════
# Yöntem 3: LightGBM [İyileştirme E — Tuning]
# ══════════════════════════════════════════════

def _select_best_lgb_params(X_train, y_train):
    """
    [E] Basit grid search ile en iyi LightGBM parametrelerini seçer.
    Eğitim setini 80/20 bölerek internal validation yapar.
    Küçük veri setlerinde varsayılan parametreleri döner.
    """
    if len(X_train) < 30:
        return {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.03,
            "num_leaves": 20, "min_child_samples": 3,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.5, "reg_lambda": 0.5,
        }

    try:
        import lightgbm as lgb
    except ImportError:
        return {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.03,
        }

    # 80/20 split (kronolojik)
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    if len(X_val) < 5:
        return {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.03,
            "num_leaves": 20, "min_child_samples": 3,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.5, "reg_lambda": 0.5,
        }

    best_score = float("inf")
    best_params = None

    grid = config.LGBM_PARAM_GRID

    for n_est in grid["n_estimators"]:
        for md in grid["max_depth"]:
            for lr in grid["learning_rate"]:
                for nl in grid["num_leaves"]:
                    params = {
                        "n_estimators": n_est, "max_depth": md,
                        "learning_rate": lr, "num_leaves": nl,
                        "min_child_samples": 3,
                        "subsample": 0.8, "colsample_bytree": 0.8,
                        "reg_alpha": 0.5, "reg_lambda": 0.5,
                        "verbose": -1, "random_state": 42,
                    }
                    model = lgb.LGBMRegressor(**params)
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                    mse = np.mean((y_val - preds) ** 2)
                    if mse < best_score:
                        best_score = mse
                        best_params = params

    return best_params


def forecast_ml(talep_df, forecast_dates, routes, tune=True):
    """LightGBM ile rota bazında tahmin. Anomali tarihleri eğitimden çıkarılır."""
    try:
        import lightgbm as lgb
        use_lgb = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        use_lgb = False
        logger.warning("LightGBM bulunamadı, sklearn GradientBoosting kullanılacak")

    df = talep_df.copy()
    df = add_features(df)
    df_clean = df[~df["tarih"].isin(ANOMALY_DATES)].copy()

    all_forecasts = {}

    for cikis, varis in routes:
        route_data = df_clean[(df_clean["cikis"] == cikis) & (df_clean["varis"] == varis)].copy()
        route_data_full = df[(df["cikis"] == cikis) & (df["varis"] == varis)].copy()

        if len(route_data) < 14:
            dow_means = route_data.groupby("dow")["desi"].mean().to_dict()
            for d in forecast_dates:
                all_forecasts[(cikis, varis, d)] = max(0, dow_means.get(d.dayofweek, route_data["desi"].mean() if len(route_data) > 0 else 0))
            continue

        route_data = add_lag_features(route_data)

        feature_cols = [
            "dow", "day_of_month", "week", "month", "is_weekend", "is_sunday",
            "is_holiday", "is_post_holiday",
            "lag_7", "lag_14", "lag_21", "lag_28",
            "rolling_7", "rolling_14", "rolling_28",
            "rolling_7_std", "same_dow_rolling_mean",
        ]

        train_data = route_data.dropna(subset=feature_cols)

        if len(train_data) < 10:
            dow_means = route_data.groupby("dow")["desi"].mean().to_dict()
            for d in forecast_dates:
                all_forecasts[(cikis, varis, d)] = max(0, dow_means.get(d.dayofweek, 0))
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data["desi"].values

        if use_lgb:
            # [E] Tuning: yüksek hacimli rotalarda parametre seçimi, diğerlerinde default
            route_total_demand = route_data["desi"].sum()
            HIGH_VOLUME_THRESHOLD = 50_000  # toplam desi eşiği

            should_tune = (
                tune and
                len(train_data) >= 30 and
                route_total_demand > HIGH_VOLUME_THRESHOLD
            )

            if should_tune:
                params = _select_best_lgb_params(X_train, y_train)
                params["verbose"] = -1
                params["random_state"] = 42
                model = lgb.LGBMRegressor(**params)
                logger.debug(f"  Tuning uygulandı: {cikis}→{varis} "
                             f"(toplam talep: {route_total_demand:,.0f})")
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.03,
                    num_leaves=20, min_child_samples=3,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=0.5,
                    verbose=-1, random_state=42,
                )
        else:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.03,
                min_samples_leaf=3, subsample=0.8, random_state=42,
            )

        model.fit(X_train, y_train)

        route_data_full = add_features(route_data_full)
        for d in forecast_dates:
            feat_row = _create_forecast_features(d, route_data_full, feature_cols)
            if feat_row is not None:
                pred = model.predict(feat_row.reshape(1, -1))[0]
                all_forecasts[(cikis, varis, d)] = max(0, pred)
            else:
                dow_means = route_data.groupby("dow")["desi"].mean().to_dict()
                all_forecasts[(cikis, varis, d)] = max(0, dow_means.get(d.dayofweek, 0))

    return all_forecasts


def _create_forecast_features(target_date, route_data, feature_cols):
    """Tahmin tarihi için feature vektörü oluşturur. Anomali lag'larını düzeltir."""
    holidays = set(pd.to_datetime(config.PUBLIC_HOLIDAYS_2026))

    dow = target_date.dayofweek
    day_of_month = target_date.day
    week = target_date.isocalendar().week
    month = target_date.month
    is_weekend = 1 if dow >= 5 else 0
    is_sunday = 1 if dow == 6 else 0
    is_holiday = 1 if target_date in holidays else 0

    is_post_holiday = 0
    yesterday = target_date - pd.Timedelta(days=1)
    if yesterday in holidays:
        is_post_holiday = 1

    sorted_data = route_data.sort_values("tarih")

    lag_vals = {}
    for lag in [7, 14, 21, 28]:
        lookback = target_date - pd.Timedelta(days=lag)
        if lookback in ANOMALY_DATES:
            dow_target = lookback.dayofweek
            dow_col = sorted_data["dow"] if "dow" in sorted_data.columns else sorted_data["tarih"].dt.dayofweek
            clean_dow = sorted_data[(dow_col == dow_target) & (~sorted_data["tarih"].isin(ANOMALY_DATES))]
            lag_vals[f"lag_{lag}"] = clean_dow["desi"].tail(4).mean() if len(clean_dow) > 0 else 0
        else:
            match = sorted_data[sorted_data["tarih"] == lookback]
            if len(match) > 0:
                lag_vals[f"lag_{lag}"] = match["desi"].values[0]
            else:
                dow_data = sorted_data[sorted_data["dow"] == dow] if "dow" in sorted_data.columns else sorted_data
                lag_vals[f"lag_{lag}"] = dow_data["desi"].iloc[-1] if len(dow_data) > 0 else 0

    recent = sorted_data[~sorted_data["tarih"].isin(ANOMALY_DATES)].tail(28)
    rolling_7 = recent.tail(7)["desi"].mean() if len(recent) >= 7 else recent["desi"].mean() if len(recent) > 0 else 0
    rolling_14 = recent.tail(14)["desi"].mean() if len(recent) >= 14 else recent["desi"].mean() if len(recent) > 0 else 0
    rolling_28 = recent["desi"].mean() if len(recent) > 0 else 0
    rolling_7_std = recent.tail(7)["desi"].std() if len(recent) >= 7 else 0
    if pd.isna(rolling_7_std):
        rolling_7_std = 0

    dow_col = sorted_data["dow"] if "dow" in sorted_data.columns else sorted_data["tarih"].dt.dayofweek
    dow_data = sorted_data[(dow_col == dow) & (~sorted_data["tarih"].isin(ANOMALY_DATES))]
    same_dow_rolling_mean = dow_data.tail(4)["desi"].mean() if len(dow_data) > 0 else 0

    features = np.array([
        dow, day_of_month, week, month, is_weekend, is_sunday,
        is_holiday, is_post_holiday,
        lag_vals["lag_7"], lag_vals["lag_14"], lag_vals["lag_21"], lag_vals["lag_28"],
        rolling_7, rolling_14, rolling_28,
        rolling_7_std, same_dow_rolling_mean,
    ])

    return features


# ══════════════════════════════════════════════
# Yöntem 4: DOW Ortalaması (stabilizatör)
# ══════════════════════════════════════════════

def forecast_dow_avg(route_series, forecast_dates):
    """Son 8 haftanın aynı gün ortalaması (temiz). En stabil yöntem."""
    df = route_series.copy().sort_values("tarih")
    df["dow"] = df["tarih"].dt.dayofweek

    df_clean = df[~df["tarih"].isin(ANOMALY_DATES)]
    holidays = set(pd.to_datetime(config.PUBLIC_HOLIDAYS_2026))
    df_clean = df_clean[~df_clean["tarih"].isin(holidays)]

    recent = df_clean[df_clean["tarih"] >= df_clean["tarih"].max() - pd.Timedelta(days=56)]
    if len(recent) < 7:
        recent = df_clean

    dow_means = recent.groupby("dow")["desi"].mean().to_dict()

    return {d: max(0, dow_means.get(d.dayofweek, 0)) for d in forecast_dates}


# ══════════════════════════════════════════════
# Ensemble Tahmin (v6)
# ══════════════════════════════════════════════

def forecast_ensemble(talep_df, forecast_dates, routes, weights=None, tune_lgb=True):
    """4 yöntemin ağırlıklı ensemble'ı + rota aktiflik filtresi."""
    if weights is None:
        weights = config.ENSEMBLE_WEIGHTS

    activity = analyze_route_activity(talep_df, routes)

    logger.info("STL Decomposition tahminleri hesaplanıyor...")
    stl_forecasts = {}
    for cikis, varis in routes:
        rd = talep_df[(talep_df["cikis"] == cikis) & (talep_df["varis"] == varis)][["tarih", "desi"]]
        for d, val in forecast_stl(rd, forecast_dates).items():
            stl_forecasts[(cikis, varis, d)] = val

    logger.info("Ağırlıklı Hareketli Ortalama tahminleri hesaplanıyor...")
    wma_forecasts = {}
    for cikis, varis in routes:
        rd = talep_df[(talep_df["cikis"] == cikis) & (talep_df["varis"] == varis)][["tarih", "desi"]]
        for d, val in forecast_wma(rd, forecast_dates).items():
            wma_forecasts[(cikis, varis, d)] = val

    logger.info("ML (LightGBM) tahminleri hesaplanıyor...")
    ml_forecasts = forecast_ml(talep_df, forecast_dates, routes, tune=tune_lgb)

    logger.info("DOW ortalaması tahminleri hesaplanıyor...")
    dow_forecasts = {}
    for cikis, varis in routes:
        rd = talep_df[(talep_df["cikis"] == cikis) & (talep_df["varis"] == varis)][["tarih", "desi"]]
        for d, val in forecast_dow_avg(rd, forecast_dates).items():
            dow_forecasts[(cikis, varis, d)] = val

    logger.info("Ensemble birleştiriliyor...")
    results = []
    skipped = 0

    w_stl = weights.get("stl", 0.30)
    w_wma = weights.get("wma", 0.25)
    w_ml = weights.get("ml", 0.30)
    w_dow = weights.get("dow_avg", 0.15)

    for cikis, varis in routes:
        for d in forecast_dates:
            act_rate = activity.get((cikis, varis, d.dayofweek), 0)

            if act_rate < 0.15:
                results.append({"cikis": cikis, "varis": varis, "tarih": d,
                                "tahmin_desi": 0, "stl_tahmin": 0, "wma_tahmin": 0, "ml_tahmin": 0})
                skipped += 1
                continue

            key = (cikis, varis, d)
            stl_val = stl_forecasts.get(key, 0)
            wma_val = wma_forecasts.get(key, 0)
            ml_val = ml_forecasts.get(key, 0)
            dow_val = dow_forecasts.get(key, 0)

            ensemble_val = w_stl * stl_val + w_wma * wma_val + w_ml * ml_val + w_dow * dow_val

            if act_rate < 0.7:
                ensemble_val *= act_rate

            results.append({
                "cikis": cikis, "varis": varis, "tarih": d,
                "tahmin_desi": round(max(0, ensemble_val), 2),
                "stl_tahmin": round(stl_val, 2),
                "wma_tahmin": round(wma_val, 2),
                "ml_tahmin": round(ml_val, 2),
            })

    logger.info(f"İnaktif rota×gün atlandı: {skipped}")
    return pd.DataFrame(results)


# ══════════════════════════════════════════════
# [A] Bias Düzeltmesi — Rolling CV Tabanlı
# ══════════════════════════════════════════════

def compute_bias_factors(talep_df, routes):
    """
    [A] Son 2-3 normal haftanın backtest'inden DOW bazlı bias faktörleri hesaplar.
    Tatil haftaları (anomali içeren) hesaba katılmaz.

    Returns:
        dict {dow: correction_factor}  — örn. {0: 1.12, 1: 1.18, ...}
    """
    logger.info("DOW bazlı bias faktörleri hesaplanıyor (rolling CV)...")

    windows = config.ROLLING_CV_WINDOWS
    dow_errors = {d: [] for d in range(7)}  # {dow: [actual/predicted ratios]}

    for holdout_start_str, holdout_end_str in windows:
        holdout_start = pd.to_datetime(holdout_start_str)
        holdout_end = pd.to_datetime(holdout_end_str)

        train_df = talep_df[talep_df["tarih"] < holdout_start].copy()
        test_df = talep_df[
            (talep_df["tarih"] >= holdout_start) & (talep_df["tarih"] <= holdout_end)
        ].copy()

        # Anomali tarihlerini test setinden çıkar
        test_df = test_df[~test_df["tarih"].isin(ANOMALY_DATES)]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        forecast_dates = pd.date_range(holdout_start, holdout_end, freq="D")

        # Hızlı ensemble (LightGBM tuning olmadan)
        predictions = forecast_ensemble(train_df, forecast_dates, routes, tune_lgb=False)

        merged = predictions.merge(
            test_df[["cikis", "varis", "tarih", "desi"]],
            on=["cikis", "varis", "tarih"],
            how="inner",
        )

        if len(merged) == 0:
            continue

        # Günlük toplam bazında bias hesapla (rota bazı çok noisy)
        merged["dow"] = merged["tarih"].dt.dayofweek
        for dow in range(7):
            dow_data = merged[merged["dow"] == dow]
            if len(dow_data) > 0:
                actual_total = dow_data["desi"].sum()
                pred_total = dow_data["tahmin_desi"].sum()
                if pred_total > 0:
                    ratio = actual_total / pred_total
                    dow_errors[dow].append(ratio)

    # Ortalama bias faktörü hesapla + damping
    damping = config.BIAS_DAMPING_FACTOR
    bias_factors = {}

    for dow in range(7):
        if len(dow_errors[dow]) > 0:
            raw_factor = np.median(dow_errors[dow])  # median daha robust
            # Damping: 1.0'a doğru çek (overfitting önleme)
            damped = 1.0 + (raw_factor - 1.0) * damping
            # Makul aralıkta tut
            bias_factors[dow] = np.clip(damped, 0.85, 1.40)
        else:
            bias_factors[dow] = 1.0

    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    for dow in range(7):
        cv_count = len(dow_errors[dow])
        logger.info(f"  {day_names[dow]}: bias={bias_factors[dow]:.3f} "
                     f"(CV windows: {cv_count}, raw ratios: {[f'{r:.2f}' for r in dow_errors[dow]]})")

    return bias_factors


def apply_bias_correction(tahmin_df, bias_factors):
    """
    [A] Tahmin DataFrame'ine DOW bazlı bias düzeltmesi uygular.

    Args:
        tahmin_df: forecast_ensemble() çıktısı
        bias_factors: compute_bias_factors() çıktısı
    Returns:
        Düzeltilmiş tahmin DataFrame
    """
    df = tahmin_df.copy()
    df["dow"] = df["tarih"].dt.dayofweek
    df["bias_factor"] = df["dow"].map(bias_factors)
    df["tahmin_desi_raw"] = df["tahmin_desi"]
    df["tahmin_desi"] = (df["tahmin_desi"] * df["bias_factor"]).round(2)
    df["tahmin_desi"] = df["tahmin_desi"].clip(lower=0)

    raw_total = df["tahmin_desi_raw"].sum()
    corrected_total = df["tahmin_desi"].sum()
    change_pct = (corrected_total - raw_total) / raw_total * 100 if raw_total > 0 else 0
    logger.info(f"Bias düzeltmesi uygulandı: {raw_total:,.0f} → {corrected_total:,.0f} desi ({change_pct:+.1f}%)")

    return df


# ══════════════════════════════════════════════
# [D] Post-Holiday Rebound Tespiti
# ══════════════════════════════════════════════

def detect_rebound_period(forecast_dates):
    """
    [D] Tahmin döneminin tatil sonrası rebound periyodunda olup olmadığını tespit eder.

    Returns:
        float: rebound faktörü (1.0 = rebound yok, >1.0 = rebound var)
    """
    if not config.REBOUND_DETECTION_ENABLED:
        return 1.0

    holidays = set(pd.to_datetime(config.PUBLIC_HOLIDAYS_2026))
    first_date = min(forecast_dates)

    # Son 14 gün içinde tatil var mı?
    lookback = config.REBOUND_LOOKBACK_DAYS
    for d in range(1, lookback + 1):
        check_date = first_date - pd.Timedelta(days=d)
        if check_date in holidays or check_date in ANOMALY_DATES:
            # Tatil ne kadar yakında? Yakınsa rebound daha güçlü
            proximity = 1.0 - (d / lookback) * 0.5  # 1.0 → 0.5 arası
            rebound_factor = 1.0 + 0.08 * proximity  # Max %8 ek talep
            logger.info(f"Post-holiday rebound tespit edildi: "
                         f"tatil {check_date.strftime('%Y-%m-%d')}, "
                         f"mesafe {d} gün, faktör {rebound_factor:.3f}")
            return rebound_factor

    return 1.0


# ══════════════════════════════════════════════
# P10-P50-P90 Quantile Tahminleri
# ══════════════════════════════════════════════

def compute_quantile_forecasts(talep_df, forecast_dates, routes, base_forecasts_df):
    """Ensemble tahminleri üzerine rota bazlı P10, P50, P90 güven aralığı üretir."""
    results = []

    for cikis, varis in routes:
        route_hist = talep_df[
            (talep_df["cikis"] == cikis) &
            (talep_df["varis"] == varis) &
            (~talep_df["tarih"].isin(ANOMALY_DATES))
        ].copy()
        route_hist["dow"] = route_hist["tarih"].dt.dayofweek

        route_fc = base_forecasts_df[
            (base_forecasts_df["cikis"] == cikis) &
            (base_forecasts_df["varis"] == varis)
        ]

        for _, fc_row in route_fc.iterrows():
            dow = fc_row["tarih"].dayofweek
            dow_vals = route_hist[route_hist["dow"] == dow]["desi"]

            if len(dow_vals) >= 4:
                cv = dow_vals.std() / dow_vals.mean() if dow_vals.mean() > 0 else 0.3
            else:
                cv = 0.3

            mu = fc_row["tahmin_desi"]
            sigma = mu * cv

            results.append({
                "cikis":           cikis,
                "varis":           varis,
                "tarih":           fc_row["tarih"],
                "tahmin_desi_p10": round(max(0, mu - 1.28 * sigma), 2),
                "tahmin_desi_p50": round(max(0, mu), 2),
                "tahmin_desi_p90": round(max(0, mu + 1.28 * sigma), 2),
            })

    return pd.DataFrame(results)


# ══════════════════════════════════════════════
# [B] Rolling Cross-Validation
# ══════════════════════════════════════════════

def rolling_cv(talep_df, routes):
    """
    [B] Multi-window sliding cross-validation.
    Her pencere için WMAPE, MAPE, RMSE hesaplar ve ortalama döner.

    Returns:
        dict: ortalama metrikler + pencere detayları
    """
    logger.info("=" * 50)
    logger.info("ROLLING CROSS-VALIDATION")
    logger.info("=" * 50)

    windows = config.ROLLING_CV_WINDOWS
    all_results = []

    for i, (holdout_start_str, holdout_end_str) in enumerate(windows):
        holdout_start = pd.to_datetime(holdout_start_str)
        holdout_end = pd.to_datetime(holdout_end_str)

        train_df = talep_df[talep_df["tarih"] < holdout_start].copy()
        test_df = talep_df[
            (talep_df["tarih"] >= holdout_start) & (talep_df["tarih"] <= holdout_end)
        ].copy()

        # Anomali tarihlerini çıkar
        test_df = test_df[~test_df["tarih"].isin(ANOMALY_DATES)]

        if len(train_df) < 50 or len(test_df) == 0:
            logger.info(f"  Window {i+1}: Yetersiz veri, atlanıyor")
            continue

        forecast_dates = pd.date_range(holdout_start, holdout_end, freq="D")

        logger.info(f"  Window {i+1}: {holdout_start_str} → {holdout_end_str} "
                     f"(train={len(train_df)}, test={len(test_df)})")

        predictions = forecast_ensemble(train_df, forecast_dates, routes, tune_lgb=False)

        merged = predictions.merge(
            test_df[["cikis", "varis", "tarih", "desi"]],
            on=["cikis", "varis", "tarih"],
            how="inner",
        )

        if len(merged) == 0:
            continue

        actual = merged["desi"].values
        predicted = merged["tahmin_desi"].values

        total_abs_error = np.sum(np.abs(actual - predicted))
        total_actual = np.sum(np.abs(actual))
        wmape = (total_abs_error / total_actual) * 100 if total_actual > 0 else 0

        mask = actual > 100
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else wmape

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))

        daily_actual = merged.groupby("tarih")["desi"].sum()
        daily_pred = merged.groupby("tarih")["tahmin_desi"].sum()
        daily_wmape = np.sum(np.abs(daily_actual - daily_pred)) / np.sum(daily_actual) * 100

        window_result = {
            "window": f"{holdout_start_str} → {holdout_end_str}",
            "wmape": wmape, "mape": mape, "daily_wmape": daily_wmape,
            "rmse": rmse, "mae": mae, "n_matched": len(merged),
        }
        all_results.append(window_result)

        logger.info(f"    WMAPE={wmape:.2f}%, Daily WMAPE={daily_wmape:.2f}%, "
                     f"MAPE={mape:.2f}%, RMSE={rmse:,.0f}")

    if not all_results:
        logger.warning("Rolling CV: Hiç sonuç üretilemedi!")
        return {"avg_wmape": None, "windows": []}

    avg_wmape = np.mean([r["wmape"] for r in all_results])
    avg_mape = np.mean([r["mape"] for r in all_results])
    avg_daily_wmape = np.mean([r["daily_wmape"] for r in all_results])
    avg_rmse = np.mean([r["rmse"] for r in all_results])
    avg_mae = np.mean([r["mae"] for r in all_results])

    logger.info(f"\n  === ROLLING CV ORTALAMA ({len(all_results)} pencere) ===")
    logger.info(f"  WMAPE: {avg_wmape:.2f}%")
    logger.info(f"  Günlük WMAPE: {avg_daily_wmape:.2f}%")
    logger.info(f"  MAPE: {avg_mape:.2f}%")
    logger.info(f"  RMSE: {avg_rmse:,.0f}")
    logger.info(f"  MAE: {avg_mae:,.0f}")

    return {
        "avg_wmape": avg_wmape, "avg_mape": avg_mape,
        "avg_daily_wmape": avg_daily_wmape,
        "avg_rmse": avg_rmse, "avg_mae": avg_mae,
        "n_windows": len(all_results), "windows": all_results,
    }


# ══════════════════════════════════════════════
# Backtesting (Single Holdout — Backward compat)
# ══════════════════════════════════════════════

def backtest(talep_df, holdout_start, holdout_end, routes):
    """Holdout test seti ile backtesting. Anomali tarihleri test setinden çıkarılır."""
    holdout_start = pd.to_datetime(holdout_start)
    holdout_end = pd.to_datetime(holdout_end)

    train_df = talep_df[talep_df["tarih"] < holdout_start].copy()
    test_df = talep_df[
        (talep_df["tarih"] >= holdout_start) & (talep_df["tarih"] <= holdout_end)
    ].copy()

    test_df = test_df[~test_df["tarih"].isin(ANOMALY_DATES)]

    forecast_dates = pd.date_range(holdout_start, holdout_end, freq="D")

    logger.info(f"Eğitim: {len(train_df)} kayıt ({train_df['tarih'].min()} - {train_df['tarih'].max()})")
    logger.info(f"Test: {len(test_df)} kayıt ({holdout_start} - {holdout_end})")

    predictions = forecast_ensemble(train_df, forecast_dates, routes, tune_lgb=False)

    merged = predictions.merge(
        test_df[["cikis", "varis", "tarih", "desi"]],
        on=["cikis", "varis", "tarih"],
        how="inner",
    )

    unmatched = predictions.merge(
        test_df[["cikis", "varis", "tarih", "desi"]],
        on=["cikis", "varis", "tarih"],
        how="left",
    )
    over_forecast = unmatched[unmatched["desi"].isna() & (unmatched["tahmin_desi"] > 0)]

    if len(merged) == 0:
        logger.error("Eşleşen veri bulunamadı!")
        return {"mape": float("inf"), "rmse": float("inf"), "mae": float("inf")}

    actual = merged["desi"].values
    predicted = merged["tahmin_desi"].values

    total_abs_error = np.sum(np.abs(actual - predicted))
    total_actual = np.sum(np.abs(actual))
    wmape = (total_abs_error / total_actual) * 100 if total_actual > 0 else 0

    mask = actual > 100
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else wmape

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    daily_actual = merged.groupby("tarih")["desi"].sum()
    daily_pred = merged.groupby("tarih")["tahmin_desi"].sum()
    daily_wmape = np.sum(np.abs(daily_actual - daily_pred)) / np.sum(daily_actual) * 100

    over_total = over_forecast["tahmin_desi"].sum() if len(over_forecast) > 0 else 0

    logger.info(f"=== BACKTEST SONUÇLARI ===")
    logger.info(f"  WMAPE (ağırlıklı): {wmape:.2f}%")
    logger.info(f"  MAPE (talep>100): {mape:.2f}%")
    logger.info(f"  Günlük WMAPE: {daily_wmape:.2f}%")
    logger.info(f"  RMSE: {rmse:,.0f}")
    logger.info(f"  MAE: {mae:,.0f}")
    logger.info(f"  Eşleşen rota×gün: {len(merged)}")
    logger.info(f"  Over-forecast: {over_total:,.0f} desi ({len(over_forecast)} rota×gün)")

    logger.info(f"  Günlük Toplam Karşılaştırma:")
    for tarih in sorted(merged["tarih"].unique()):
        day_actual = merged[merged["tarih"] == tarih]["desi"].sum()
        day_pred = merged[merged["tarih"] == tarih]["tahmin_desi"].sum()
        day_name = pd.to_datetime(tarih).strftime("%a")
        ape = abs(day_actual - day_pred) / day_actual * 100 if day_actual > 0 else 0
        logger.info(f"    {pd.to_datetime(tarih).strftime('%Y-%m-%d')} ({day_name}): "
                     f"Actual={day_actual:,.0f}, Pred={day_pred:,.0f}, APE={ape:.1f}%")

    return {
        "mape": mape, "wmape": wmape, "daily_wmape": daily_wmape,
        "rmse": rmse, "mae": mae,
        "details_df": merged, "over_forecast_desi": over_total,
    }


# ══════════════════════════════════════════════
# [v7] Birleşik Rolling CV + Bias Hesaplama
# ══════════════════════════════════════════════

def compute_rolling_cv_and_bias(talep_df, routes, windows=None):
    """
    Rolling CV metriklerini ve bias faktörlerini TEK geçişte hesaplar.
    Her pencere için forecast_ensemble yalnızca bir kez çağrılır.
    Eski rolling_cv() + compute_bias_factors() ikilisinin birleşimi.

    Returns:
        cv_summary (dict): avg_wmape, avg_daily_wmape, n_windows, windows listesi
        bias_factors (dict): {dow: correction_factor}
    """
    if windows is None:
        windows = config.ROLLING_CV_WINDOWS

    logger.info("=" * 50)
    logger.info("ROLLING CV + BIAS (birleştirilmiş — tek geçiş)")
    logger.info("=" * 50)

    dow_errors = {d: [] for d in range(7)}
    all_cv_results = []

    for i, (start_str, end_str) in enumerate(windows):
        holdout_start = pd.to_datetime(start_str)
        holdout_end = pd.to_datetime(end_str)

        train_df = talep_df[talep_df["tarih"] < holdout_start].copy()
        test_df = talep_df[
            (talep_df["tarih"] >= holdout_start) &
            (talep_df["tarih"] <= holdout_end)
        ].copy()
        test_df = test_df[~test_df["tarih"].isin(ANOMALY_DATES)]

        if len(train_df) < 50 or len(test_df) == 0:
            logger.info(f"  Window {i+1}: Yetersiz veri, atlanıyor")
            continue

        forecast_dates = pd.date_range(holdout_start, holdout_end, freq="D")
        logger.info(f"  Window {i+1}: {start_str} → {end_str} "
                     f"(train={len(train_df)}, test={len(test_df)})")

        # TEK ÇAĞRI — hem CV hem bias için kullanılır
        predictions = forecast_ensemble(
            train_df, forecast_dates, routes, tune_lgb=False
        )

        merged = predictions.merge(
            test_df[["cikis", "varis", "tarih", "desi"]],
            on=["cikis", "varis", "tarih"],
            how="inner",
        )
        if len(merged) == 0:
            continue

        actual = merged["desi"].values
        predicted = merged["tahmin_desi"].values

        # CV metrikleri
        wmape = (np.sum(np.abs(actual - predicted)) /
                 np.sum(np.abs(actual))) * 100

        mask = actual > 100
        mape = (np.mean(np.abs((actual[mask] - predicted[mask]) /
                actual[mask])) * 100) if mask.sum() > 0 else wmape

        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))

        merged["dow"] = merged["tarih"].dt.dayofweek
        daily_actual = merged.groupby("tarih")["desi"].sum()
        daily_pred = merged.groupby("tarih")["tahmin_desi"].sum()
        daily_wmape = (np.sum(np.abs(daily_actual - daily_pred)) /
                       np.sum(daily_actual)) * 100

        all_cv_results.append({
            "window":      f"{start_str} → {end_str}",
            "wmape":       wmape,
            "mape":        mape,
            "daily_wmape": daily_wmape,
            "rmse":        rmse,
            "mae":         mae,
            "n_matched":   len(merged),
        })

        logger.info(f"    WMAPE={wmape:.2f}%, Daily={daily_wmape:.2f}%, "
                     f"MAPE={mape:.2f}%, RMSE={rmse:,.0f}")

        # Bias için DOW hataları topla
        for dow in range(7):
            dow_data = merged[merged["dow"] == dow]
            act_total = dow_data["desi"].sum()
            pred_total = dow_data["tahmin_desi"].sum()
            if pred_total > 0:
                dow_errors[dow].append(act_total / pred_total)

    # CV özeti
    if not all_cv_results:
        logger.warning("Rolling CV+Bias: Hiç sonuç üretilemedi!")
        return {"avg_wmape": None, "windows": []}, {d: 1.0 for d in range(7)}

    cv_summary = {
        "avg_wmape":       np.mean([r["wmape"]       for r in all_cv_results]),
        "avg_mape":        np.mean([r["mape"]        for r in all_cv_results]),
        "avg_daily_wmape": np.mean([r["daily_wmape"] for r in all_cv_results]),
        "avg_rmse":        np.mean([r["rmse"]        for r in all_cv_results]),
        "avg_mae":         np.mean([r["mae"]         for r in all_cv_results]),
        "n_windows":       len(all_cv_results),
        "windows":         all_cv_results,
    }

    logger.info(f"\n  === ROLLING CV ORTALAMA ({len(all_cv_results)} pencere) ===")
    logger.info(f"  WMAPE: {cv_summary['avg_wmape']:.2f}%")
    logger.info(f"  Günlük WMAPE: {cv_summary['avg_daily_wmape']:.2f}%")
    logger.info(f"  MAPE: {cv_summary['avg_mape']:.2f}%")

    # Bias faktörleri
    damping = config.BIAS_DAMPING_FACTOR
    bias_factors = {}
    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]

    for dow in range(7):
        if dow_errors[dow]:
            raw = np.median(dow_errors[dow])
            damped = 1.0 + (raw - 1.0) * damping
            bias_factors[dow] = np.clip(damped, 0.85, 1.40)
        else:
            bias_factors[dow] = 1.0

        logger.info(f"  Bias {day_names[dow]}: ×{bias_factors[dow]:.3f} "
                     f"(CV windows: {len(dow_errors[dow])}, "
                     f"raw: {[f'{r:.2f}' for r in dow_errors[dow]]})")

    return cv_summary, bias_factors

