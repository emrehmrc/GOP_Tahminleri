"""
monitoring/scorecard.py — ADM + GDZ ortak daily_scorecard (Faz 2, 2026-07-10).

2026-07-10'a kadar bu mantık iki ayrı, birebir aynı kopya olarak yaşıyordu
(adm live/src/scorecard.py, gdz talep/live/src/scorecard.py) — GDZ'ninki
hiç YAZILMAMIŞTI bile, run_daily.py'ye eklenmemişti, kimse fark etmemişti.
Bu paketin var olma sebebi tam olarak bu sınıf hatayı bir daha imkânsız
kılmak: tek kopya, `TenantConfig` ile parametrize.

Şema/karar kaynağı: stlf_faz1_scorecard_tasarim.md (K1-K5).
"""

from __future__ import annotations

import json
import logging
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from monitoring.tenant_config import TenantConfig

HOUR_BLOCKS = {
    "night":   range(0, 6),
    "morning": range(6, 10),
    "pv":      range(10, 17),
    "evening": range(17, 22),
}

_MAD_CONST = 1.4826


def _mape(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna()
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs((actual[valid] - pred[valid]) / (actual[valid] + 1e-10))) * 100)


def _wape(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna()
    if not valid.any():
        return np.nan
    denom = np.abs(actual[valid]).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(actual[valid] - pred[valid]).sum() / denom * 100)


def _rmse(pred: pd.Series, actual: pd.Series) -> float:
    valid = pred.notna() & actual.notna()
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((pred[valid] - actual[valid]) ** 2)))


def _me(pred: pd.Series, actual: pd.Series) -> float:
    """K3: pozitif = fazla tahmin (over-forecast)."""
    valid = pred.notna() & actual.notna()
    if not valid.any():
        return np.nan
    return float(np.mean(pred[valid] - actual[valid]))


def _joined_hourly(con: duckdb.DuckDBPyConnection, window_days: int) -> pd.DataFrame:
    """Faz 2 (2026-07-13): a7 — 168 saat (7 gün) önceki aynı-saat actual, naive
    benchmark için. "Model geçen haftanın aynı gününü kopyalamaktan iyi mi?"
    sorusunun kalıcı otomatik cevabı — 07-12 Pazar post-mortem'inde ADM'nin
    muhtemelen bu basit taban çizgisinin bile ALTINDA kaldığı görüldü, ama bunu
    ölçen hiçbir mekanizma yoktu. LEFT JOIN: sistemin ilk haftasında a7 doğal
    olarak NULL — bu satırlar için mape_naive_lag168 NaN kalır (hata değil)."""
    cutoff = (date.today() - pd.Timedelta(days=window_days)).isoformat()
    sql = """
        SELECT f.edas_id, f.target_date, f.horizon_day, f.target_ts, f.flag_holiday,
               f.y_pred_xgb, f.y_pred_lgbm, f.y_pred_cat, f.y_pred_chronos,
               f.y_pred_ens_raw, f.y_pred_final, f.meta_method,
               f.wx_temp_fcst, f.wx_ghi_fcst,
               a.y_actual, a.wx_temp_actual, a.wx_ghi_actual,
               a.data_quality_flag, a.known_event,
               a7.y_actual AS y_actual_lag168
        FROM forecast_log_v f
        INNER JOIN actuals_log_v a
          ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts
        LEFT JOIN actuals_log_v a7
          ON f.edas_id = a7.edas_id AND a7.target_ts = f.target_ts - INTERVAL '7 days'
        WHERE f.target_date >= ? AND a.y_actual IS NOT NULL
    """
    return con.execute(sql, [cutoff]).df()


def _daily_agg(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (edas_id, target_date, horizon_day), g in hourly.groupby(
        ["edas_id", "target_date", "horizon_day"]
    ):
        pred_final = g["y_pred_final"]
        actual = g["y_actual"]
        hours = pd.to_datetime(g["target_ts"]).dt.hour

        row = {
            "edas_id": edas_id,
            "target_date": target_date,
            "horizon_day": horizon_day,
            "n_hours": len(g),
            "mape": _mape(pred_final, actual),
            "wape": _wape(pred_final, actual),
            "rmse": _rmse(pred_final, actual),
            "me": _me(pred_final, actual),
            "mape_xgb": _mape(g["y_pred_xgb"], actual),
            "mape_lgbm": _mape(g["y_pred_lgbm"], actual),
            "mape_cat": _mape(g["y_pred_cat"], actual),
            "mape_chronos": _mape(g["y_pred_chronos"], actual),
            "mape_ens_raw": _mape(g["y_pred_ens_raw"], actual),
            "mape_final": _mape(pred_final, actual),
            "mape_naive_lag168": _mape(g["y_actual_lag168"], actual),
            "meta_method": (
                g["meta_method"].mode().iat[0]
                if "meta_method" in g.columns and g["meta_method"].notna().any() else None
            ),
            "flag_holiday": bool(g["flag_holiday"].iloc[0]) if "flag_holiday" in g.columns else False,
            "data_quality_flag_count": int((g["data_quality_flag"].fillna("") != "").sum()),
            "known_event_present": bool(g["known_event"].notna().any()),
        }

        # "Model geçen haftayı kopyalamaktan iyi mi?" — direkt sorulabilir bool.
        mape_final = row["mape_final"]
        mape_naive = row["mape_naive_lag168"]
        row["beats_naive_lag168"] = (
            bool(mape_final < mape_naive) if not (np.isnan(mape_final) or np.isnan(mape_naive)) else None
        )

        for label, hrs in HOUR_BLOCKS.items():
            mask = hours.isin(hrs)
            row[f"mape_{label}"] = _mape(pred_final[mask], actual[mask]) if mask.any() else np.nan

        ape = np.abs((actual - pred_final) / (actual + 1e-10)) * 100
        if ape.notna().any():
            idx = ape.idxmax()
            row["max_ape_hour"] = int(pd.to_datetime(g.loc[idx, "target_ts"]).hour)
            row["max_ape_value"] = float(ape.loc[idx])
        else:
            row["max_ape_hour"] = None
            row["max_ape_value"] = np.nan

        wx_temp_actual = g["wx_temp_actual"]
        wx_ghi_actual = g["wx_ghi_actual"]
        has_wx_actual = wx_temp_actual.notna().any() or wx_ghi_actual.notna().any()
        row["actuals_wave"] = "complete" if has_wx_actual else "load_only"
        row["temp_fcst_error"] = (
            float((g["wx_temp_fcst"] - wx_temp_actual).mean()) if wx_temp_actual.notna().any() else np.nan
        )
        row["ghi_fcst_error"] = (
            float((g["wx_ghi_fcst"] - wx_ghi_actual).mean()) if wx_ghi_actual.notna().any() else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


def _add_robust_z(daily: pd.DataFrame, config: TenantConfig) -> pd.DataFrame:
    """(mape - median_30d)/(1.4826*MAD_30d), (edas_id, horizon_day, flag_holiday) bazında,
    bugün kendi baseline'ından hariç (shift(1)) — tatil günleri hafta-içi baseline'ı kirletmez."""
    daily = daily.sort_values("target_date").copy()
    daily["robust_z"] = np.nan
    daily["baseline_mode"] = "warmup"
    daily["alert_flag"] = False

    win = config.z_baseline_window_days
    warmup_min = config.z_warmup_min_days

    for _, idx in daily.groupby(["edas_id", "horizon_day", "flag_holiday"]).groups.items():
        g = daily.loc[idx].sort_values("target_date")
        mape = g["mape"]
        prior_count = np.arange(len(mape))

        rolling_median = mape.shift(1).rolling(win, min_periods=1).median()
        abs_dev = (mape.shift(1) - rolling_median).abs()
        rolling_mad = abs_dev.rolling(win, min_periods=1).median()
        z = (mape - rolling_median) / (_MAD_CONST * rolling_mad.replace(0, np.nan))

        warm = prior_count < warmup_min
        p95_60d = mape.shift(1).rolling(60, min_periods=1).quantile(0.95)

        alert = np.where(warm, mape > p95_60d, z > config.z_threshold)
        mode = np.where(warm, "warmup", "robust")

        daily.loc[g.index, "robust_z"] = z.to_numpy()
        daily.loc[g.index, "baseline_mode"] = mode
        daily.loc[g.index, "alert_flag"] = np.where(pd.isna(alert), False, alert).astype(bool)

    return daily


def _load_verdicts(config: TenantConfig) -> pd.DataFrame:
    """Manuel verdict tablosu — rebuild'in ASLA silmediği ayrı bir CSV."""
    path = config.log_root / "verdicts.csv"
    if not path.exists():
        return pd.DataFrame(columns=["edas_id", "target_date", "horizon_day", "verdict_code"])
    return pd.read_csv(path)


def build_daily_scorecard(config: TenantConfig, window_days: int | None = None) -> dict:
    window_days = window_days or config.scorecard_rebuild_window_days
    if not config.monitoring_db.exists():
        return {"status": "no_monitoring_db"}

    con = duckdb.connect(str(config.monitoring_db))
    try:
        views = con.execute("SHOW TABLES").df()["name"].tolist()
        if "forecast_log_v" not in views or "actuals_log_v" not in views:
            return {"status": "views_missing"}

        hourly = _joined_hourly(con, window_days)
        if hourly.empty:
            return {"status": "no_joined_rows"}

        daily = _daily_agg(hourly)
        daily = _add_robust_z(daily, config)

        verdicts = _load_verdicts(config)
        if not verdicts.empty:
            daily = daily.merge(
                verdicts, on=["edas_id", "target_date", "horizon_day"], how="left"
            )
        else:
            daily["verdict_code"] = None

        daily["built_at"] = pd.Timestamp.now()

        con.register("daily_scorecard_df", daily)
        con.execute("CREATE OR REPLACE TABLE daily_scorecard AS SELECT * FROM daily_scorecard_df")
    finally:
        con.close()

    logging.getLogger(config.logger_name).info(
        f"[Scorecard] {len(daily)} satır (edas_id x target_date x horizon_day)")
    return {"status": "ok", "rows": len(daily)}


def latest_scorecard(config: TenantConfig, edas_id: str | None = None,
                      horizon: str | None = None) -> pd.DataFrame:
    edas_id = edas_id or config.edas_id
    horizon = horizon or config.headline_horizon
    if not config.monitoring_db.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        return con.execute(
            "SELECT * FROM daily_scorecard WHERE edas_id = ? AND horizon_day = ? "
            "ORDER BY target_date DESC",
            [edas_id, horizon],
        ).df()
    finally:
        con.close()


def window_report(config: TenantConfig, windows: tuple[int, ...] | None = None,
                   edas_id: str | None = None, horizon: str | None = None) -> dict:
    """7g: operasyonel sağlık | 30g: sistematik bias/mevsim geçişi | 365g: yapısal drift."""
    windows = windows or config.scorecard_windows
    df = latest_scorecard(config, edas_id, horizon)
    if df.empty:
        return {w: {"status": "no_data"} for w in windows}

    df = df.sort_values("target_date", ascending=False)
    metric_cols = [
        "mape", "wape", "rmse", "me",
        "mape_xgb", "mape_lgbm", "mape_cat", "mape_chronos",
        "mape_ens_raw", "mape_final", "mape_naive_lag168",
        "mape_night", "mape_morning", "mape_pv", "mape_evening",
        "temp_fcst_error", "ghi_fcst_error",
    ]
    report = {}
    for w in windows:
        sub = df.head(w)
        agg = {c: float(sub[c].mean()) if sub[c].notna().any() else None for c in metric_cols}
        agg["n_days"] = len(sub)
        agg["corrector_gain_bps"] = (
            (agg["mape_ens_raw"] - agg["mape_final"]) * 100
            if agg["mape_ens_raw"] is not None and agg["mape_final"] is not None else None
        )
        # Faz 2: model son `w` günde geçen haftayı kopyalamaktan kaç bps daha
        # iyi/kötü — pozitif=model naive'i geçiyor, negatif=model naive'den kötü
        # (07-12 Pazar'da ADM için muhtemelen negatifti, bu artık ölçülüyor).
        agg["vs_naive_lag168_bps"] = (
            (agg["mape_naive_lag168"] - agg["mape_final"]) * 100
            if agg.get("mape_naive_lag168") is not None and agg["mape_final"] is not None else None
        )
        agg["beats_naive_lag168_rate"] = (
            float(sub["beats_naive_lag168"].dropna().mean())
            if "beats_naive_lag168" in sub.columns and sub["beats_naive_lag168"].notna().any() else None
        )
        agg["alert_days"] = int(sub["alert_flag"].sum()) if "alert_flag" in sub.columns else None
        report[w] = agg
    return report


def check_alerts(config: TenantConfig, z_threshold: float | None = None) -> list[dict]:
    """En güncel target_date'in alert_flag=true satırlarını logs/alerts/<date>.json'a yazar."""
    df = latest_scorecard(config)
    if df.empty:
        return []

    latest_date = df["target_date"].max()
    today_rows = df[df["target_date"] == latest_date]
    alerts = today_rows[today_rows["alert_flag"]].to_dict(orient="records")

    if alerts:
        config.alerts_dir.mkdir(parents=True, exist_ok=True)
        date_str = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
        path = config.alerts_dir / f"{date_str}.json"
        path.write_text(json.dumps(alerts, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        logging.getLogger(config.logger_name).warning(f"[Alert] {len(alerts)} alarm -> {path}")

    return alerts
