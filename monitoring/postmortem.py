"""monitoring/postmortem.py — Faz 2 2a-3 (2026-07-13): günlük post-mortem artefaktı.

"Günün Karnesi"nin veri hali — actual bir target_date için forecast_log_v'ye
girdikten sonra tek günün derinlemesine karnesini üretir: saatlik final tahmin
vs actual, per-model MAPE + en kötü 3 saat, naive lag168 benchmark farkı,
gün-tipi bağlamı, basit hava-tahmini-hata payı.

ADM+GDZ paylaşımlı (monitoring/scorecard.py ile aynı desen): TenantConfig alır,
`ui/`e bağımlı değil. Çıktı konumu (output/daily/<gün>/) çağıran tarafından
belirlenir (bkz. generate_postmortem.py) — bu modül path'lere dokunmaz, sadece
dict/markdown üretir.

Not (kapsam sınırlaması): "hava tahmini hatası payı" burada TAM perfect-prog
ayrıştırması DEĞİL (modeli gerçek hava ile yeniden koşturmuyor — bu, model
reload gerektiren ayrı, daha büyük bir iş). Burada sadece wx_temp_fcst/
wx_ghi_fcst'in gerçekleşenden ne kadar saptığı raporlanır — kaba bir "hava mı
suçlu, model mi suçlu" ön-işareti. Tam ayrıştırma Faz 4b'nin (diagnostic HTML)
kapsamına bırakıldı.
"""

from __future__ import annotations

import json

import duckdb
import numpy as np
import pandas as pd

from monitoring.scorecard import (
    DAY_TYPE_GROUPS,
    HOUR_BLOCKS,
    MODEL_PRED_COLS,
    _mape,
    _me,
)
from monitoring.tenant_config import TenantConfig

MODEL_LABELS = {
    "xgb": "XGB", "lgbm": "LGBM", "cat": "CAT", "chronos": "CHRONOS",
    "ens_raw": "Ensemble", "final": "Final",
}


def _query_day(con: duckdb.DuckDBPyConnection, target_date: str, horizon: str) -> pd.DataFrame:
    sql = """
        SELECT f.edas_id, f.target_ts, f.horizon_day, f.day_type, f.flag_holiday,
               f.flag_bridge, f.flag_ramadan,
               f.y_pred_xgb, f.y_pred_lgbm, f.y_pred_cat, f.y_pred_chronos,
               f.y_pred_ens_raw, f.y_pred_final,
               f.wx_temp_fcst, f.wx_ghi_fcst,
               a.y_actual, a.wx_temp_actual, a.wx_ghi_actual,
               a7.y_actual AS y_actual_lag168
        FROM forecast_log_v f
        INNER JOIN actuals_log_v a
          ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts
        LEFT JOIN actuals_log_v a7
          ON f.edas_id = a7.edas_id AND a7.target_ts = f.target_ts - INTERVAL '7 days'
        WHERE f.target_date = ? AND f.horizon_day = ? AND a.y_actual IS NOT NULL
        ORDER BY f.target_ts
    """
    return con.execute(sql, [target_date, horizon]).df()


def build_postmortem(config: TenantConfig, target_date: str, horizon: str | None = None) -> dict:
    """Tek gün, tek horizon için post-mortem dict'i. `status` alanı:
    "ok" | "no_monitoring_db" | "no_actuals" (actual henüz gelmemiş)."""
    horizon = horizon or config.headline_horizon
    if not config.monitoring_db.exists():
        return {"status": "no_monitoring_db", "edas_id": config.edas_id, "target_date": target_date}

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        views = con.execute("SHOW TABLES").df()["name"].tolist()
        if "forecast_log_v" not in views or "actuals_log_v" not in views:
            return {"status": "no_monitoring_db", "edas_id": config.edas_id, "target_date": target_date}
        day = _query_day(con, target_date, horizon)
    finally:
        con.close()

    if day.empty:
        return {"status": "no_actuals", "edas_id": config.edas_id, "target_date": target_date,
                "horizon_day": horizon}

    day = day.copy()
    day["hour"] = pd.to_datetime(day["target_ts"]).dt.hour
    day["hour_block"] = None
    for label, hrs in HOUR_BLOCKS.items():
        day.loc[day["hour"].isin(hrs), "hour_block"] = label

    actual = day["y_actual"]

    per_model_mape = {}
    for model, pred_col in MODEL_PRED_COLS.items():
        if pred_col in day.columns:
            per_model_mape[model] = _mape(day[pred_col], actual)

    final_pred = day["y_pred_final"]
    ape_final = np.abs((actual - final_pred) / (actual + 1e-10)) * 100
    worst_idx = ape_final.nlargest(3).index
    worst_hours = [
        {
            "hour": int(pd.to_datetime(day.loc[i, "target_ts"]).hour),
            "actual": float(actual.loc[i]),
            "pred_final": float(final_pred.loc[i]),
            "ape_pct": float(ape_final.loc[i]),
            "error_mwh": float(final_pred.loc[i] - actual.loc[i]),
        }
        for i in worst_idx
    ]

    mape_final = per_model_mape.get("final", np.nan)
    mape_naive = _mape(day["y_actual_lag168"], actual)
    vs_naive_bps = (
        (mape_naive - mape_final) * 100
        if not (np.isnan(mape_naive) or np.isnan(mape_final)) else None
    )

    hour_block_mape = {
        label: _mape(final_pred[day["hour_block"] == label], actual[day["hour_block"] == label])
        for label in HOUR_BLOCKS
    }

    raw_day_type = day["day_type"].mode().iat[0] if day["day_type"].notna().any() else None
    day_type_group = DAY_TYPE_GROUPS.get(raw_day_type, "ozel_gun") if raw_day_type else None

    wx_temp_actual = day["wx_temp_actual"]
    wx_ghi_actual = day["wx_ghi_actual"]
    weather = {
        "temp_fcst_error_mean": (
            float((day["wx_temp_fcst"] - wx_temp_actual).mean()) if wx_temp_actual.notna().any() else None
        ),
        "ghi_fcst_error_mean": (
            float((day["wx_ghi_fcst"] - wx_ghi_actual).mean()) if wx_ghi_actual.notna().any() else None
        ),
        "has_actual_weather": bool(wx_temp_actual.notna().any() or wx_ghi_actual.notna().any()),
    }

    return {
        "status": "ok",
        "edas_id": config.edas_id,
        "target_date": target_date,
        "horizon_day": horizon,
        "day_type": raw_day_type,
        "day_type_group": day_type_group,
        "flag_holiday": bool(day["flag_holiday"].iloc[0]),
        "flag_bridge": bool(day["flag_bridge"].iloc[0]) if "flag_bridge" in day.columns else False,
        "flag_ramadan": bool(day["flag_ramadan"].iloc[0]) if "flag_ramadan" in day.columns else False,
        "n_hours": int(len(day)),
        "per_model_mape": per_model_mape,
        "mape_naive_lag168": mape_naive if not np.isnan(mape_naive) else None,
        "vs_naive_lag168_bps": vs_naive_bps,
        "beats_naive_lag168": (
            bool(mape_final < mape_naive)
            if not (np.isnan(mape_final) or np.isnan(mape_naive)) else None
        ),
        "hour_block_mape": hour_block_mape,
        "worst_hours": worst_hours,
        "weather": weather,
    }


def render_postmortem_md(result: dict) -> str:
    if result["status"] != "ok":
        return (f"# Post-mortem {result['edas_id']} {result['target_date']}\n\n"
                f"Durum: {result['status']} — henüz veri yok.\n")

    lines = [
        f"# Post-mortem — {result['edas_id']} {result['target_date']} ({result['horizon_day']})",
        "",
        f"Gün tipi: **{result['day_type']}** ({result['day_type_group']})"
        + (" — TATİL" if result["flag_holiday"] else "")
        + (" — KÖPRÜ" if result["flag_bridge"] else "")
        + (" — RAMAZAN" if result["flag_ramadan"] else ""),
        "",
        "## Per-model MAPE",
        "",
    ]
    for model, mape in result["per_model_mape"].items():
        label = MODEL_LABELS.get(model, model)
        lines.append(f"- {label}: {mape:.2f}%" if not np.isnan(mape) else f"- {label}: n/a")

    mape_naive = result["mape_naive_lag168"]
    vs_bps = result["vs_naive_lag168_bps"]
    lines += ["", "## Naive lag168 (geçen hafta aynı saat) karşılaştırması", ""]
    if mape_naive is None:
        lines.append("Henüz 7 gün öncesi actual yok — naive benchmark hesaplanamadı.")
    else:
        beats = "GEÇTİ ✅" if result["beats_naive_lag168"] else "GEÇEMEDİ ⚠️"
        lines.append(f"- mape_naive_lag168: {mape_naive:.2f}%")
        lines.append(f"- vs_naive_lag168_bps: {vs_bps:+.0f} (pozitif=model naive'i geçiyor) — {beats}")

    lines += ["", "## Saat-bloğu MAPE (Final)", ""]
    for label, mape in result["hour_block_mape"].items():
        lines.append(f"- {label}: {mape:.2f}%" if not np.isnan(mape) else f"- {label}: n/a")

    lines += ["", "## En kötü 3 saat (Final)", ""]
    for wh in result["worst_hours"]:
        lines.append(
            f"- {wh['hour']:02d}:00 — APE={wh['ape_pct']:.1f}%  actual={wh['actual']:.0f} MWh  "
            f"pred={wh['pred_final']:.0f} MWh  hata={wh['error_mwh']:+.0f} MWh"
        )

    weather = result["weather"]
    lines += ["", "## Hava tahmini hata payı (ön-işaret, tam perfect-prog değil)", ""]
    if not weather["has_actual_weather"]:
        lines.append("Gerçekleşen hava verisi henüz yok.")
    else:
        if weather["temp_fcst_error_mean"] is not None:
            lines.append(f"- Sıcaklık tahmin hatası (ort.): {weather['temp_fcst_error_mean']:+.1f}°C")
        if weather["ghi_fcst_error_mean"] is not None:
            lines.append(f"- GHI tahmin hatası (ort.): {weather['ghi_fcst_error_mean']:+.0f} W/m²")

    return "\n".join(lines) + "\n"


def write_postmortem(config: TenantConfig, target_date: str, out_dir, horizon: str | None = None) -> dict:
    """`out_dir` altına postmortem_<edas_id>.md + .json yazar (dizin çağıran
    tarafından, tipik olarak `OUTPUT_DIR / "daily" / target_date`, verilir)."""
    from pathlib import Path
    out_dir = Path(out_dir)
    result = build_postmortem(config, target_date, horizon)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"postmortem_{config.edas_id}.md"
    json_path = out_dir / f"postmortem_{config.edas_id}.json"
    md_path.write_text(render_postmortem_md(result), encoding="utf-8")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result
