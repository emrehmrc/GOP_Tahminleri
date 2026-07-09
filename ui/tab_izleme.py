"""
tab_izleme.py — Faz 3 İzleme & Analiz Dashboard
================================================
Forecast vs actual, gün/model seçici, scorecard trend, tenant dropdown.
"""
from __future__ import annotations
import sys, os, json
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

UI_DIR = Path(__file__).parent
LIVE_DIR = UI_DIR.parent
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))
if str(LIVE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(LIVE_DIR / "src"))

import config_live as C

# ── DuckDB bağlantı yolu ───────────────────────────────────────────────────────
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
TENANT_DBS = {
    "ADM": LOCALAPPDATA / "adm_live_logs" / "monitoring.duckdb",
    "GDZ": LOCALAPPDATA / "gdz_live_logs" / "monitoring.duckdb",
}

MODEL_LABELS: dict[str, str] = {
    "Ensemble": "y_pred_final",
    "XGB": "y_pred_xgb",
    "LGBM": "y_pred_lgbm",
    "CAT": "y_pred_cat",
    "Chronos": "y_pred_chronos",
    "Ensemble Raw": "y_pred_ens_raw",
}

COLORS = px.colors.qualitative.Plotly


def _connect(edas: str) -> Optional[duckdb.DuckDBPyConnection]:
    db = TENANT_DBS.get(edas)
    if db is None or not db.exists():
        return None
    return duckdb.connect(str(db), read_only=True)


def _load_scorecard(con: duckdb.DuckDBPyConnection, edas: str,
                    days: int = 60) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    sql = """
        SELECT * FROM daily_scorecard
        WHERE edas_id = ? AND target_date >= ?
        ORDER BY target_date DESC
    """
    df = con.execute(sql, [edas, cutoff]).df()
    if not df.empty:
        df["target_date"] = pd.to_datetime(df["target_date"])
    return df


def _load_hourly(con: duckdb.DuckDBPyConnection, edas: str,
                 start: str, end: str) -> pd.DataFrame:
    sql = """
        SELECT f.*, a.y_actual, a.wx_temp_actual, a.wx_ghi_actual,
               a.data_quality_flag
        FROM forecast_log_v f
        LEFT JOIN actuals_log_v a
          ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts
        WHERE f.edas_id = ?
          AND f.target_date >= ? AND f.target_date <= ?
          AND a.y_actual IS NOT NULL
        ORDER BY f.target_ts
    """
    df = con.execute(sql, [edas, start, end]).df()
    if not df.empty:
        df["target_ts"] = pd.to_datetime(df["target_ts"])
        df["target_date"] = pd.to_datetime(df["target_date"])
        df["hour"] = df["target_ts"].dt.hour
        df["dow"] = df["target_ts"].dt.dayofweek
    return df


# ── Grafik fonksiyonlari ───────────────────────────────────────────────────────

def plot_forecast_vs_actual(df: pd.DataFrame, model_col: str,
                            model_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["target_ts"], y=df["y_actual"], mode="lines+markers",
        name="Gerçekleşen", line=dict(color=COLORS[1], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["target_ts"], y=df[model_col], mode="lines+markers",
        name=f"Tahmin ({model_label})", line=dict(color=COLORS[0], width=2, dash="dot"),
    ))
    fig.update_layout(
        title=f"Tahmin vs Gerçekleşen — {model_label}",
        xaxis_title="Tarih/Saat", yaxis_title="MWh",
        hovermode="x unified", height=400,
    )
    return fig


def plot_residuals(df: pd.DataFrame, model_col: str) -> go.Figure:
    residual = df["y_actual"] - df[model_col]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["target_ts"], y=residual,
        marker_color=np.where(residual >= 0, COLORS[0], COLORS[1]),
        opacity=0.7, name="Residual (Actual − Forecast)",
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(
        title="Saatlik Residual", xaxis_title="Tarih/Saat",
        yaxis_title="MWh", hovermode="x unified", height=250,
    )
    return fig


def plot_mape_trend(scorecard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label, c in [("mape", "Final MAPE", COLORS[0]),
                           ("mape_ens_raw", "Ensemble Raw", COLORS[2]),
                           ("mape_xgb", "XGB", COLORS[3]),
                           ("mape_chronos", "Chronos", COLORS[4])]:
        if col in scorecard.columns:
            fig.add_trace(go.Scatter(
                x=scorecard["target_date"], y=scorecard[col],
                mode="lines+markers", name=label, line=dict(color=c),
            ))
    fig.update_layout(
        title="Günlük MAPE Trendi",
        xaxis_title="Tarih", yaxis_title="MAPE (%)",
        hovermode="x unified", height=350,
    )
    return fig


def plot_hour_block_heatmap(df: pd.DataFrame) -> go.Figure:
    blocks = {"Gece (0-5)": (0, 6), "Sabah (6-9)": (6, 10),
              "PV (10-16)": (10, 17), "Akşam (17-21)": (17, 22),
              "Gece geç (22-23)": (22, 24)}
    rows = []
    for label, (h_start, h_end) in blocks.items():
        part = df[df["hour"].between(h_start, h_end - 1)]
        if part.empty:
            continue
        ape = np.abs((part["y_actual"] - part["y_pred_final"]) / (part["y_actual"] + 1e-10)) * 100
        me = (part["y_pred_final"] - part["y_actual"]).mean()
        rows.append({"Block": label, "MAPE": ape.mean(), "ME": me, "N": len(part)})
    block_df = pd.DataFrame(rows)
    if block_df.empty:
        return go.Figure()
    fig = px.bar(block_df, x="Block", y="MAPE", color="Block",
                 text=block_df["MAPE"].round(1).astype(str) + "%",
                 title="Saat Bloğu MAPE")
    fig.update_layout(showlegend=False, height=300)
    return fig


def plot_model_comparison_bar(scorecard: pd.DataFrame, latest: bool = True) -> go.Figure:
    if latest:
        row = scorecard.sort_values("target_date", ascending=False).iloc[0]
        data = {k: v for k, v in row.items()
                if k.startswith("mape_") and k != "mape_final" and isinstance(v, (int, float))}
        title = f"Son Gün Model MAPE ({row['target_date'].date()})"
    else:
        data = {"mape_xgb": scorecard["mape_xgb"].mean(),
                "mape_lgbm": scorecard["mape_lgbm"].mean(),
                "mape_cat": scorecard["mape_cat"].mean(),
                "mape_chronos": scorecard["mape_chronos"].mean(),
                "mape_ens_raw": scorecard["mape_ens_raw"].mean(),
                "mape_final": scorecard["mape_final"].mean()}
        title = "Ortalama Model MAPE (tüm aralık)"
    labels = {"mape_xgb": "XGB", "mape_lgbm": "LGBM", "mape_cat": "CAT",
              "mape_chronos": "Chronos", "mape_ens_raw": "Ensemble Raw",
              "mape_final": "Final"}
    names, vals = [], []
    for k, v in data.items():
        if v is not None and not np.isnan(v):
            names.append(labels.get(k, k))
            vals.append(v)
    fig = go.Figure(go.Bar(x=names, y=vals, text=[f"{v:.1f}%" for v in vals],
                           marker_color=COLORS[:len(names)]))
    fig.update_layout(title=title, yaxis_title="MAPE (%)", height=300)
    return fig


def plot_corrector_gain(scorecard: pd.DataFrame, window: int = 30) -> go.Figure:
    sub = scorecard.sort_values("target_date").tail(window).copy()
    sub["corrector_gain"] = sub["mape_ens_raw"] - sub["mape_final"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["target_date"], y=sub["corrector_gain"],
        mode="lines+markers", name="Corrector Gain (bps)",
        line=dict(color=COLORS[5]),
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(
        title=f"Corrector Katkısı (son {window} gün)",
        xaxis_title="Tarih", yaxis_title="MAPE İyileşme (pp)",
        hovermode="x unified", height=250,
    )
    return fig


# ── Ana render ──────────────────────────────────────────────────────────────────

def render():
    st.header("📊 İzleme & Analiz")

    # Tenant seçici
    available = [e for e, p in TENANT_DBS.items() if p.exists()]
    if not available:
        st.warning("Hiçbir tenant için monitoring.duckdb bulunamadı. "
                   "Önce run_daily.py veya backtest çalıştırın.")
        return

    edas = st.selectbox("EDAŞ", available, key="izleme_edas")

    con = _connect(edas)
    if con is None:
        st.error(f"{edas} için monitoring.duckdb bağlantısı kurulamadı.")
        return

    try:
        # Varsayılan değerler
        today = date.today()
        day_count = st.sidebar.slider("Gösterilecek gün sayısı", 7, 365, 30, key="izleme_days")
        model_label = st.sidebar.selectbox(
            "Model", list(MODEL_LABELS.keys()), index=0, key="izleme_model",
        )
        model_col = MODEL_LABELS[model_label]
        horizon = st.sidebar.multiselect("Ufuk", ["T+1", "T+2"], default=["T+1", "T+2"], key="izleme_horizon")

        # Scorecard yükle
        scorecard = _load_scorecard(con, edas, days=day_count)
        if scorecard.empty:
            st.info(f"{edas} için scorecard verisi yok (< {day_count} gün geriye dönük).")
            con.close()
            return

        if horizon:
            scorecard = scorecard[scorecard["horizon_day"].isin(horizon)]

        scorecard_latest = scorecard["target_date"].max()
        default_start = scorecard_latest - timedelta(days=min(day_count, 14))

        # Gün seçici
        min_date = scorecard["target_date"].min().date()
        max_date = scorecard["target_date"].max().date()
        date_range = st.date_input(
            "Tarih aralığı",
            value=(default_start.date(), scorecard_latest.date()),
            min_value=min_date, max_value=max_date,
            key="izleme_daterange",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range[0], date_range[1]
        else:
            start_d, end_d = min_date, max_date

        start_str = start_d.isoformat()
        end_str = end_d.isoformat()

        # Hourly veri yükle
        hourly = _load_hourly(con, edas, start_str, end_str)
        if not hourly.empty and horizon:
            hourly = hourly[hourly["horizon_day"].isin(horizon)]

        # Filter scorecard to date range
        sc = scorecard[(scorecard["target_date"] >= pd.Timestamp(start_d))
                       & (scorecard["target_date"] <= pd.Timestamp(end_d))]

        # Scorecard metric özeti (üst satır)
        if not sc.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            last = sc.sort_values("target_date", ascending=False).iloc[0]
            avg = sc.mean(numeric_only=True)
            col1.metric("MAPE (son gün)", f"{last['mape']:.2f}%" if pd.notna(last['mape']) else "—",
                        delta=f"{last['mape'] - avg.get('mape', 0):+.2f}pp")
            col2.metric("WAPE (son gün)", f"{last['wape']:.2f}%" if pd.notna(last['wape']) else "—")
            col3.metric("RMSE (son gün)", f"{last['rmse']:.0f}" if pd.notna(last['rmse']) else "—")
            col4.metric("ME (son gün)", f"{last['me']:.0f}" if pd.notna(last['me']) else "—",
                        delta_color="inverse")
            col5.metric("N gün", len(sc))

            st.caption(f"Seçili aralık: {start_d} → {end_d}   |   "
                       f"EDAŞ: {edas}   |   Ufuk: {horizon if horizon else 'Tüm'}")

        # Tab layout
        tab_fc, tab_mape, tab_models, tab_hourly, tab_raw = st.tabs(
            ["📈 Tahmin vs Actual", "📉 MAPE Trend", "📊 Model Karşılaştırma",
             "🕐 Saatlik Detay", "📋 Ham Veri"]
        )

        with tab_fc:
            if not hourly.empty and model_col in hourly.columns:
                st.plotly_chart(plot_forecast_vs_actual(hourly, model_col, model_label),
                                use_container_width=True)
                st.plotly_chart(plot_residuals(hourly, model_col), use_container_width=True)
            else:
                st.info(f"Seçili aralıkta {edas} için model verisi yok veya "
                        f"'{model_col}' forecast_log'da mevcut değil.")

        with tab_mape:
            if not sc.empty:
                st.plotly_chart(plot_mape_trend(sc), use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(plot_hour_block_heatmap(hourly) if not hourly.empty
                                    else go.Figure(), use_container_width=True)
                with col_b:
                    st.plotly_chart(plot_corrector_gain(sc, 30), use_container_width=True)

                # Alert tablosu
                alerts = sc[sc.get("alert_flag", False)]
                if not alerts.empty:
                    with st.expander(f"⚠ Alert (<color-red>{len(alerts)} gün)</color>)"):
                        st.dataframe(
                            alerts[["target_date", "horizon_day", "mape", "robust_z",
                                    "baseline_mode", "verdict_code"]]
                            .sort_values("target_date", ascending=False),
                            use_container_width=True,
                        )
            else:
                st.info("Seçili aralıkta scorecard verisi yok. Tahmin edilen günlere "
                        "actual geldikçe scorecard otomatik türetilir.")

        with tab_models:
            if not sc.empty:
                col_c, col_d = st.columns(2)
                with col_c:
                    st.plotly_chart(plot_model_comparison_bar(sc, latest=True),
                                    use_container_width=True)
                with col_d:
                    st.plotly_chart(plot_model_comparison_bar(sc, latest=False),
                                    use_container_width=True)

                # Ensemble weight trend (meta_w_* alanlarından)
                if not hourly.empty and "meta_w_xgb" in hourly.columns:
                    hourly_fc = hourly[hourly["meta_w_xgb"].notna()].copy()
                    if not hourly_fc.empty:
                        hourly_fc["run_date"] = hourly_fc["target_ts"].dt.date
                        weights = hourly_fc.groupby("run_date")[
                            ["meta_w_xgb", "meta_w_lgbm", "meta_w_cat", "meta_w_chronos"]
                        ].first().reset_index()
                        weights = weights.melt(id_vars="run_date", var_name="Model", value_name="Ağırlık")
                        fig_weights = px.line(
                            weights, x="run_date", y="Ağırlık", color="Model",
                            title="Meta Ağırlık Trendi",
                            markers=True,
                        )
                        fig_weights.update_layout(height=300)
                        st.plotly_chart(fig_weights, use_container_width=True)

        with tab_hourly:
            if not hourly.empty:
                # Saatlik tablo
                display_cols = ["target_ts", "horizon_day", "y_actual", model_col,
                                "y_pred_final", "day_type", "flag_holiday"]
                display_cols = [c for c in display_cols if c in hourly.columns]
                if "data_quality_flag" in hourly.columns:
                    display_cols.append("data_quality_flag")

                df_display = hourly[display_cols].copy()
                df_display["hr"] = df_display["target_ts"].dt.hour
                # Residual
                if model_col in hourly.columns:
                    df_display["Residual"] = (hourly["y_actual"] - hourly[model_col]).round(1)
                    df_display["APE%"] = (
                        np.abs((hourly["y_actual"] - hourly[model_col])
                               / (hourly["y_actual"] + 1e-10) * 100).round(2)
                    )

                st.dataframe(
                    df_display.sort_values("target_ts", ascending=False)
                    .head(200).reset_index(drop=True),
                    use_container_width=True,
                    height=400,
                )
                st.caption("Son 200 satır")
            else:
                st.info("Seçili aralıkta saatlik veri yok.")

        with tab_raw:
            if not sc.empty:
                st.dataframe(sc.sort_values("target_date", ascending=False)
                             .head(50).reset_index(drop=True),
                             use_container_width=True, height=400)
                st.caption("Scorecard — son 50 satır")

            # Window report
            try:
                from src.scorecard import window_report
                wr = window_report(edas_id=edas, horizon="T+2")
                if wr:
                    st.subheader("Pencere Raporu (T+2)")
                    wr_df = pd.DataFrame(wr).T
                    wr_df.index.name = "Gün"
                    st.dataframe(wr_df.round(3), use_container_width=True)
            except Exception as e:
                st.caption(f"Pencere raporu alınamadı: {e}")

    finally:
        con.close()
