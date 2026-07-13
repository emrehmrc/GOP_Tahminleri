"""
tab_izleme.py — Faz 3 Izleme & Analiz Dashboard
================================================
Forecast vs actual, gun/model secici, scorecard trend, tenant dropdown.
"""
from __future__ import annotations
import sys, os
from pathlib import Path
from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

UI_DIR = Path(__file__).parent
LIVE_DIR = UI_DIR.parent
if str(LIVE_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DIR))
if str(LIVE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(LIVE_DIR / "src"))

import config_live as C

# ── DuckDB baglanti yolu ───────────────────────────────────────────────────────
LOCALAPPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
TENANT_DBS = {
    "ADM": LOCALAPPDATA / "adm_live_logs" / "monitoring.duckdb",
    "GDZ": LOCALAPPDATA / "gdz_live_logs" / "monitoring.duckdb",
}

MODEL_LABELS: dict[str, str] = {
    "Ensemble": "y_pred_final",
    "XGB":      "y_pred_xgb",
    "LGBM":     "y_pred_lgbm",
    "CAT":      "y_pred_cat",
    "Chronos":  "y_pred_chronos",
    "Ensemble Raw": "y_pred_ens_raw",
}

COLORS = px.colors.qualitative.Plotly
# ensure enough colors
while len(COLORS) < 10:
    COLORS = COLORS + COLORS


# ── Veri yukleme ───────────────────────────────────────────────────────────────

def _connect(edas: str) -> Optional[duckdb.DuckDBPyConnection]:
    db = TENANT_DBS.get(edas)
    if db is None or not db.exists():
        return None
    return duckdb.connect(str(db), read_only=True)


def _has_table(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    try:
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        return name in tables
    except Exception:
        return False


def _load_scorecard(con: duckdb.DuckDBPyConnection, edas: str,
                    days: int = 60) -> pd.DataFrame:
    if not _has_table(con, "daily_scorecard"):
        return pd.DataFrame()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    try:
        df = con.execute(
            "SELECT * FROM daily_scorecard "
            "WHERE edas_id = ? AND target_date >= ? "
            "ORDER BY target_date DESC",
            [edas, cutoff],
        ).df()
        if not df.empty:
            df["target_date"] = pd.to_datetime(df["target_date"])
        return df
    except Exception:
        return pd.DataFrame()


def _load_hourly(con: duckdb.DuckDBPyConnection, edas: str,
                 start: str, end: str) -> pd.DataFrame:
    """forecast_log_v + actuals_log_v LEFT JOIN. actuals olmayan satirlar da gelir
    (y_actual = NaN) — grafiklerde tahmin cizgisi gorunur, karsilastirma yok."""
    if not _has_table(con, "forecast_log_v"):
        return pd.DataFrame()
    try:
        df = con.execute(
            "SELECT f.*, a.y_actual, a.wx_temp_actual, a.wx_ghi_actual, "
            "       a.data_quality_flag "
            "FROM forecast_log_v f "
            "LEFT JOIN actuals_log_v a "
            "  ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts "
            "WHERE f.edas_id = ? "
            "  AND f.target_date >= ? AND f.target_date <= ? "
            "ORDER BY f.target_ts",
            [edas, start, end],
        ).df()
        if not df.empty:
            df["target_ts"]   = pd.to_datetime(df["target_ts"])
            df["target_date"]  = pd.to_datetime(df["target_date"])
            df["hour"]         = df["target_ts"].dt.hour
            df["dow"]          = df["target_ts"].dt.dayofweek
        return df
    except Exception:
        return pd.DataFrame()


def _forecast_date_range(con: duckdb.DuckDBPyConnection, edas: str) -> tuple[date, date]:
    """forecast_log_v icin min/max target_date."""
    if not _has_table(con, "forecast_log_v"):
        return date.today(), date.today()
    try:
        r = con.execute(
            "SELECT MIN(target_date), MAX(target_date) "
            "FROM forecast_log_v WHERE edas_id = ?",
            [edas],
        ).fetchone()
        if r and r[0] and r[1]:
            from datetime import date as dt_date
            return (
                r[0] if isinstance(r[0], dt_date) else pd.Timestamp(r[0]).date(),
                r[1] if isinstance(r[1], dt_date) else pd.Timestamp(r[1]).date(),
            )
    except Exception:
        pass
    return date.today(), date.today()


def _missing_forecast_dates(hourly: pd.DataFrame, start_d: date, end_d: date) -> list:
    """Secili aralikta+ufukta HIC satiri olmayan takvim gunlerini dondur.
    forecast_log'da bir gun icin kayit yoksa (log kaybi / pipeline atlanmasi)
    grafik o gunu sessizce atlayip komsu gunleri birbirine baglar — bu da
    veri kaybini 'garip bir egri' gibi gosterip UI hatasi zannettirir. Bu
    fonksiyon o bosluklari acikca listeler."""
    all_days = pd.date_range(start_d, end_d, freq="D").date
    if hourly.empty:
        return list(all_days)
    present = set(pd.to_datetime(hourly["target_date"]).dt.date.unique())
    return [d for d in all_days if d not in present]


def _reindex_hourly_gaps(df: pd.DataFrame, start_d: date, end_d: date) -> pd.DataFrame:
    """Eksik saatleri NaN satir olarak ekler ki Plotly cizgiyi (connectgaps=False
    varsayilaniyla) kirsin, komsu gunleri duz cizgiyle birlestirmesin. Sadece
    TEK bir ufuk seciliyken guvenli cagrilmali (target_ts o zaman tekil olur)."""
    if df.empty:
        return df
    full_idx = pd.date_range(start_d, pd.Timestamp(end_d) + pd.Timedelta(hours=23), freq="h")
    out = df.set_index("target_ts").reindex(full_idx)
    out.index.name = "target_ts"
    return out.reset_index()


def _forecast_edge_date_for_horizon(con: duckdb.DuckDBPyConnection, edas: str,
                                     horizon: list[str], agg: str, fallback: date) -> date:
    """Secili ufuk(lar)a gore forecast_log_v MIN/MAX(target_date). T+1 ile T+2
    farkli gunlerde baslar/biter (issue+1 vs issue+2 — bkz. issue_date farki) —
    genel fc_min/fc_max'i (tum ufuklarin karisimi) varsayilan Baslangic/Bitis
    olarak kullanmak, her gun sahte bir bosluk uyarisi verir (secili ufuktaki
    ilk/son gun, digerinin dolu oldugu bir gune denk gelir). Bu yuzden
    varsayilanlari SADECE secili ufkun kendi ucuna gore hesapliyoruz."""
    if not horizon or not _has_table(con, "forecast_log_v"):
        return fallback
    try:
        placeholders = ",".join(["?"] * len(horizon))
        r = con.execute(
            f"SELECT {agg}(target_date) FROM forecast_log_v "
            f"WHERE edas_id = ? AND horizon_day IN ({placeholders})",
            [edas, *horizon],
        ).fetchone()
        if r and r[0]:
            return r[0] if isinstance(r[0], date) else pd.Timestamp(r[0]).date()
    except Exception:
        pass
    return fallback


def _available_models(edas: str) -> dict[str, str]:
    """edas'a gore mevcut model kolonlarini dondur (GDZ'de bazi kolonlar NaN)."""
    con = _connect(edas)
    if con is None:
        return MODEL_LABELS
    try:
        if not _has_table(con, "forecast_log_v"):
            return MODEL_LABELS
        df = con.execute(
            "SELECT * FROM forecast_log_v WHERE edas_id = ? LIMIT 1",
            [edas],
        ).df()
        result = {}
        for label, col in MODEL_LABELS.items():
            if col in df.columns and not (col in df.columns and df[col].isna().all()):
                result[label] = col
        return result if result else MODEL_LABELS
    except Exception:
        return MODEL_LABELS
    finally:
        con.close()


# ── Grafik fonksiyonlari ───────────────────────────────────────────────────────

def _has_actuals(df: pd.DataFrame) -> bool:
    """df'de en az bir y_actual var mi?"""
    return "y_actual" in df.columns and df["y_actual"].notna().any()


def plot_forecast_vs_actual(df: pd.DataFrame, model_col: str,
                            model_label: str) -> go.Figure:
    fig = go.Figure()
    if _has_actuals(df):
        fig.add_trace(go.Scatter(
            x=df["target_ts"], y=df["y_actual"], mode="lines+markers",
            name="Gerceklesen", line=dict(color=COLORS[1], width=2),
        ))
    if model_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df["target_ts"], y=df[model_col], mode="lines+markers",
            name=f"Tahmin ({model_label})",
            line=dict(color=COLORS[0], width=2, dash="dot"),
        ))
    fig.update_layout(
        title=f"Tahmin vs Gerceklesen — {model_label}" +
              ("" if _has_actuals(df) else " (actuals henuz yok)"),
        xaxis_title="Tarih/Saat", yaxis_title="MWh",
        hovermode="x unified", height=400,
    )
    return fig


def plot_residuals(df: pd.DataFrame, model_col: str) -> go.Figure:
    if model_col not in df.columns or not _has_actuals(df):
        return go.Figure()
    residual = df["y_actual"] - df[model_col]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["target_ts"], y=residual,
        marker_color=np.where(residual >= 0, COLORS[0], COLORS[1]),
        opacity=0.7, name="Residual (Actual - Forecast)",
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(
        title="Saatlik Residual",
        xaxis_title="Tarih/Saat", yaxis_title="MWh",
        hovermode="x unified", height=250,
    )
    return fig


def _scalar(val):
    """pandas scalar -> native Python (seri/nan safe)."""
    if val is None:
        return None
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except (TypeError, ValueError):
        return val


def plot_mape_trend(scorecard: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, label, c in [("mape", "Final MAPE", COLORS[0]),
                           ("mape_ens_raw", "Ensemble Raw", COLORS[2]),
                           ("mape_xgb", "XGB", COLORS[3]),
                           ("mape_chronos", "Chronos", COLORS[4])]:
        if col in scorecard.columns:
            fig.add_trace(go.Scatter(
                x=scorecard["target_date"], y=scorecard[col],
                mode="lines+markers", name=label,
                line=dict(color=c),
            ))
    fig.update_layout(
        title="Gunluk MAPE Trendi",
        xaxis_title="Tarih", yaxis_title="MAPE (%)",
        hovermode="x unified", height=350,
    )
    return fig


def plot_hour_block_heatmap(df: pd.DataFrame) -> go.Figure:
    blocks = {"Gece (0-5)": (0, 6), "Sabah (6-9)": (6, 10),
              "PV (10-16)": (10, 17), "Aksam (17-21)": (17, 22),
              "Gece gec (22-23)": (22, 24)}
    rows = []
    for label, (h_start, h_end) in blocks.items():
        part = df[df["hour"].between(h_start, h_end - 1)]
        if part.empty or "y_pred_final" not in part.columns:
            continue
        ape = np.abs(
            (part["y_actual"] - part["y_pred_final"]) / (part["y_actual"] + 1e-10)
        ) * 100
        me = (part["y_pred_final"] - part["y_actual"]).mean()
        rows.append({"Block": label, "MAPE": ape.mean(), "ME": me, "N": len(part)})
    block_df = pd.DataFrame(rows)
    if block_df.empty:
        return go.Figure()
    fig = px.bar(
        block_df, x="Block", y="MAPE", color="Block",
        text=block_df["MAPE"].round(1).astype(str) + "%",
        title="Saat Blogu MAPE",
    )
    fig.update_layout(showlegend=False, height=300)
    return fig


def plot_model_comparison_bar(scorecard: pd.DataFrame) -> go.Figure:
    cols = [c for c in ["mape_xgb", "mape_lgbm", "mape_cat",
                         "mape_chronos", "mape_ens_raw", "mape_final"]
            if c in scorecard.columns]
    if cols:
        row = scorecard.sort_values("target_date", ascending=False).iloc[0]
        data = {c: _scalar(row[c]) for c in cols if _scalar(row[c]) is not None}
        title = f"Son Gun Model MAPE ({row['target_date'].date()})"
    else:
        data = {}
        title = "Model MAPE (veri yok)"
    labels = {"mape_xgb": "XGB", "mape_lgbm": "LGBM", "mape_cat": "CAT",
              "mape_chronos": "Chronos", "mape_ens_raw": "Ensemble Raw",
              "mape_final": "Final"}
    names, vals = [], []
    for k, v in data.items():
        if v is not None:
            names.append(labels.get(k, k))
            vals.append(v)
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        text=[f"{v:.1f}%" for v in vals],
        marker_color=COLORS[:len(names)],
    ))
    fig.update_layout(title=title, yaxis_title="MAPE (%)", height=300)
    return fig


def plot_corrector_gain(scorecard: pd.DataFrame, window: int = 30) -> go.Figure:
    if "mape_ens_raw" not in scorecard.columns or "mape_final" not in scorecard.columns:
        return go.Figure()
    sub = scorecard.sort_values("target_date").tail(window).copy()
    sub["corrector_gain"] = sub["mape_ens_raw"] - sub["mape_final"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["target_date"], y=sub["corrector_gain"],
        mode="lines+markers", name="Corrector Gain (pp)",
        line=dict(color=COLORS[5]),
    ))
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(
        title=f"Corrector Katkisi (son {window} gun)",
        xaxis_title="Tarih", yaxis_title="MAPE Iyilesme (pp)",
        hovermode="x unified", height=250,
    )
    return fig


# ── Ana render ─────────────────────────────────────────────────────────────────

def render():
    st.header("Izleme & Analiz")

    # Tenant secici
    available = [e for e, p in TENANT_DBS.items() if p.exists()]
    if not available:
        st.warning("Hicbir tenant icin monitoring.duckdb bulunamadi.")
        return

    edas = st.selectbox("EDAS", available, key="izleme_edas")

    con = _connect(edas)
    if con is None:
        st.error(f"{edas} icin monitoring.duckdb baglantisi kurulamadi.")
        return

    try:
        # Sidebar ayarlari
        day_count = st.sidebar.slider(
            "Gosterilecek gun sayisi", 7, 365, 30, key="izleme_days")
        available_models = _available_models(edas)
        if not available_models:
            st.warning(f"{edas} icin forecast_log_v bos.")
            return
        model_label = st.sidebar.selectbox(
            "Model", list(available_models.keys()), index=0, key="izleme_model")
        model_col = available_models[model_label]
        # Kullanici tercihi: iki tenant'ta da her zaman T+2, sabit/surekli.
        horizon_default = ["T+2"]
        if st.session_state.get("izleme_horizon_edas_seen") != edas:
            st.session_state["izleme_horizon"] = horizon_default
            st.session_state["izleme_horizon_edas_seen"] = edas
        horizon = st.sidebar.multiselect(
            "Ufuk", ["T+1", "T+2"], key="izleme_horizon")

        # Scorecard
        scorecard = _load_scorecard(con, edas, days=day_count)

        # Hourly + date range defaults (forecast_log_v kaynakli)
        today = date.today()
        fc_min, fc_max = _forecast_date_range(con, edas)
        if not scorecard.empty:
            sc_latest = pd.to_datetime(scorecard["target_date"]).max().date()
        else:
            sc_latest = fc_max

        # Baslangic/Bitis: secili ufkun KENDI min/max'i (bkz.
        # _forecast_edge_date_for_horizon docstring) — genel fc_min/fc_max
        # (tum ufuklarin karisimi) kullanilirsa sahte 'bu gun icin veri yok'
        # uyarisi cikar (T+1/T+2 farkli gunlerde baslar/biter).
        horizon_min = _forecast_edge_date_for_horizon(con, edas, horizon, "MIN", fc_min)
        horizon_max = _forecast_edge_date_for_horizon(con, edas, horizon, "MAX", fc_max)
        default_start = max(horizon_min, sc_latest - timedelta(days=min(day_count, 14)))
        default_end = horizon_max
        min_date = fc_min
        max_date = fc_max

        # DIKKAT: st.date_input `key` verildiginde `value=` SADECE ilk renderda
        # kullanilir; sonraki rerun'larda widget kendi session_state[key]'ini
        # okur ve `value=` sessizce yok sayilir. fc_max her gun yeni tahminle
        # ilerledigi icin (dunku run T+2 olarak yarini, bugunku T+2 olarak
        # yarinin-yarinini ekler) bu yuzden Bitis tarihi eski gunde donup
        # kalir, yeni gun hic gorunmez. fc_max/ufuk (tenant+horizon bazli)
        # degistiginde state'i widget olusmadan ONCE elle resetleyerek
        # pencereyi ilerletiyoruz.
        #
        # DIKKAT 2 (bulundu 2026-07-10, ADM'i StreamlitAPIException ile
        # cokertiyordu): key'ler ONCEDEN tenant'a gore ayri degildi
        # ("izleme_start"/"izleme_end" HER edas icin AYNI session_state
        # anahtariydi) — GDZ'de secilen bir tarih (ornegin fc_min=06-28),
        # ADM'e gecince ADM'in kendi min_date'inin (06-29) ALTINDA kalip
        # "value must lie between min/max" hatasi firlatiyordu. Key'leri
        # tenant'a gore ayirdik + asagida DEFANSIF clamp ekledik: reset
        # mantigi bir kombinasyonu atlarsa bile widget'a ASLA sinir-disi
        # deger gitmez.
        start_key = f"izleme_start__{edas}"
        end_key = f"izleme_end__{edas}"
        fc_seen_key = f"izleme_fc_seen__{edas}__{','.join(horizon)}"
        fc_seen_val = (horizon_min, horizon_max)
        if st.session_state.get(fc_seen_key) != fc_seen_val:
            st.session_state[start_key] = default_start
            st.session_state[end_key] = default_end
            st.session_state[fc_seen_key] = fc_seen_val

        # Defansif clamp: hangi sebeple olursa olsun (eski/paylasilan state,
        # gozden kacan kombinasyon) session_state'te [min_date,max_date]
        # disinda bir deger varsa widget kurulmadan ONCE ice cek — Streamlit
        # `value` argumanini key varken yok saydigi icin bunu WIDGET
        # OLUSMADAN ONCE session_state'i elle duzelterek yapmak zorundayiz.
        for k, fallback in ((start_key, default_start), (end_key, default_end)):
            cur = st.session_state.get(k, fallback)
            st.session_state[k] = min(max(cur, min_date), max_date)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_d = st.date_input(
                "Baslangic",
                min_value=min_date, max_value=max_date, key=start_key)
        with col_d2:
            end_d = st.date_input(
                "Bitis",
                min_value=min_date, max_value=max_date, key=end_key)

        if start_d > end_d:
            start_d, end_d = end_d, start_d

        start_str = start_d.isoformat()
        end_str = end_d.isoformat()

        # Hourly yukle
        hourly = _load_hourly(con, edas, start_str, end_str)
        if not hourly.empty and horizon:
            hourly = hourly[hourly["horizon_day"].isin(horizon)]

        # Scorecard filtrele
        if not scorecard.empty:
            scorecard["target_date"] = pd.to_datetime(scorecard["target_date"])
            sc = scorecard[
                (scorecard["target_date"] >= pd.Timestamp(start_d))
                & (scorecard["target_date"] <= pd.Timestamp(end_d))
            ]
            if horizon:
                sc = sc[sc["horizon_day"].isin(horizon)]
        else:
            sc = pd.DataFrame()

        # ── Ust metrikler ──
        # Ek bilgi: forecast_log_v'deki en son target_date (tahmin guncellemesi)
        fc_last_str = "?"
        try:
            fc_last = con.execute(
                "SELECT MAX(target_date) FROM forecast_log_v WHERE edas_id = ?", [edas]
            ).fetchone()[0]
            if fc_last:
                fc_last_str = str(fc_last) if isinstance(fc_last, date) else str(pd.Timestamp(fc_last).date())
        except Exception:
            pass
        # Actual gelmis ama tahminle joinlenmis kac gun var
        if not sc.empty:
            n_with_actual = int(sc["mape"].notna().sum())
        else:
            n_with_actual = 0

        if not sc.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            last = sc.sort_values("target_date", ascending=False).iloc[0]
            last_mape = _scalar(last["mape"])
            avg_mape = _scalar(sc["mape"].mean(numeric_only=True))
            hz_label = ", ".join(horizon) if horizon else "Tum"
            col1.metric(
                f"{hz_label} MAPE (son gun)",
                f"{last_mape:.2f}%" if last_mape is not None else "-",
                delta=(f"{last_mape - avg_mape:+.2f}pp"
                       if last_mape is not None and avg_mape is not None else None),
            )
            col2.metric(f"{hz_label} WAPE (son gun)",
                        f"{_scalar(last['wape']):.2f}%" if _scalar(last.get("wape")) is not None else "-")
            col3.metric(f"{hz_label} RMSE (son gun)",
                        f"{_scalar(last['rmse']):.0f}" if _scalar(last.get("rmse")) is not None else "-")
            col4.metric(f"{hz_label} ME (son gun)",
                        f"{_scalar(last['me']):.0f}" if _scalar(last.get("me")) is not None else "-",
                        delta_color="inverse")
            col5.metric("Actual gun", n_with_actual)
            st.caption(
                f"Secili aralik: {start_d} -> {end_d}   |   Son tahmin: {fc_last_str}   "
                f"|   EDAS: {edas}   |   Ufuk: {horizon if horizon else 'Tum'}"
            )
        elif not hourly.empty:
            st.caption(
                f"Secili aralik: {start_d} -> {end_d}   |   Son tahmin: {fc_last_str}   "
                f"|   EDAS: {edas}   |   Ufuk: {horizon if horizon else 'Tum'}"
            )
        else:
            st.info(
                f"{edas} icin secili aralikta ({start_d} -> {end_d}) "
                "veri bulunamadi. forecast_log_v veya actuals_log_v bos olabilir. "
                "Once run_daily.py calistirildigindan emin olun."
            )
            return

        # ── Tab'lar ──
        tab_fc, tab_mape, tab_models, tab_hourly_detail, tab_raw = st.tabs(
            ["Tahmin vs Actual", "MAPE Trend", "Model Karsilastirma",
             "Saatlik Detay", "Ham Veri"]
        )

        # Tab 1: Forecast vs Actual
        with tab_fc:
            missing_days = _missing_forecast_dates(hourly, start_d, end_d)
            if missing_days:
                missing_str = ", ".join(d.isoformat() for d in missing_days)
                st.warning(
                    f"⚠ Bu gunler icin secili ufukta ({', '.join(horizon) if horizon else 'Tum'}) "
                    f"forecast_log'da HIC kayit yok: {missing_str}. "
                    "UI hatasi degil — o gunun tahmini pipeline'da hic uretilmemis/kaydedilmemis "
                    "(log kaybi). Grafikte bu gunler bos birakilir, komsu gunlere baglanmaz."
                )
            if not hourly.empty and model_col in hourly.columns:
                horizon_str = ", ".join(horizon) if horizon else "Tum"
                plot_df = (
                    _reindex_hourly_gaps(hourly, start_d, end_d)
                    if len(horizon) == 1 else hourly
                )
                fig_fc = plot_forecast_vs_actual(plot_df, model_col, model_label)
                fig_fc.update_layout(
                    title=f"Tahmin vs Gerceklesen — {model_label} ({horizon_str})"
                )
                st.plotly_chart(fig_fc, use_container_width=True)
                if _has_actuals(hourly):
                    st.plotly_chart(
                        plot_residuals(plot_df, model_col),
                        use_container_width=True,
                    )
                else:
                    st.info("Bu aralikta actuals henuz gelmemis — "
                            "sadece tahmin cizgisi gosteriliyor. "
                            "Actuals D+1'de 01_ingest ile yuklenir.")
            else:
                st.info(
                    f"Secili aralikta {edas} icin saatlik veri yok veya "
                    f"'{model_col}' kolonu forecast_log_v'de mevcut degil."
                )

        # Tab 2: MAPE Trend
        with tab_mape:
            if not sc.empty:
                horizon_str = ", ".join(horizon) if horizon else "Tum"
                fig_mape = plot_mape_trend(sc)
                fig_mape.update_layout(title=f"Gunluk MAPE Trendi ({horizon_str})")
                st.plotly_chart(fig_mape, use_container_width=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig_hb = plot_hour_block_heatmap(hourly) if not hourly.empty else go.Figure()
                    fig_hb.update_layout(title=f"Saat Blogu MAPE ({horizon_str})")
                    st.plotly_chart(fig_hb, use_container_width=True)
                with col_b:
                    fig_cg = plot_corrector_gain(sc, 30)
                    fig_cg.update_layout(title=f"Corrector Katkisi — son 30 gun ({horizon_str})")
                    st.plotly_chart(fig_cg, use_container_width=True)

                alerts = sc[sc.get("alert_flag") == True] if "alert_flag" in sc.columns else pd.DataFrame()
                if not alerts.empty:
                    with st.expander(f"UYARI: {len(alerts)} gun alert tespit edildi"):
                        st.dataframe(
                            alerts[["target_date", "horizon_day", "mape", "robust_z",
                                    "baseline_mode", "verdict_code"]]
                            .sort_values("target_date", ascending=False),
                            use_container_width=True,
                        )
            else:
                st.info(
                    "Scorecard bos — forecast_log_v'ye forecast, actuals_log_v'ye "
                    "y_actual yazildiktan sonra scorecard otomatik turetilecek. "
                    "Orn: python -c \"from scorecard import build_daily_scorecard; build_daily_scorecard()\""
                )

        # Tab 3: Model Karsilastirma
        with tab_models:
            if not sc.empty:
                col_c, col_d = st.columns(2)
                with col_c:
                    fig_mc = plot_model_comparison_bar(sc)
                    horizon_str = ", ".join(horizon) if horizon else "Tum"
                    # Başlığı güncelle (varsa mevcut tarih korunsun)
                    old_title = fig_mc.layout.title.text if hasattr(fig_mc.layout, 'title') else ""
                    fig_mc.update_layout(title=f"{old_title} ({horizon_str})" if old_title else f"Model MAPE ({horizon_str})")
                    st.plotly_chart(fig_mc, use_container_width=True)
                with col_d:
                    # Tum aralik ortalamasi
                    avg_metrics = {}
                    for c in ["mape_xgb", "mape_lgbm", "mape_cat",
                               "mape_chronos", "mape_ens_raw", "mape_final"]:
                        v = _scalar(sc[c].mean())
                        if v is not None:
                            avg_metrics[c] = v
                    fig = go.Figure(go.Bar(
                        x=[v.split("_", 1)[1].upper() for v in avg_metrics.keys()],
                        y=list(avg_metrics.values()),
                        text=[f"{v:.1f}%" for v in avg_metrics.values()],
                        marker_color=COLORS[:len(avg_metrics)],
                    ))
                    fig.update_layout(
                        title="Ortalama Model MAPE (tum aralik)",
                        yaxis_title="MAPE (%)", height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Meta agirlik trend
            if not hourly.empty and "meta_w_xgb" in hourly.columns:
                hourly_w = hourly[hourly["meta_w_xgb"].notna()].copy()
                if not hourly_w.empty:
                    hourly_w["run_date"] = hourly_w["target_ts"].dt.date
                    wcols = [c for c in ["meta_w_xgb", "meta_w_lgbm",
                                          "meta_w_cat", "meta_w_chronos"]
                             if c in hourly_w.columns]
                    if wcols:
                        wdf = hourly_w.groupby("run_date")[wcols].first().reset_index()
                        wdf = wdf.melt(id_vars="run_date", var_name="Model",
                                       value_name="Agirlik")
                        fig_w = px.line(
                            wdf, x="run_date", y="Agirlik", color="Model",
                            title="Meta Agirlik Trendi", markers=True,
                        )
                        fig_w.update_layout(height=300)
                        st.plotly_chart(fig_w, use_container_width=True)

        # Tab 4: Saatlik Detay
        with tab_hourly_detail:
            if not hourly.empty:
                display_cols = [
                    "target_ts", "horizon_day", "y_actual", model_col,
                    "day_type", "flag_holiday",
                ]
                display_cols = [c for c in display_cols if c in hourly.columns]
                if "data_quality_flag" in hourly.columns:
                    display_cols.append("data_quality_flag")

                df_display = hourly[display_cols].copy()
                df_display["hr"] = df_display["target_ts"].dt.hour
                if model_col in hourly.columns and "y_actual" in hourly.columns:
                    df_display["Residual"] = (
                        hourly["y_actual"] - hourly[model_col]
                    ).round(1)
                    df_display["APE%"] = (
                        np.abs(
                            (hourly["y_actual"] - hourly[model_col])
                            / (hourly["y_actual"] + 1e-10)
                        ) * 100
                    ).round(2)

                st.dataframe(
                    df_display.sort_values("target_ts", ascending=False)
                    .head(200).reset_index(drop=True),
                    use_container_width=True, height=400,
                )
                st.caption("Son 200 satir")
            else:
                st.info("Secili aralikta saatlik veri yok.")

        # Tab 5: Ham Veri
        with tab_raw:
            if not sc.empty:
                st.dataframe(
                    sc.sort_values("target_date", ascending=False)
                    .head(50).reset_index(drop=True),
                    use_container_width=True, height=400,
                )
                st.caption("Scorecard — son 50 satir")
            else:
                st.info("Scorecard verisi yok.")

            # Window report
            if not sc.empty:
                try:
                    from src.scorecard import window_report
                    wr = window_report(edas_id=edas, horizon="T+2")
                    if wr:
                        st.subheader("Pencere Raporu (T+2)")
                        wr_df = pd.DataFrame(wr).T
                        wr_df.index.name = "Gun"
                        st.dataframe(wr_df.round(3), use_container_width=True)
                except Exception as e:
                    st.caption(f"Pencere raporu alinamadi: {e}")

    finally:
        con.close()
