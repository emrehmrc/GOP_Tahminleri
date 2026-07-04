"""
tab_veri_durumu.py — Sekme 1: Veri Güncelliği
================================================
ADM ve GDZ master serilerinin ne kadar güncel olduğunu gösterir:
son tarih, kaç gün geride, boşluk taraması, son 14 gün grafiği.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    MASTER_PARQUET, RAW_TARGET_COL, RAW_DATE_COL, RAW_HOUR_COL,
    import_gdz_ingest, render_freshness_card, render_pending_card, LIVE_DATA_DIR,
)
from src.data_scanner import get_ingestion_candidates


def _load_adm() -> pd.DataFrame:
    return pd.read_parquet(MASTER_PARQUET)


def _load_gdz() -> pd.DataFrame:
    gdz_mod = import_gdz_ingest()
    return pd.read_parquet(gdz_mod.GDZ_MASTER_PATH), gdz_mod.GDZ_RAW_TARGET_COL


def render():
    st.subheader("Veri Güncelliği")

    render_pending_card()

    adm_df = _load_adm()
    gdz_df, gdz_target_col = _load_gdz()

    col_adm, col_gdz = st.columns(2)

    with col_adm:
        st.markdown("### ADM (Aydem — Muğla/Denizli/Aydın)")
        adm_stats = render_freshness_card(
            "ADM", adm_df, RAW_DATE_COL, RAW_TARGET_COL, hour_col=RAW_HOUR_COL,
        )

    with col_gdz:
        st.markdown("### GDZ (Gediz — İzmir/Manisa)")
        gdz_stats = render_freshness_card(
            "GDZ", gdz_df, "Tarih", gdz_target_col, hour_col=None,
        )

    st.divider()
    st.markdown("### Son 14 Gün — Karşılaştırma")

    adm_recent = adm_df.copy()
    adm_recent["Datetime"] = pd.to_datetime(adm_recent[RAW_DATE_COL]) + pd.to_timedelta(adm_recent[RAW_HOUR_COL], unit="h")
    adm_recent = adm_recent.sort_values("Datetime").tail(14 * 24)

    gdz_recent = gdz_df.copy()
    gdz_recent["Datetime"] = pd.to_datetime(gdz_recent["Tarih"])
    gdz_recent = gdz_recent.sort_values("Datetime").tail(14 * 24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=adm_recent["Datetime"], y=adm_recent[RAW_TARGET_COL],
        name="ADM", mode="lines", yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=gdz_recent["Datetime"], y=gdz_recent[gdz_target_col],
        name="GDZ", mode="lines", yaxis="y2",
    ))
    fig.update_layout(
        yaxis=dict(title="ADM (MWh)"),
        yaxis2=dict(title="GDZ (MWh)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
