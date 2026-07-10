"""
tab_veri_yukleme.py — Sekme 2: Veri Yükleme
==============================================
LIVE_DATA_DIR / DD.MM / subfolder yapısından otomatik tarama,
önizleme, onay ve commit. Her gün ayrı ayrı veya toplu işlenebilir.
"""

import json
from datetime import date
from pathlib import Path

import streamlit as st

from common import (
    LIVE_DATA_DIR, MASTER_PARQUET,
    import_pipeline_step, import_gdz_ingest,
    get_pending_for_network,
    run_gdz_pipeline_async, gdz_summary_path,
)
from src.data_scanner import find_csv_for_date, scan_available_days


def render():
    st.subheader("Veri Yükleme")

    net = st.radio("Ağ seç", ["ADM (Aydem)", "GDZ (Gediz)"], horizontal=True)
    net_key = "ADM" if net.startswith("ADM") else "GDZ"
    region = "aydem" if net_key == "ADM" else "gediz"

    pending_days = get_pending_for_network(region)
    all_available = scan_available_days(LIVE_DATA_DIR, region)

    if not all_available:
        st.info(f"📂 **{LIVE_DATA_DIR}** içinde {region} CSV bulunamadı.")
        return

    if pending_days:
        n = len(pending_days)
        st.success(f"📥 **{n} bekleyen gün** tespit edildi: "
                   f"{pending_days[0]} … {pending_days[-1]}")
    else:
        all_dates = [d for d in all_available if d <= date.today()]
        if all_dates:
            st.info("✅ Tüm mevcut veriler master'a eklenmiş durumda.")
            latest = max(all_dates)
            st.caption(f"Son işlenmemiş veri: {latest} (master'da var mı kontrol et)")
        else:
            st.info("Mevcut veri bulunamadı.")

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date_str = st.selectbox(
            "İngest edilecek gün",
            options=[str(d) for d in sorted(all_available, reverse=True)],
            index=0,
            key=f"{net_key}_date_select",
        )
    target_date = date.fromisoformat(selected_date_str)

    with col2:
        preview_btn = st.button("🔍 Önizle", key=f"{net_key}_preview", use_container_width=True)

    preview_key = f"{net_key}_prev"
    preview_info_key = f"{net_key}_prev_info"

    if preview_btn or st.session_state.get(preview_key) is not None:
        csv_path = find_csv_for_date(LIVE_DATA_DIR, target_date, region)
        if csv_path is None:
            st.error(f"{target_date} için CSV bulunamadı.")
            st.session_state[preview_key] = None
            st.session_state[preview_info_key] = None
        else:
            try:
                if net_key == "ADM":
                    mod = import_pipeline_step("01_ingest_actual")
                else:
                    mod = import_gdz_ingest()
                raw = mod.load_source_csv(csv_path)
                validated = mod.validate(raw)
                st.session_state[preview_key] = validated
                st.session_state[preview_info_key] = {
                    "path": str(csv_path),
                    "date": str(target_date),
                }
            except Exception as e:
                st.session_state[preview_key] = None
                st.session_state[preview_info_key] = None
                st.error(f"Önizleme başarısız: {e}")

    if st.session_state.get(preview_key) is not None:
        vdf = st.session_state[preview_key]
        info = st.session_state[preview_info_key]
        is_fresh = (info["date"] == str(target_date))

        if is_fresh:
            mean_val = vdf["ADM_Dağıtılan_Enerji_(MWh)" if net_key == "ADM" else "GDZ- Dağıtılan Enerji (MWh)"].mean()
            st.caption(
                f"📄 {info['path']}  |  "
                f"{len(vdf)} satır  |  "
                f"Ortalama: {mean_val:.0f} MWh  |  "
                f"Tarih: {vdf['Tarih'].iloc[0].date()}"
            )
            st.dataframe(vdf, use_container_width=True, height=300)

            auto_build = st.checkbox(
                "İngest sonrası feature matrisini kur (Adım 03)",
                value=True, key=f"{net_key}_build",
            )

            if st.button("✅ Onayla ve Ekle", type="primary", key=f"{net_key}_commit"):
                _do_commit(net_key, region, target_date, auto_build)
                st.session_state[preview_key] = None
                st.session_state[preview_info_key] = None
                st.rerun()
        else:
            st.info("Seçilen gün değişti — önizlemeyi tekrarlayın.")
            st.session_state[preview_key] = None

    st.divider()
    st.markdown("**Toplu işlem**")
    if pending_days and st.button(f"📥 Tümünü sırayla işle ({len(pending_days)} gün)",
                                   key=f"{net_key}_batch"):
        auto_build = st.checkbox(
            "Feature build ile", value=True, key=f"{net_key}_batch_build",
        )
        success_count = 0
        for pd_ in pending_days:
            try:
                _do_commit(net_key, region, pd_, auto_build)
                success_count += 1
                st.success(f"✓ {pd_}")
            except Exception as e:
                st.error(f"✗ {pd_}: {e}")
                break
        st.success(f"Toplam {success_count}/{len(pending_days)} gün işlendi.")
        st.rerun()


def _do_commit(net_key: str, region: str, target_date: date, auto_build: bool):
    """Ingest + opsiyonel feature build."""
    if net_key == "ADM":
        mod = import_pipeline_step("01_ingest_actual")
        result = mod.run(target_date=target_date, source_name=region)
    else:
        mod = import_gdz_ingest()
        csv_path = find_csv_for_date(LIVE_DATA_DIR, target_date, region)
        if csv_path is None:
            raise FileNotFoundError(f"{target_date} için CSV yok")
        result = mod.run(csv_path=str(csv_path))

    rows = result.get("rows_added", result.get("sources", {}).get("aydem", {}).get("rows_added", 0))
    st.success(f"✓ {rows} satır eklendi — {target_date}")

    if net_key == "ADM" and auto_build:
        with st.spinner("Feature matrisi kuruluyor (Adım 03)..."):
            step03 = import_pipeline_step("03_build_features")
            feat_result = step03.run()
        st.info(
            f"✓ Feature matrisi: {feat_result['n_rows']} satır, "
            f"{feat_result['n_features']} feature"
        )
    elif net_key == "GDZ" and auto_build:
        # GDZ'nin pipeline'ı ayrı process olarak çağrılır (bare `run_context`/
        # `forecast_logger` importları ADM ile isim çakışıyor — bkz. tab_tahmin_uret.py
        # docstring'i). 01 zaten manuel yapıldı (yukarıda), 02'yi tekrar çekme —
        # ADM'nin "sadece 03'ü çalıştır" davranışıyla simetrik: --skip-ingest
        # --skip-weather --dry-run, 01-03'te durur (04+ tetiklemez).
        with st.spinner("Feature matrisi kuruluyor (GDZ Adım 03, subprocess)..."):
            proc = run_gdz_pipeline_async(["--skip-ingest", "--skip-weather", "--dry-run"])
            proc.wait()
        if proc.returncode != 0:
            st.error(f"✗ GDZ feature build başarısız (exit code {proc.returncode})")
        else:
            try:
                summary = json.loads(gdz_summary_path().read_text(encoding="utf-8"))
                feat_result = summary.get("steps", {}).get("03_features", {})
                st.info(
                    f"✓ GDZ Feature matrisi: {feat_result.get('n_rows', '?')} satır, "
                    f"{feat_result.get('n_features', '?')} feature"
                )
            except Exception as e:
                st.warning(f"Summary okunamadı ama işlem tamamlandı: {e}")
