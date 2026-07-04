"""
tab_tahmin_uret.py — Sekme 3: Tahmin Üret
============================================
ADM için 6 adımlı günlük tahmin pipeline'ını st.status() içinde adım-adım
çalıştırır (senkron — tek kullanıcı, günde bir kez tetiklenen araç için
en basit yaklaşım). GDZ için henüz canlı tahmin pipeline'ı yok — placeholder.

run_daily.py'nin run_step() yardımcısı KASITLI olarak burada tekrar
kullanılmıyor: o CLI/log odaklı (hata -> logla + yeniden fırlat), bu UI'ın
politikası farklı (hata -> yakala + göster + durdur). İki farklı hata
politikasını tek fonksiyona zorlamak yanlış olurdu.
"""

import time
import logging
import traceback

import streamlit as st

from common import import_pipeline_step
# common.py LIVE_DIR/src'i sys.path'e ekledi → run_context doğrudan import edilebilir.
from run_context import start_run, archive_models, prune_archive, write_summary

log = logging.getLogger("adm_live")  # handler'ları start_run() kurar

STEPS = [
    ("01_ingest_actual", "Gerçekleşme verisi ingest ediliyor"),
    ("02_fetch_weather", "Hava tahmini çekiliyor"),
    ("03_build_features", "Feature matrisi kuruluyor"),
    ("04_predict_48h", "Modeller eğitiliyor + tahmin (yavaş, ~birkaç dk)"),
    ("05_postprocess", "Post-process uygulanıyor"),
    ("06_deliver", "Müşteri dosyası yazılıyor"),
]


def _render_adm():
    st.markdown("### ADM Günlük Tahmin Çalıştır")

    skip_ingest = st.checkbox("Adım 01 (ingest) atla", value=False, key="adm_skip_ingest")
    skip_weather = st.checkbox("Adım 02 (hava) atla", value=False, key="adm_skip_weather")
    target_date = st.text_input("Teslim günü (YYYY-MM-DD, boş=otomatik)", value="", key="adm_target_date")

    st.session_state.setdefault("forecast_running", False)
    st.session_state.setdefault("forecast_steps", {})
    st.session_state.setdefault("forecast_error", None)

    run_clicked = st.button(
        "▶ Tahmini Başlat", type="primary", key="adm_forecast_run_btn",
        disabled=st.session_state["forecast_running"],
    )

    if run_clicked:
        st.session_state["forecast_running"] = True
        st.session_state["forecast_steps"] = {}
        st.session_state["forecast_error"] = None

        skip_map = {"01_ingest_actual": skip_ingest, "02_fetch_weather": skip_weather}

        # Run kimliği + paylaşılan dosya loglaması (UI da artık loglar ve summary yazar).
        ctx = start_run(target_date=target_date or None)

        with st.status("Tahmin pipeline çalışıyor...", expanded=True) as status:
            for step_name, label in STEPS:
                if skip_map.get(step_name, False):
                    st.write(f"⏭ {label} — atlandı")
                    log.info(f"{step_name} atlandı (UI)")
                    continue
                st.write(f"**{label}**...")
                log.info(f"══ {step_name} BAŞLIYOR ══")
                try:
                    mod = import_pipeline_step(step_name)
                    kwargs = {"target_date": target_date or None} if step_name == "06_deliver" else {}
                    t0 = time.time()
                    result = mod.run(**kwargs)
                    elapsed = time.time() - t0
                    st.session_state["forecast_steps"][step_name] = result
                    st.write(f"✓ Tamam ({elapsed:.0f}s) — {result}")
                    log.info(f"══ {step_name} TAMAM ({elapsed:.0f}s) | {result}")

                    # 04 başarılı → o run'ın modelleri diske yazıldı, arşivle.
                    if step_name == "04_predict_48h":
                        try:
                            archive_models(ctx)
                            prune_archive()
                        except Exception as arch_err:
                            log.warning(f"Model arşivleme hatası (tahmin etkilenmez): {arch_err}")
                            st.warning(f"⚠ Model arşivlenemedi: {arch_err}")
                except Exception as e:
                    st.session_state["forecast_error"] = traceback.format_exc()
                    log.error(f"══ {step_name} HATA\n{traceback.format_exc()}")
                    write_summary(ctx, st.session_state["forecast_steps"], "error")
                    status.update(label=f"HATA: {step_name}", state="error")
                    st.error(f"✗ {step_name} başarısız: {e}")
                    st.exception(e)
                    st.session_state["forecast_running"] = False
                    st.stop()
            status.update(label="✓ Pipeline tamamlandı", state="complete")

        write_summary(ctx, st.session_state["forecast_steps"], "ok")
        st.session_state["forecast_running"] = False

    if st.session_state.get("forecast_error"):
        st.error("Son çalıştırmada hata oluştu:")
        st.code(st.session_state["forecast_error"])

    deliver_result = st.session_state.get("forecast_steps", {}).get("06_deliver")
    if deliver_result:
        st.success(f"Teslim dosyası: {deliver_result.get('output_file', '?')}")


def _render_gdz():
    st.divider()
    st.markdown("### GDZ Günlük Tahmin Çalıştır")
    st.info("🚧 Yakında — GDZ için canlı tahmin pipeline'ı henüz hazır değil (feature/model çalışması devam ediyor).")
    st.button("▶ Tahmini Başlat", disabled=True, key="gdz_forecast_btn_disabled")


def render():
    st.subheader("Tahmin Üret")
    _render_adm()
    _render_gdz()
