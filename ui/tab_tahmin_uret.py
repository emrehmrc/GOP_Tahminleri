"""
tab_tahmin_uret.py — Sekme 3: Tahmin Üret
============================================
ADM için 9 adımlı günlük tahmin pipeline'ının tamamını (ingest → ... →
deliver → forecast_log/scorecard → Excel rapor → diagnostic HTML → email)
st.status() içinde adım-adım çalıştırır (senkron — tek kullanıcı, günde bir
kez tetiklenen araç için en basit yaklaşım). UI artık run_daily.py CLI ile
aynı kapsamı kapsıyor — pipeline'ı tetiklemenin TEK yeri burası olsun diye
(07/08/09 daha önce sadece CLI'de çalışıyordu, UI'dan tetiklenince Excel
rapor / diagnostic HTML / email hiç üretilmiyordu — bkz. run_daily.py).

GDZ için pipeline'ı SUBPROCESS olarak çalıştırır (gdz talep/live/run_daily.py),
in-process DEĞİL: ADM bu dosyanın modül-seviyesinde bare `from run_context import
...` / `from forecast_logger import ...` yapıyor — bunlar sys.modules'a ADM'nin
versiyonuyla cache'leniyor. GDZ'nin `live/src/` altında AYNI isimde ama farklı
içerikli (config_live_gdz'ye bağlı) kendi run_context.py/forecast_logger.py'si var;
aynı process'te çalıştırılsa Python sys.modules'daki ADM versiyonunu döndürür —
GDZ kodu yanlış şemalı fonksiyonları çağırmış olur (sessiz çakışma). Subprocess
bu riski komple ortadan kaldırır (ayrı process = ayrı sys.modules), sıfır kod
değişikliğiyle GDZ'nin zaten çalışan CLI'ını (run_daily.py) yeniden kullanır.

run_daily.py'nin run_step() yardımcısı KASITLI olarak burada tekrar
kullanılmıyor: o CLI/log odaklı (hata -> logla + yeniden fırlat), bu UI'ın
politikası farklı (hata -> yakala + göster + durdur). İki farklı hata
politikasını tek fonksiyona zorlamak yanlış olurdu.
"""

import json
import time
import logging
import traceback

import streamlit as st

from common import (
    import_pipeline_step,
    run_gdz_pipeline_async, gdz_stdout_path, gdz_summary_path,
)
# common.py LIVE_DIR/src'i sys.path'e ekledi → run_context doğrudan import edilebilir.
from run_context import start_run, archive_models, prune_archive, write_summary
from forecast_logger import write_forecast_log, rebuild_duckdb_views, backup_logs_zip
from scorecard import build_daily_scorecard, check_alerts

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

                    # 06 başarılı → forecast_log yaz + DuckDB view + günlük yedek + scorecard/alarm.
                    if step_name == "06_deliver":
                        try:
                            st.session_state["forecast_steps"]["forecast_log"] = write_forecast_log(ctx)
                            rebuild_duckdb_views()
                            backup_logs_zip()
                        except Exception as log_err:
                            log.warning(f"Forecast log yazımı hatası (teslimi etkilemez): {log_err}")
                            st.warning(f"⚠ Forecast log yazılamadı: {log_err}")

                        try:
                            st.session_state["forecast_steps"]["scorecard"] = build_daily_scorecard()
                            alerts = check_alerts()
                            st.session_state["forecast_steps"]["alerts"] = {"count": len(alerts)}
                            if alerts:
                                st.warning(f"⚠ {len(alerts)} kötü-gün alarmı — logs/alerts/ altına bak")
                        except Exception as sc_err:
                            log.warning(f"Scorecard/alarm hatası (teslimi etkilemez): {sc_err}")
                            st.warning(f"⚠ Scorecard hesaplanamadı: {sc_err}")

                        # Adım 07 — STLF LIVE RAPOR (Excel, ADM+GDZ 5 tablo)
                        try:
                            result07 = import_pipeline_step("07_report_excel").run()
                            st.session_state["forecast_steps"]["07_report"] = result07
                            st.write(f"✓ **STLF Rapor (Excel)** — {result07.get('file', '?')}")
                            log.info(f"══ 07_report_excel TAMAM | {result07}")
                        except Exception as rep_err:
                            log.warning(f"STLF Rapor hatası (teslimi etkilemez): {rep_err}")
                            st.warning(f"⚠ STLF Rapor oluşturulamadı: {rep_err}")

                        # Adım 08 — STLF DIAGNOSTIC HTML (Chart.js dashboard)
                        try:
                            result08 = import_pipeline_step("08_diagnostic_html").run()
                            st.session_state["forecast_steps"]["08_diagnostic"] = result08
                            st.write(f"✓ **Diagnostic HTML** — {result08.get('file', '?')}")
                            log.info(f"══ 08_diagnostic_html TAMAM | {result08}")
                        except Exception as diag_err:
                            log.warning(f"Diagnostic HTML hatası (teslimi etkilemez): {diag_err}")
                            st.warning(f"⚠ Diagnostic HTML oluşturulamadı: {diag_err}")

                        # Adım 09 — Email rapor (SMTP ayarlanmamışsa sessizce atlanır)
                        try:
                            result09 = import_pipeline_step("09_email_report").run()
                            st.session_state["forecast_steps"]["09_email"] = result09
                            if result09.get("status") == "ok":
                                st.write(f"✓ **Email** — gönderildi: {', '.join(result09.get('to', []))}")
                            else:
                                st.write(f"⏭ **Email** — {result09.get('reason', result09.get('status'))}")
                            log.info(f"══ 09_email_report TAMAM | {result09}")
                        except Exception as mail_err:
                            log.warning(f"Email rapor hatası (teslimi etkilemez): {mail_err}")
                            st.warning(f"⚠ Email gönderilemedi: {mail_err}")
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


def _tail_lines(path, n=15) -> str:
    """Log dosyasının son n satırını döndür (dosya henüz yoksa boş)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _render_gdz():
    st.divider()
    st.markdown("### GDZ Günlük Tahmin Çalıştır")
    st.caption("Ayrı process'te çalışır (gdz talep/live/run_daily.py) — ADM panelinden bağımsız.")

    gdz_skip_ingest = st.checkbox("Adım 01 (ingest) atla", value=False, key="gdz_skip_ingest")
    gdz_skip_weather = st.checkbox("Adım 02 (hava) atla", value=False, key="gdz_skip_weather")
    gdz_target_date = st.text_input("Teslim günü (YYYY-MM-DD, boş=otomatik)", value="", key="gdz_target_date")

    st.session_state.setdefault("gdz_forecast_running", False)

    gdz_run_clicked = st.button(
        "▶ GDZ Tahmini Başlat", type="primary", key="gdz_forecast_run_btn",
        disabled=st.session_state["gdz_forecast_running"],
    )

    if gdz_run_clicked:
        st.session_state["gdz_forecast_running"] = True

        args = []
        if gdz_skip_ingest:
            args.append("--skip-ingest")
        if gdz_skip_weather:
            args.append("--skip-weather")
        if gdz_target_date:
            args += ["--target", gdz_target_date]

        proc = run_gdz_pipeline_async(args)
        stdout_path = gdz_stdout_path()
        summary_path = gdz_summary_path()

        with st.status("GDZ pipeline çalışıyor (subprocess, ~5-8dk — Chronos dahil)...", expanded=True) as status:
            placeholder = st.empty()
            while proc.poll() is None:
                tail = _tail_lines(stdout_path)
                placeholder.code(tail or "Başlatılıyor...", language="text")
                time.sleep(2)
            # process bitti — son hali göster
            placeholder.code(_tail_lines(stdout_path, n=30) or "(log yok)", language="text")

            returncode = proc.returncode
            if returncode != 0:
                status.update(label="HATA — GDZ pipeline başarısız", state="error")
                st.error(f"✗ GDZ subprocess hata ile bitti (exit code {returncode})")
                st.code(_tail_lines(stdout_path, n=40), language="text")
            else:
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception as e:
                    summary = None
                    st.warning(f"Summary okunamadı: {e}")

                if summary and summary.get("status") == "ok":
                    status.update(label="✓ GDZ pipeline tamamlandı", state="complete")
                    for step_name, result in summary.get("steps", {}).items():
                        st.write(f"✓ **{step_name}** — {result}")
                    deliver = summary.get("steps", {}).get("06_deliver")
                    if deliver and deliver.get("output_file"):
                        st.success(f"Teslim dosyası: {deliver['output_file']}")
                else:
                    status.update(label="⚠ GDZ pipeline durum belirsiz", state="error")
                    st.warning(f"Summary status: {summary.get('status') if summary else '?'}")
                    st.json(summary)

        st.session_state["gdz_forecast_running"] = False


def render():
    st.subheader("Tahmin Üret")
    _render_adm()
    _render_gdz()
