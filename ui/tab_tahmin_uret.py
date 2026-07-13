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
import forecast_adjustment

from dashboard_common import (
    import_pipeline_step,
    run_gdz_pipeline_async, gdz_stdout_path, gdz_summary_path,
)
# common.py LIVE_DIR/src'i sys.path'e ekledi → run_context doğrudan import edilebilir.
from run_context import start_run, archive_models, prune_archive, write_summary
from forecast_logger import write_forecast_log, rebuild_duckdb_views, backup_logs_zip, reconcile
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
                        # Ortak STLF raporu e-posta onayini beklemez. ADM tahmini
                        # olusur olusmaz ADM sheet'i yazilir; GDZ kendi pipeline'i
                        # bittiginde ayni dosyanin ikinci sheet'ini gunceller.
                        try:
                            report_result = import_pipeline_step("07_report_excel").run(
                                target_date=result.get("target_date"), edas="ADM",
                            )
                            st.session_state["forecast_steps"]["07_report"] = report_result
                            st.write(f"✓ **STLF LIVE raporu** — {report_result.get('file', '?')}")
                            log.info(f"══ 07_report_excel TAMAM | {report_result}")
                        except Exception as report_err:
                            log.warning(f"STLF raporu guncellenemedi (teslimi etkilemez): {report_err}")
                            st.warning(f"⚠ STLF LIVE raporu güncellenemedi: {report_err}")

                        try:
                            st.session_state["forecast_steps"]["forecast_log"] = write_forecast_log(ctx)
                            rebuild_duckdb_views()
                            backup_logs_zip()
                        except Exception as log_err:
                            log.warning(f"Forecast log yazımı hatası (teslimi etkilemez): {log_err}")
                            st.warning(f"⚠ Forecast log yazılamadı: {log_err}")

                        try:
                            reconcile_result = reconcile()
                            st.session_state["forecast_steps"]["reconcile"] = reconcile_result
                            n_healed = reconcile_result.get("heal", {}).get("n_healed", 0)
                            if n_healed:
                                st.info(f"🩹 Arşivden {n_healed} eksik/kısmi hücre onarıldı")
                            n_gaps = len(reconcile_result.get("gaps", []))
                            if n_gaps:
                                st.warning(f"⚠ {n_gaps} hücre gerçek kayıp (arşiv de yok) — logs/gaps/ altına bak")
                        except Exception as heal_err:
                            log.warning(f"Reconcile (heal+tamlık) hatası (teslimi etkilemez): {heal_err}")
                            st.warning(f"⚠ Reconcile başarısız: {heal_err}")

                        try:
                            st.session_state["forecast_steps"]["scorecard"] = build_daily_scorecard()
                            alerts = check_alerts()
                            st.session_state["forecast_steps"]["alerts"] = {"count": len(alerts)}
                            if alerts:
                                st.warning(f"⚠ {len(alerts)} kötü-gün alarmı — logs/alerts/ altına bak")
                        except Exception as sc_err:
                            log.warning(f"Scorecard/alarm hatası (teslimi etkilemez): {sc_err}")
                            st.warning(f"⚠ Scorecard hesaplanamadı: {sc_err}")

                        # Adım 08 — STLF DIAGNOSTIC HTML (Chart.js dashboard)
                        try:
                            result08 = import_pipeline_step("08_diagnostic_html").run()
                            st.session_state["forecast_steps"]["08_diagnostic"] = result08
                            st.write(f"✓ **Diagnostic HTML** — {result08.get('file', '?')}")
                            log.info(f"══ 08_diagnostic_html TAMAM | {result08}")
                        except Exception as diag_err:
                            log.warning(f"Diagnostic HTML hatası (teslimi etkilemez): {diag_err}")
                            st.warning(f"⚠ Diagnostic HTML oluşturulamadı: {diag_err}")

                        # Adım 09 — Email: burada OTOMATIK ATILMAZ. Pipeline
                        # diagnostic'te biter, "onay bekliyor" durumuna geçer.
                        # Email yalnızca aşağıdaki onay panelinden "Müşteriye
                        # Gönder" ile tetiklenir (bkz. _render_approval_panel).
                        st.info("📋 Tahmin üretildi — diagnostic'i inceleyip "
                                "aşağıdan **Müşteriye Gönder** ile onaylayın.")
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


def _render_approval_panel():
    """Müşteri teslimi — insan onayı bölümü.

    Pipeline artık email'i otomatik atmıyor. Panel hem ADM hem GDZ'nin BUGÜN
    çalışıp çalışmadığını ve hedef tarihlerinin eşleşip eşleşmediğini
    (check_readiness, diskteki <tarih>_summary.json'lardan — session_state'e
    bağlı değil) kontrol eder; "Müşteriye Gönder" yalnızca ikisi de hazırsa
    aktif olur (kullanıcı talebi: ADM+GDZ ikisi üretilmeden gönderilmesin).
    Başarılı gönderim bir işaret dosyası yazar (aynı gün ikinci gönderimi
    engeller, "🔄 Tekrar gönder" ile bilinçli olarak aşılabilir).
    """
    email_mod = import_pipeline_step("09_email_report")
    readiness = email_mod.check_readiness()
    adm_r, gdz_r = readiness["adm"], readiness["gdz"]

    # Ne ADM ne GDZ bugün çalışmadıysa panel gösterilecek bir şey yok.
    if not adm_r["ran_today"] and not gdz_r["ran_today"]:
        return

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if adm_r["ran_today"]:
            st.success(f"✓ ADM hazır — teslim: {adm_r['target_date']}")
        else:
            st.warning("⏳ ADM bugün henüz çalıştırılmadı")
    with col2:
        if gdz_r["ran_today"]:
            st.success(f"✓ GDZ hazır — teslim: {gdz_r['target_date']}")
        else:
            st.warning("⏳ GDZ bugün henüz çalıştırılmadı")

    if adm_r["ran_today"] and gdz_r["ran_today"] and not readiness["target_match"]:
        st.error(f"⚠ Hedef tarihler uyuşmuyor — ADM: {adm_r['target_date']}, "
                 f"GDZ: {gdz_r['target_date']}. Gönderilemez.")

    # ADM+GDZ ikisi de çalışmış ve tarihleri eşleşmiş olsa bile, email'e
    # gidecek 5 dosyadan biri (rapor/diagnostic) üretilirken sessizce
    # başarısız olmuş olabilir (07/08 adımları hata yutar) — bu durumda
    # readiness["ready"] False kalır ama SEBEP burada gösterilmezse buton
    # neden pasif anlaşılmaz. missing_files listesini açıkça göster.
    if readiness.get("missing_files"):
        st.error("⚠ Şu dosyalar eksik/üretilememiş, gönderilemez:\n\n"
                  + "\n".join(f"- {f}" for f in readiness["missing_files"]))

    if not readiness["ready"]:
        st.info("⏳ Onay bekliyor — ADM ve GDZ ikisi de bugün, aynı hedef tarihle bitince aktif olur.")
        st.button("🏢 MRC İçi Paylaş", key="internal_send_btn", disabled=True)
        st.button("📧 Müşteriye Gönder", type="primary", key="customer_send_btn", disabled=True)
        return

    st.markdown("#### 🏢 MRC İçi Paylaş")
    st.caption("Yalnızca emre.hangul@mrc-tr.com ve cagatay.bayrak@mrc-tr.com adreslerine gider.")
    st.session_state.setdefault("internal_confirm_pending", False)
    if email_mod.is_sent("internal"):
        info = email_mod.sent_info("internal")
        st.success(f"✓ İç paylaşım yapıldı — {info.get('sent_at', '?')}")
        if st.button("🔄 MRC içine tekrar gönder", key="internal_resend_btn"):
            st.session_state["internal_confirm_pending"] = True
    elif st.button("🏢 MRC İçi Paylaş", key="internal_send_btn"):
        st.session_state["internal_confirm_pending"] = True

    _render_internal_confirmation(email_mod)

    st.divider()
    st.markdown("#### 📧 Müşteriye Gönder")
    st.caption("Test alıcısı: " + ", ".join(email_mod.CUSTOMER_TO))
    st.session_state.setdefault("customer_confirm_stage", 0)
    if email_mod.is_sent("customer"):
        info = email_mod.sent_info("customer")
        st.success(f"✓ Müşteri gönderimi tamamlandı — {info.get('sent_at', '?')}")
        if st.button("🔄 Müşteriye tekrar gönder", key="customer_resend_btn"):
            st.session_state["customer_confirm_stage"] = 1
    else:
        st.warning("Son forecast ve diagnostic dosyalarını kontrol ederek müşteri gönderimini onaylayın.")
        if st.button("📧 Müşteriye Gönder", type="primary", key="customer_send_btn"):
            st.session_state["customer_confirm_stage"] = 1

    _render_customer_double_confirmation(email_mod)


def _render_internal_confirmation(email_mod):
    """MRC ici email icin tek asamali bilincli teyit ister."""
    if not st.session_state.get("internal_confirm_pending", False):
        return
    st.warning("MRC içine mail atılacak, emin misiniz?")
    yes_col, cancel_col = st.columns(2)
    with yes_col:
        if st.button("Evet, MRC içine gönder", type="primary",
                     key="internal_confirm_yes", use_container_width=True):
            st.session_state["internal_confirm_pending"] = False
            _do_send_email(email_mod, "internal")
    with cancel_col:
        if st.button("Hayır, iptal et", key="internal_confirm_no", use_container_width=True):
            st.session_state["internal_confirm_pending"] = False
            _safe_rerun()


def _render_customer_double_confirmation(email_mod):
    """Musteri email'i icin iki ayri, ard arda bilincli teyit ister."""
    stage = st.session_state.get("customer_confirm_stage", 0)
    if stage == 1:
        st.warning("Müşteriye mail atılacak, emin misiniz?")
        yes_col, cancel_col = st.columns(2)
        with yes_col:
            if st.button("Evet, devam et", key="customer_confirm_first_yes", use_container_width=True):
                st.session_state["customer_confirm_stage"] = 2
                _safe_rerun()
        with cancel_col:
            if st.button("Hayır, iptal et", key="customer_confirm_first_no", use_container_width=True):
                st.session_state["customer_confirm_stage"] = 0
                _safe_rerun()
    elif stage == 2:
        st.error("Emin olduğunuzu bir kez daha teyit edin.")
        yes_col, back_col = st.columns(2)
        with yes_col:
            if st.button("Evet, eminim ve gönder", type="primary",
                         key="customer_confirm_second_yes", use_container_width=True):
                st.session_state["customer_confirm_stage"] = 0
                _do_send_email(email_mod, "customer")
        with back_col:
            if st.button("Geri dön", key="customer_confirm_second_no", use_container_width=True):
                st.session_state["customer_confirm_stage"] = 1
                _safe_rerun()


def _do_send_email(email_mod, audience: str):
    """Forecast sirasinda hazirlanan gunluk ciktilari email ile gonder."""
    with st.spinner("Hazır ADM+GDZ tahmin dosyaları email ile gönderiliyor..."):
        try:
            result = email_mod.run(audience=audience)
        except Exception as e:
            log.error(f"Email gönderim hatası\n{traceback.format_exc()}")
            st.error(f"✗ Email gönderilemedi: {e}")
            return

    log.info(f"══ 09_email_report ({audience}, onaylı) | {result}")
    status = result.get("status")
    if status == "ok":
        st.success(f"✓ Gönderildi: {', '.join(result.get('to', []))}")
        _safe_rerun()
    elif status == "skipped":
        st.info(f"SMTP ayarlı değil — email atlanmış görünüyor "
                f"({result.get('reason', '?')}). Env: STLF_SMTP_HOST/USER/PASS.")
    else:
        st.error(f"✗ Gönderilemedi: {result.get('error', result)}")


def _safe_rerun():
    """Streamlit sürümünden bağımsız güvenli rerun (yoksa sessizce geç)."""
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()


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

        stdout_path = gdz_stdout_path()
        summary_path = gdz_summary_path()
        # Önceki denemenin özeti yeni başarısız koşuyu başarılı göstermesin.
        try:
            summary_path.unlink(missing_ok=True)
        except OSError as e:
            st.warning(f"Eski GDZ summary temizlenemedi: {e}")
        proc = run_gdz_pipeline_async(args)

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

                summary_is_gdz = summary and summary.get("edas_id") == "GDZ"
                if summary_is_gdz and summary.get("status") == "ok":
                    status.update(label="✓ GDZ pipeline tamamlandı", state="complete")
                    for step_name, result in summary.get("steps", {}).items():
                        st.write(f"✓ **{step_name}** — {result}")
                    deliver = summary.get("steps", {}).get("06_deliver")
                    if deliver and deliver.get("output_file"):
                        st.success(f"Teslim dosyası: {deliver['output_file']}")
                else:
                    status.update(label="HATA — GDZ çalışma özeti oluşmadı", state="error")
                    st.error(
                        "GDZ subprocess çıkış kodu 0 olsa da bu çalışmaya ait geçerli "
                        "GDZ summary dosyası oluşmadı. Son log satırları aşağıdadır."
                    )
                    if summary:
                        st.json(summary)
                    st.code(_tail_lines(stdout_path, n=60), language="text")

        st.session_state["gdz_forecast_running"] = False


def render():
    st.subheader("Tahmin Üret")
    _render_adm()
    _render_gdz()
    email_mod = import_pipeline_step("09_email_report")
    readiness = email_mod.check_readiness()
    if readiness.get("ready"):
        forecast_adjustment.render(
            readiness["adm"]["target_date"],
            customer_sent=email_mod.is_sent("customer"),
        )
    _render_approval_panel()
