"""
09_email_report.py — STLF LIVE Rapor Email Gonderimi
======================================================
Ic ekibe: emre.hangul@mrc-tr.com & cagatay.bayrak@mrc-tr.com

ÖNEMLI (2026-07-10): Bu adim artik pipeline sonunda OTOMATIK CALISMAZ.
run_daily.py / UI pipeline'i 08_diagnostic'te durur ("onay bekliyor").
Email yalnizca kullanici UI'da "Musteriye Gonder" butonuna basinca bu run()
cagirilir. Basarili gonderimde DELIVERY_ROOT altina bir "gonderildi" isareti
(<tarih>_EMAIL_SENT.json) yazilir — ayni gun yanlislikla iki kez gonderimi
engellemek ve UI'da durumu gostermek icin.

KRITIK (2026-07-10, kullanici duzeltmesi): diagnostic dosyasi ISSUE tarihiyle
(bugun) degil TARGET/teslim tarihiyle (yarin) adlandiriliyor — 06_deliver
varsayilan olarak yarini hedefliyor ("bugun yarini tahmin ediyoruz"). Bu yuzden
HTML_DIAG'i date.today() ile aramak YANLIS (o gunun BAYAT/eski dosyasini
bulabilir) — 08_diagnostic_html.py'nin kendi mantigi kullanilmali (en guncel
*_forecast.xlsx dosyasindan tarih cikar).

KRITIK 2: Musteri teslimi hem ADM hem GDZ tahmin urettikten SONRA gonderilmeli
(kullanici talebi) — check_readiness() ikisinin de BUGUN calisip calismadigini
ve hedef tarihlerinin eslesip eslesmedigini kontrol eder; run() bu kontrolden
gecmeden gondermeyi REDDEDER (sadece UI uyarisi degil, gercek engel).

KLASORLEME (2026-07-12 guncelleme): Email 5 dosya icerir — birlesik
STLF_LIVE_RAPOR.xlsx + ADM'nin KENDI teslim excel'i + GDZ'nin KENDI teslim
excel'i + ikisinin diagnostic HTML'i. Bu 5 dosya artik proje disindaki
paylasilan DELIVERY_ROOT/YYYY.MM/D/ klasorune (bkz. src/output_paths.py)
DOGRUDAN 06/07/08 adimlarinda yazilir — ayrica bir "gunluk arsiv klasorune
kopyala" adimina gerek yok (eskiden vardi, artik dosyalar zaten dogru
klasorde dogar). REGEN/backtest dosyalari (asof_regen.py, backtest_7d/30d.py,
export_hourly_mape_7d.py) hala yerel output/'ta kaliyor, degismedi.

SMTP ayarlari MRC mail sunucusuna gore yapilandirilmalidir (env: STLF_SMTP_*).
"""
import sys, smtplib, os, json
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from email import encoders
from email.utils import formatdate, make_msgid
from datetime import date, datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# .env dosyasi (ROOT/.env, git'e girmez — bkz. .gitignore) varsa yukle.
# Zaten export edilmis gercek ortam degiskenlerini EZMEZ (override=False varsayilan),
# yani Windows'ta setx ile kalici ayarlanmissa o oncelikli kalir.
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── SMTP Konfigurasyonu (env var / .env ile) ────────────────────────
SMTP_HOST = os.getenv("STLF_SMTP_HOST")
SMTP_PORT = int(os.getenv("STLF_SMTP_PORT", "587"))
SMTP_USER = os.getenv("STLF_SMTP_USER", "cagatay.bayrak@mrc-tr.com")
SMTP_PASS = os.getenv("STLF_SMTP_PASS")
SMTP_SEND_AS_CONFIRMED = os.getenv("STLF_SMTP_SEND_AS_CONFIRMED", "").strip().lower() in {
    "1", "true", "yes", "evet",
}


def _env_recipients(name: str, default: list[str]) -> list[str]:
    """Virgul/noktali virgul ayrimli alici ortam degiskenini listeye cevir."""
    raw = os.getenv(name)
    if not raw:
        return list(default)
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]

# Faz 1 (2026-07-13, kullanıcı talimatı): "SADECE MRC MAİLLERİ İLK BAŞTA, BİZ
# SONRA KENDİMİZ MANUEL BİR BAKICAZ" — musteri adresi (aydemenerji) ic
# dogrulama bitene kadar KAPALI. audience="customer" cagrisi bile artik
# sadece MRC'ye gider; asagidaki satiri acmak musteriye gercek gonderim
# icin yeterli olacak (Faz 4'te resmi karar).
INTERNAL_TO = _env_recipients(
    "STLF_INTERNAL_TO",
    ["emre.hangul@mrc-tr.com", "cagatay.bayrak@mrc-tr.com"],
)
# Test asamasi: musteri akisi gercek Aydem/data-analytics alicilarina acilmadan
# yalnizca Emre'ye gider. Canliya geciste kod degisikligi yerine STLF_CUSTOMER_TO
# ortam degiskeni kullanilir.
CUSTOMER_TO = _env_recipients("STLF_CUSTOMER_TO", ["emre.hangul@mrc-tr.com"])
FROM = os.getenv("STLF_FROM_ADDRESS", "talep.tahmin@mrc-tr.com")
CUSTOMER_SUBJECT = os.getenv("STLF_CUSTOMER_SUBJECT", "MRC-AYDEM Demo GÖP Tahmin Sonuçları")
BUSINESS_CARD_PATH = Path(os.getenv(
    "STLF_BUSINESS_CARD_PATH",
    str(ROOT.parent.parent / "Emre" / "kartvizit.png"),
))

# ── Dosya yollari ──────────────────────────────────────────────────
from src.output_paths import dated_output_path, resolve_output_file, DELIVERY_ROOT
import config_live as _ADM_CONFIG
ADM_FILENAME_TEMPLATE = getattr(_ADM_CONFIG, "OUTPUT_FILENAME_TEMPLATE", "ADM_forecast_{date}.xlsx")
GDZ_LIVE_ROOT = getattr(_ADM_CONFIG, "GDZ_LIVE_ROOT", ROOT.parent / "gdz talep" / "live")
EXCEL_REPORT = DELIVERY_ROOT / "STLF_LIVE_RAPOR.xlsx"
EMAIL_STATE_DIR = DELIVERY_ROOT / ".email_state"

# GDZ'nin kendi kok dizini (run summary log'lari + OUTPUT_FILENAME_TEMPLATE icin)
# — Faz 1 (2026-07-13): config_live.GDZ_LIVE_ROOT'tan okunur (eskiden burada
# ayrica hardcoded'di, pipeline/07_report_excel.py'de de ayrica hardcoded'di).
GDZ_LIVE_DIR = GDZ_LIVE_ROOT
sys.path.insert(0, str(GDZ_LIVE_DIR))
from config_live_gdz import OUTPUT_FILENAME_TEMPLATE as GDZ_FILENAME_TEMPLATE


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def expected_files(fc_date: str, include_report: bool = True) -> dict[str, tuple[Path, str]]:
    """Musteri teslimine dahil olmasi gereken TUM dosyalar: {key: (kaynak_yol, ek_adi)}.

    check_readiness() (dosyalar GERCEKTEN var mi) VE run() (attach) AYNI bu
    fonksiyonu kullanir — boylece ikisi arasinda mantik drifti (readiness
    "hazir" derken run()'un farkli/eksik dosya bulmasi — 2026-07-10'da
    bulunan bug, bkz. asagidaki KRITIK 3 notu) yapisal olarak imkansiz hale
    gelir. ADM ve GDZ ayni paylasilan DELIVERY_ROOT'u kullandigi icin GDZ'nin
    kendi config'ini yuklemeye artik gerek yok.
    """
    adm_forecast_name = ADM_FILENAME_TEMPLATE.format(date=fc_date)
    gdz_forecast_name = GDZ_FILENAME_TEMPLATE.format(date=fc_date)
    files = {
        "adm_forecast": (resolve_output_file(DELIVERY_ROOT, adm_forecast_name), adm_forecast_name),
        "adm_diagnostic": (resolve_output_file(DELIVERY_ROOT, f"diagnostic_{fc_date}.html"), f"diagnostic_adm_{fc_date}.html"),
        "gdz_forecast": (resolve_output_file(DELIVERY_ROOT, gdz_forecast_name), gdz_forecast_name),
        "gdz_diagnostic": (resolve_output_file(DELIVERY_ROOT, f"diagnostic_gdz_{fc_date}.html"), f"diagnostic_gdz_{fc_date}.html"),
    }
    if include_report:
        files = {"excel_report": (dated_output_path(DELIVERY_ROOT, fc_date, "STLF_LIVE_RAPOR.xlsx"), "STLF_LIVE_RAPOR.xlsx"), **files}
    return files


def check_readiness() -> dict:
    """ADM ve GDZ BUGUN calisti mi, hedef tarihleri eslesiyor mu, VE musteriye
    gidecek 5 dosyanin (excel rapor + 2 teslim excel + 2 diagnostic) HEPSI
    GERCEKTEN diskte var mi?

    KRITIK 3 (2026-07-10, bagimsiz denetimde bulundu): eskiden sadece
    "06_deliver.status == ok" kontrol ediliyordu — 07_report_excel/08_diagnostic
    adimlari run_daily.py'de ayri try/except icinde oldugu icin (hata olursa
    sadece log.warning basip pipeline'i "ok" ile bitiriyorlar), 06 basarili
    olup 07/08 SESSIZCE basarisiz olsa bile "ready": True donuyordu — UI'da
    yesil buton gorunuyordu ama gonderilen email'de rapor/diagnostic eksik
    olabiliyordu, kullaniciya hicbir uyari gitmiyordu. Simdi expected_files()
    ile dosyalarin GERCEKTEN var olup olmadigi da kontrol ediliyor.

    Gunluk run summary'lerini (run_context.write_summary'nin yazdigi
    <tarih>_summary.json) okuyarak kontrol eder — session_state'e bagli
    degildir, UI yeniden baslasa/farkli sekmede acilsa bile diskten dogru
    sonuc verir.
    """
    today = date.today().strftime("%Y-%m-%d")

    adm = _read_json(ROOT / "logs" / f"{today}_summary.json")
    gdz = _read_json(GDZ_LIVE_DIR / "logs" / f"{today}_summary.json")

    adm_deliver = (adm.get("steps") or {}).get("06_deliver") or {}
    gdz_deliver = (gdz.get("steps") or {}).get("06_deliver") or {}

    adm_target = adm_deliver.get("target_date")
    gdz_target = gdz_deliver.get("target_date")

    adm_ran = adm_deliver.get("status") == "ok"
    gdz_ran = gdz_deliver.get("status") == "ok"

    target_match = bool(adm_target) and bool(gdz_target) and (adm_target == gdz_target)

    missing_files: list[str] = []
    if adm_ran and gdz_ran and target_match:
        # Ortak STLF raporu forecast pipeline'lari tarafindan daha once uretilmis
        # olmalidir. MRC/musteri gonderim butonu rapor olusturma noktasi degildir.
        for src, archive_name in expected_files(adm_target, include_report=True).values():
            if not src or not src.exists():
                missing_files.append(archive_name)

    return {
        "ready": adm_ran and gdz_ran and target_match and not missing_files,
        "adm": {"ran_today": adm_ran, "target_date": adm_target},
        "gdz": {"ran_today": gdz_ran, "target_date": gdz_target},
        "target_match": target_match,
        "missing_files": missing_files,
    }


# ── Onay/gonderim isareti (UI ile ortak) ───────────────────────────
def sent_marker_path(audience: str = "internal", d: date | None = None) -> Path:
    """O gunun 'email gonderildi' isaret dosyasinin yolu."""
    d = d or date.today()
    suffix = "CUSTOMER_EMAIL_SENT" if audience == "customer" else "INTERNAL_EMAIL_SENT"
    return dated_output_path(DELIVERY_ROOT, str(d), f"{d.strftime('%Y-%m-%d')}_{suffix}.json", create=True)


def is_sent(audience: str = "internal", d: date | None = None) -> bool:
    """Bugun (ya da verilen gun) email zaten gonderildi mi?"""
    marker = sent_marker_path(audience, d)
    if audience == "internal" and not marker.exists():
        legacy = resolve_output_file(DELIVERY_ROOT, f"{(d or date.today()).strftime('%Y-%m-%d')}_EMAIL_SENT.json")
        return legacy.exists()
    return marker.exists()


def sent_info(audience: str = "internal", d: date | None = None) -> dict:
    """Gonderim isaretinin icerigi (sent_at, to). Yoksa bos dict."""
    p = sent_marker_path(audience, d)
    if audience == "internal" and not p.exists():
        p = resolve_output_file(DELIVERY_ROOT, f"{(d or date.today()).strftime('%Y-%m-%d')}_EMAIL_SENT.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _attach(msg: MIMEMultipart, path: Path | None, filename: str, maintype: str, subtype: str) -> None:
    """path varsa dosyayi msg'e ekle (base64) — filename kaynak dosya adindan
    FARKLI olabilir (orn. diagnostic_2026-07-11.html diskte, email'de
    diagnostic_adm_2026-07-11.html olarak gorunsun diye)."""
    if not path or not path.exists():
        print(f"     (uyari) ek bulunamadi, atlandi: {filename}")
        return
    with open(path, "rb") as f:
        part = MIMEBase(maintype, subtype)
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={filename}")
        msg.attach(part)


def _attach_xlsx(msg: MIMEMultipart, path: Path | None, filename: str) -> None:
    _attach(msg, path, filename, "application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def _attach_html(msg: MIMEMultipart, path: Path | None, filename: str) -> None:
    _attach(msg, path, filename, "text", "html")


def forecast_date_display(fc_date: str) -> str:
    """Summary'deki forecast HEDEF tarihini musteri formatina cevir.

    Gonderim/issue tarihi kullanilmaz: 12 Temmuz run'i 13 Temmuz tahminini
    gonderiyorsa metinde 13.07.2026 yazar.
    """
    return date.fromisoformat(fc_date).strftime("%d.%m.%Y")


def thread_state_path(audience: str) -> Path:
    """Gunler arasi Outlook conversation zincirini surduren durum dosyasi."""
    return EMAIL_STATE_DIR / f"{audience}_thread.json"


def _load_thread_state(audience: str) -> dict:
    if audience != "customer":
        return {}
    return _read_json(thread_state_path(audience))


def _add_thread_headers(msg: MIMEMultipart, audience: str, state: dict | None = None) -> dict:
    """Sabit konuya ek olarak RFC uyumlu reply-chain basliklarini ekle.

    SMTP'de sunucu tarafinda gercek bir ReplyAll komutu yoktur. Ayni konuyu ve
    Message-ID/In-Reply-To/References zincirini kullanmak Outlook'un gunluk
    iletileri ayni conversation altinda gostermesini saglar.
    """
    domain = FROM.rsplit("@", 1)[-1] if "@" in FROM else None
    message_id = make_msgid(domain=domain)
    msg["Message-ID"] = message_id
    msg["Date"] = formatdate(localtime=True)

    state = dict(state if state is not None else _load_thread_state(audience))
    previous = state.get("last_message_id")
    references = list(state.get("references") or [])
    if previous:
        msg["In-Reply-To"] = previous
        if previous not in references:
            references.append(previous)
        # Baslik boyutunun sinirsiz buyumesini engellerken ilk kok mesaji koru.
        references = references[:1] + references[-49:] if len(references) > 50 else references
        msg["References"] = " ".join(references)

    return {
        "root_message_id": state.get("root_message_id") or message_id,
        "last_message_id": message_id,
        "references": references or [message_id],
    }


def _save_thread_state(audience: str, state: dict, recipients: list[str]) -> None:
    if audience != "customer":
        return
    EMAIL_STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        **state,
        "subject": CUSTOMER_SUBJECT,
        "from": FROM,
        "to": recipients,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    thread_state_path(audience).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _attach_inline_business_card(container: MIMEMultipart) -> bool:
    """Kartviziti HTML govdesinde cid:kartvizit olarak goster."""
    if not BUSINESS_CARD_PATH.exists():
        print(f"     (uyari) kartvizit bulunamadi: {BUSINESS_CARD_PATH}")
        return False
    with BUSINESS_CARD_PATH.open("rb") as fh:
        image = MIMEImage(fh.read())
    image.add_header("Content-ID", "<kartvizit>")
    image.add_header("Content-Disposition", "inline", filename=BUSINESS_CARD_PATH.name)
    container.attach(image)
    return True


def _build_message(
    audience: str,
    fc_date: str,
    recipients: list[str],
    thread_state: dict | None = None,
) -> tuple[MIMEMultipart, dict]:
    """Gonderimden bagimsiz, test edilebilir MIME mesajini olustur."""
    msg = MIMEMultipart("mixed")
    delivery_display = forecast_date_display(fc_date)
    msg["Subject"] = CUSTOMER_SUBJECT if audience == "customer" else f"[STLF LIVE] {delivery_display} Talep Tahminleri"
    msg["From"] = FROM
    msg["Reply-To"] = FROM
    msg["To"] = ", ".join(recipients)
    effective_thread_state = thread_state if thread_state is not None else _load_thread_state(audience)
    previous_recipients = (effective_thread_state or {}).get("to")
    if previous_recipients and {x.lower() for x in previous_recipients} != {x.lower() for x in recipients}:
        # Test alici listesinden canli listeye geciste eski conversation'i
        # yeni alicilara yapay olarak tasimak yerine temiz bir zincir baslat.
        effective_thread_state = {}
    next_thread_state = _add_thread_headers(msg, audience, effective_thread_state)

    related = MIMEMultipart("related")
    if audience == "customer":
        card_available = BUSINESS_CARD_PATH.exists()
        card_html = (
            '<p style="margin-top:24px"><img src="cid:kartvizit" '
            'alt="MRC kartvizit" style="display:block;max-width:100%;height:auto"></p>'
            if card_available else ""
        )
        body = f"""
<html><body style="font-family:Segoe UI,sans-serif">
<p>Merhabalar,</p>
<p>{delivery_display} tarihine dair tahminlerimiz ektedir. Bilginize sunarız.</p>
<p>Kolay gelsin iyi çalışmalar.</p>
{card_html}
</body></html>
        """
        related.attach(MIMEText(body, "html", "utf-8"))
        if card_available:
            _attach_inline_business_card(related)
    else:
        body = f"""
<html>
<body style="font-family:Segoe UI,sans-serif;background:#f5f5f5;padding:20px">
<div style="max-width:600px;margin:auto;background:white;border-radius:12px;padding:24px">
<h2 style="color:#1a73e8">STLF LIVE — Gunluk Tahmin Raporu</h2>
<p style="color:#555;font-size:14px">Teslim Günü: <b>{fc_date}</b> &nbsp;|&nbsp; Hazırlanma: {date.today()}</p>
<hr style="border:0;border-top:1px solid #ddd">

<h3>📊 Ekli Dosyalar</h3>
<ul>
  <li><b>STLF_LIVE_RAPOR.xlsx</b> — ADM + GDZ; gerceklesen, D+2 tahmin ve mutlak sapma tablolari</li>
  <li><b>ADM_forecast_{fc_date}.xlsx</b> — ADM teslim excel'i</li>
  <li><b>GDZ_forecast_{fc_date}.xlsx</b> — GDZ teslim excel'i</li>
  <li><b>diagnostic_adm_{fc_date}.html</b> — ADM interaktif Chart.js dashboard</li>
  <li><b>diagnostic_gdz_{fc_date}.html</b> — GDZ interaktif Chart.js dashboard</li>
</ul>

<h3>🔗 Hizli Linkler</h3>
<ul>
  <li><a href="https://onedrive/...">ADM Dashboard</a></li>
  <li><a href="https://onedrive/...">GDZ Dashboard</a></li>
</ul>

<hr style="border:0;border-top:1px solid #ddd">
<p style="color:#999;font-size:12px">Bu email otomatik uretilmistir. Lutfen yanitlamayiniz.</p>
</div>
</body>
</html>
        """
        related.attach(MIMEText(body, "html", "utf-8"))

    msg.attach(related)
    return msg, next_thread_state


def run(audience: str = "internal") -> dict:
    """
    STLF LIVE Raporu email ile gonder.
    SMTP ayarlari yapilmamissa atlar (hata vermez).
    ADM+GDZ ikisi de bugun calisip ayni hedef tarihi uretmemisse REDDEDER
    (bkz. check_readiness) — kullanici talebi: "hem ADM hem GDZ tahminler
    uretildikten sonra gonderelim".
    """
    if audience not in {"internal", "customer"}:
        raise ValueError(f"Gecersiz email hedefi: {audience}")
    recipients = CUSTOMER_TO if audience == "customer" else INTERNAL_TO
    print(f"\n[09] STLF EMAIL gonderiliyor ({audience})...")

    readiness = check_readiness()
    if not readiness["ready"]:
        print(f"     Gonderim reddedildi — ADM+GDZ ikisi de bugun hazir degil: {readiness}")
        return {"status": "not_ready", "readiness": readiness}

    global EXCEL_REPORT
    fc_date = readiness["adm"]["target_date"]  # == gdz target_date (check_readiness garanti eder)
    EXCEL_REPORT = dated_output_path(DELIVERY_ROOT, fc_date, "STLF_LIVE_RAPOR.xlsx", create=True)

    # Rapor burada uretilmez. ADM/GDZ forecast ve tahmin revizyon akislari raporu
    # aninda gunceller; readiness yukarida eksik/bayat olmayan dosyayi zorunlu tutar.
    print(f"     Ortak ADM+GDZ raporu hazir: {EXCEL_REPORT}")

    if not SMTP_HOST or not SMTP_USER:
        print("     SMTP ayarlari yapilmamis (env: STLF_SMTP_HOST/USER/PASS) — rapor olustu, email atlandi.")
        return {"status": "skipped", "reason": "SMTP not configured", "report": str(EXCEL_REPORT)}
    if SMTP_USER.lower() != FROM.lower() and not SMTP_SEND_AS_CONFIRMED:
        reason = (
            f"SMTP giris hesabi ({SMTP_USER}) ile gorunen gonderen ({FROM}) farkli. "
            "Exchange yoneticisi Send As yetkisini verdikten sonra "
            "STLF_SMTP_SEND_AS_CONFIRMED=true ayarlanmalidir."
        )
        print(f"     Email atlandi — {reason}")
        return {"status": "skipped", "reason": reason, "report": str(EXCEL_REPORT)}

    # check_readiness ile AYNI kaynak (expected_files) — dosyalarin var oldugu
    # zaten dogrulandi (readiness["ready"] burada True), sadece yollari al.
    files = expected_files(fc_date)
    EXCEL_SRC, EXCEL_NAME = files["excel_report"]
    ADM_FORECAST, ADM_FORECAST_NAME = files["adm_forecast"]
    HTML_DIAG, HTML_DIAG_NAME = files["adm_diagnostic"]
    GDZ_FORECAST, GDZ_FORECAST_NAME = files["gdz_forecast"]
    GDZ_HTML_DIAG, GDZ_HTML_DIAG_NAME = files["gdz_diagnostic"]

    # Mesaj olustur. Musteri iletilerinde onceki gunun Message-ID'si kullanilarak
    # Outlook conversation/reply-all zinciri korunur.
    msg, next_thread_state = _build_message(audience, fc_date, recipients)

    _attach_xlsx(msg, EXCEL_SRC, EXCEL_NAME)
    _attach_xlsx(msg, ADM_FORECAST, ADM_FORECAST_NAME)
    _attach_xlsx(msg, GDZ_FORECAST, GDZ_FORECAST_NAME)
    _attach_html(msg, HTML_DIAG, HTML_DIAG_NAME)
    _attach_html(msg, GDZ_HTML_DIAG, GDZ_HTML_DIAG_NAME)

    # Gonder
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"     Email gonderildi: {', '.join(recipients)}")

        # Yalnizca SMTP gonderimi basarili olduktan sonra zinciri ilerlet.
        try:
            _save_thread_state(audience, next_thread_state, recipients)
        except Exception as thread_err:
            print(f"     (uyari) email zincir bilgisi yazilamadi: {thread_err}")

        # Gonderim isaretini yaz (UI 'gonderildi' durumunu buradan okur;
        # ayni gun yanlislikla ikinci gonderimi engeller).
        sent_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            sent_marker_path(audience).write_text(
                json.dumps({"sent_at": sent_at, "to": recipients, "audience": audience}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as mk_err:
            print(f"     (uyari) gonderim isareti yazilamadi: {mk_err}")

        return {"status": "ok", "to": recipients, "sent_at": sent_at, "audience": audience}
    except Exception as e:
        print(f"     Email hatasi: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    result = run()
    print(result)
