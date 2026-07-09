"""
09_email_report.py — STLF LIVE Rapor Email Gonderimi
======================================================
Her pipeline sonrasi otomatik: emre.hangul@mrc-tr.com & cagatay.bayrak@mrc-tr.com

SMTP ayarlari MRC mail sunucusuna gore yapilandirilmalidir.
Bu script bir template'dir — SMTP bilgilerini config'e ekleyin.
"""
import sys, smtplib, os
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from datetime import date

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── SMTP Konfigurasyonu (env var ile) ──────────────────────────────
SMTP_HOST = os.getenv("STLF_SMTP_HOST")
SMTP_PORT = int(os.getenv("STLF_SMTP_PORT", "587"))
SMTP_USER = os.getenv("STLF_SMTP_USER")
SMTP_PASS = os.getenv("STLF_SMTP_PASS")

TO = ["emre.hangul@mrc-tr.com", "cagatay.bayrak@mrc-tr.com"]
FROM = SMTP_USER or "stlf@mrc-tr.com"

# ── Dosya yollari ──────────────────────────────────────────────────
from config_live import OUTPUT_DIR
EXCEL_REPORT = OUTPUT_DIR / "STLF_LIVE_RAPOR.xlsx"
HTML_DIAG = OUTPUT_DIR / f"diagnostic_{date.today().strftime('%Y-%m-%d')}.html"


def run() -> dict:
    """
    STLF LIVE Raporu email ile gonder.
    SMTP ayarlari yapilmamissa atlar (hata vermez).
    """
    print("\n[09] STLF EMAIL gonderiliyor...")

    if not SMTP_HOST or not SMTP_USER:
        print("     SMTP ayarları yapılmamış (env: STLF_SMTP_HOST/USER/PASS) — email atlandı.")
        return {"status": "skipped", "reason": "SMTP not configured"}

    # Mesaj olustur
    msg = MIMEMultipart()
    msg["Subject"] = f"[STLF LIVE] Gunluk Tahmin Raporu — {date.today()}"
    msg["From"] = FROM
    msg["To"] = ", ".join(TO)

    body = f"""
<html>
<body style="font-family:Segoe UI,sans-serif;background:#f5f5f5;padding:20px">
<div style="max-width:600px;margin:auto;background:white;border-radius:12px;padding:24px">
<h2 style="color:#1a73e8">STLF LIVE — Gunluk Tahmin Raporu</h2>
<p style="color:#555;font-size:14px">Tarih: <b>{date.today()}</b></p>
<hr style="border:0;border-top:1px solid #ddd">

<h3>📊 Ekli Dosyalar</h3>
<ul>
  <li><b>STLF_LIVE_RAPOR.xlsx</b> — ADM + GDZ, 5 tablo, guncel veri</li>
  <li><b>diagnostic_{date.today()}.html</b> — Interaktif Chart.js dashboard</li>
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
    msg.attach(MIMEText(body, "html"))

    # Excel ekle
    if EXCEL_REPORT.exists():
        with open(EXCEL_REPORT, "rb") as f:
            part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={EXCEL_REPORT.name}")
            msg.attach(part)

    # HTML ekle
    if HTML_DIAG.exists():
        with open(HTML_DIAG, "rb") as f:
            part = MIMEBase("text", "html")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={HTML_DIAG.name}")
            msg.attach(part)

    # Gonder
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"     Email gonderildi: {', '.join(TO)}")
        return {"status": "ok", "to": TO}
    except Exception as e:
        print(f"     Email hatasi: {e}")
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    result = run()
    print(result)
