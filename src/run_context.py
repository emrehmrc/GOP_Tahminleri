"""
run_context.py — Run kimliği, model arşivi ve paylaşılan loglama (Faz -1)
=========================================================================
Tekrarlanabilirliğin sıfır noktası. Bir tahmin çalıştırmasını (run) kimliklendirir,
o run'ın modellerini arşivler ve hem UI hem CLI için ortak dosya loglaması kurar.

NEDEN paylaşılan modül: Canlı tahminler Streamlit UI (ui/tab_tahmin_uret.py)
üzerinden tetikleniyor; UI run_daily.py'yi HİÇ çağırmıyor. Bu yüzden run_id /
arşiv / loglama sadece run_daily.py'ye konursa production'da çalışmaz. Bu modülü
her iki yol da (UI + CLI) aynı şekilde çağırır → tek doğruluk kaynağı.

Üretilen primitifler Faz 0 forecast_log'unun `config_hash` / `run_id` /
`model_versions` alanlarını besler.

Kullanım:
    from run_context import start_run, archive_models, prune_archive, write_summary
    ctx = start_run()                 # run başında
    ...                               # 04 (modeller diske yazılır)
    archive_models(ctx)               # 04 başarılıysa
    prune_archive()
    write_summary(ctx, steps, "ok")   # run sonunda (başarı VEYA hata)
"""

import json
import shutil
import logging
import hashlib
from pathlib import Path
from datetime import date, datetime

import config_live as C

log = logging.getLogger("adm_live")

CONFIG_FILE = C.LIVE_DIR / "config_live.py"


# ── Hash yardımcıları ─────────────────────────────────────────────────────────
def _hash_file(path: Path, length: int = 64) -> str | None:
    """Bir dosyanın sha256 hex özeti (ilk `length` karakter). Dosya yoksa None."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _dir_fingerprint(path: Path) -> dict | None:
    """Büyük donmuş dizin (Chronos adapter) için hafif parmak izi: mtime + toplam boyut.
    Her run tam hash almak (19 MB) gereksiz — değişiklik tespiti için bu yeter."""
    p = Path(path)
    if not p.is_dir():
        return None
    total, latest = 0, 0.0
    for f in p.rglob("*"):
        if f.is_file():
            st = f.stat()
            total += st.st_size
            latest = max(latest, st.st_mtime)
    return {"size_bytes": total, "mtime": datetime.fromtimestamp(latest).isoformat(timespec="seconds")}


def compute_config_hash(length: int = 8) -> str:
    """config_live.py içeriğinin sha256'sı (ilk `length` karakter).
    run_id'nin ikinci parçası; config değişimi changepoint analizinde otomatik işaret olur."""
    h = _hash_file(CONFIG_FILE, length=length)
    return h if h is not None else "nohash00"


# ── Paylaşılan loglama ────────────────────────────────────────────────────────
def _setup_file_logging(issue_date: date) -> Path:
    """'adm_live' logger'ına logs/<issue_date>_run.log FileHandler'ı ekle (idempotent).
    Streamlit uzun ömürlü süreç → aynı handler'ı iki kez eklememek için koruma var."""
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = C.LOGS_DIR / f"{issue_date}_run.log"

    log.setLevel(logging.INFO)
    log.propagate = False
    target = str(log_file)
    have_file = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(Path(target).resolve())
        for h in log.handlers
    )
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not have_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in log.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log_file


# ── Run yaşam döngüsü ─────────────────────────────────────────────────────────
def start_run(target_date: str | None = None) -> dict:
    """Run kimliğini üret, kalıcılaştır, loglamayı kur.

    issue_date = bugün (run'ın koşturulduğu gün). run_id = <issue_date>_<config_hash8>.
    target_date (teslim günü, T+2) yalnızca metadata olarak kaydedilir.
    """
    issue_date = date.today()
    config_hash = compute_config_hash()
    run_id = f"{issue_date}_{config_hash}"

    ctx = {
        "edas_id": C.EDAS_ID,
        "run_id": run_id,
        "config_hash": config_hash,
        "issue_date": str(issue_date),
        "target_date": target_date or None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    _setup_file_logging(issue_date)
    C.RUN_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(C.RUN_CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info(f"RUN BAŞLADI — run_id={run_id} — config_hash={config_hash} — edas={C.EDAS_ID}")
    log.info("=" * 60)
    return ctx


def get_run_context() -> dict:
    """data/run_context.json'ı oku (pipeline adımları / UI mevcut run_id'i buradan alır)."""
    if not C.RUN_CONTEXT_PATH.is_file():
        raise FileNotFoundError(f"Run context yok: {C.RUN_CONTEXT_PATH} — önce start_run() çağır.")
    with open(C.RUN_CONTEXT_PATH, encoding="utf-8") as f:
        return json.load(f)


def archive_models(ctx: dict) -> dict:
    """Günlük eğitilen modelleri models/archive/<run_id>/'a kopyala + manifest yaz.

    - Kopyalanır: DAILY_RETRAINED_MODELS (~7.5 MB) + config_live.py snapshot.
    - Kopyalanmaz (sadece manifest'e hash): FROZEN_ARTEFACTS (git-tracked) + Chronos adapter (büyük).
    - manifest.model_versions Faz 0 forecast_log `model_versions` kolonunu besler.
    04 başarıyla bittikten SONRA çağrılır (modeller o adımda diske yazılır)."""
    run_id = ctx["run_id"]
    dest = C.MODEL_ARCHIVE_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)

    model_versions: dict[str, str | None] = {}

    for src in C.DAILY_RETRAINED_MODELS:
        src = Path(src)
        if src.is_file():
            shutil.copy2(src, dest / src.name)
            model_versions[src.name] = _hash_file(src, length=12)
        else:
            model_versions[src.name] = None
            log.warning(f"Arşiv: günlük model bulunamadı, atlandı → {src.name}")

    shutil.copy2(CONFIG_FILE, dest / "config_live.py")

    frozen_versions = {Path(p).name: _hash_file(p, length=12) for p in C.FROZEN_ARTEFACTS}
    chronos_fp = _dir_fingerprint(C.CHRONOS_ADAPTER_DIR)

    manifest = {
        **{k: ctx[k] for k in ("edas_id", "run_id", "config_hash", "issue_date", "target_date", "started_at")},
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "model_versions": model_versions,          # kopyalanan günlük modeller
        "frozen_versions": frozen_versions,         # git-tracked kalibrasyon (referans)
        "chronos_adapter": chronos_fp,              # büyük donmuş adapter parmak izi
    }
    with open(dest / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    log.info(f"Modeller arşivlendi → {dest} ({sum(v is not None for v in model_versions.values())} model)")
    return {"archive_dir": str(dest), "model_versions": model_versions}


def prune_archive(keep_days: int | None = None) -> int:
    """models/archive/ altında issue_date'i keep_days'ten eski run dizinlerini sil.
    run_id formatı '<YYYY-MM-DD>_<hash>' → tarih prefiksinden yaş hesaplanır."""
    keep_days = keep_days if keep_days is not None else C.ARCHIVE_RETENTION_DAYS
    if not C.MODEL_ARCHIVE_DIR.is_dir():
        return 0
    today = date.today()
    removed = 0
    for d in C.MODEL_ARCHIVE_DIR.iterdir():
        if not d.is_dir():
            continue
        try:
            issue = date.fromisoformat(d.name.split("_", 1)[0])
        except ValueError:
            log.warning(f"Arşiv budama: tarih ayrıştırılamadı, atlandı → {d.name}")
            continue
        if (today - issue).days > keep_days:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    if removed:
        log.info(f"Arşiv budama: {removed} eski run dizini silindi (>{keep_days} gün)")
    return removed


def write_summary(ctx: dict, steps: dict, status: str) -> Path:
    """logs/<issue_date>_summary.json yaz — HEM UI HEM CLI çağırır.
    (Bugün yalnızca run_daily yazıyor; UI yolu hiç summary üretmiyordu.)"""
    summary = {
        **{k: ctx.get(k) for k in ("edas_id", "run_id", "config_hash", "issue_date", "target_date", "started_at")},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "steps": steps,
    }
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.LOGS_DIR / f"{ctx['issue_date']}_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    log.info(f"Özet yazıldı → {path} (status={status})")
    return path
