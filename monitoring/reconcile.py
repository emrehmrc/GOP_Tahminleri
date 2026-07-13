"""
monitoring/reconcile.py — Faz 3 (2026-07-11): günlük tamlık kontrolü.

STLF_MONITORING_REFACTOR_PLAN.md §5. Faz 1'in `heal_forecast_log_gaps()`'i
zaten arşivi OLAN boşlukları sessizce dolduruyor. Bu modül onun bir adım
ÜSTÜNE, "arşivi de yok" durumunu — yani gerçek bir veri kaybını — sessiz
`log.warning(...)` yerine görünür `logs/gaps/<run_date>.json` dosyasına yazan
tek bir `reconcile()` fonksiyonu ekler. run_daily.py'nin heal için ayrı
çağırdığı try/except bloğu artık bunu çağırır.

Kapsam BİLEREK forecast_log ile sınırlı: actuals_log'daki "gecikme"
(D+1 yük, ~D+6 hava) zaten beklenen bir durum — bunu "kayıp" saymak yanlış
pozitif üretir, o yüzden burada kontrol edilmiyor.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import duckdb

from monitoring.forecast_logger import heal_forecast_log_gaps, _scan_archive_coverage
from monitoring.tenant_config import TenantConfig

DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_HORIZON_LABELS = ("T+1", "T+2")


def _horizon_days_for_label(config: TenantConfig, label: str) -> int:
    return int(label.replace("T+", "")) - config.horizon_day_label_offset


def _system_start_issue_date(config: TenantConfig, con: duckdb.DuckDBPyConnection,
                              horizon_labels: tuple[str, ...]) -> date | None:
    """forecast_log_v'deki EN ESKİ target_date, tanım gereği sistem henüz yeni
    başlamışken üretilmiş olabilir — o gün için sadece en KISA ufuk (en küçük
    horizon_days) mevcut olabilir, daha uzun ufuklar (örn T+2) o tarihte
    YAPISAL OLARAK imkânsızdır (issue_date sistem başlamadan önceye düşer).
    Bu, "sistem henüz çalışmıyordu" ile "run başarısız oldu" ayrımını yapar —
    ilki gerçek kayıp değildir. issue_date kolonu Faz 1 öncesi satırlarda NULL
    olabildiği için horizon_day STRING üzerinden türetilir (her zaman dolu)."""
    t0_row = con.execute(
        "SELECT MIN(target_date) FROM forecast_log_v WHERE edas_id = ?",
        [config.edas_id],
    ).fetchone()
    if t0_row is None or t0_row[0] is None:
        return None
    t0 = date.fromisoformat(t0_row[0])

    present_labels = con.execute(
        "SELECT DISTINCT horizon_day FROM forecast_log_v WHERE edas_id = ? AND target_date = ?",
        [config.edas_id, t0_row[0]],
    ).df()["horizon_day"].tolist()
    known_hdays = [_horizon_days_for_label(config, l) for l in present_labels if l in horizon_labels]
    if not known_hdays:
        return None
    return t0 - timedelta(days=min(known_hdays))


def _check_completeness(config: TenantConfig, lookback_days: int,
                         horizon_labels: tuple[str, ...]) -> list[dict]:
    """Son `lookback_days` gün x `horizon_labels` için forecast_log_v'de 24
    satır var mı bak. Eksikse VE arşivde tam (24 satır) karşılığı da yoksa
    VE hücrenin gerektirdiği issue_date sistem başladıktan sonraysa (bkz.
    `_system_start_issue_date`) 'gerçek kayıp' say (heal zaten bir önceki
    adımda arşivi olanları doldurdu — burada hâlâ eksikse ya arşiv de yok ya
    da heal başarısız oldu)."""
    if not config.monitoring_db.exists():
        return []

    con = duckdb.connect(str(config.monitoring_db), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").df()["name"].tolist()
        if "forecast_log_v" not in tables:
            return []
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        placeholders = ",".join("?" * len(horizon_labels))
        counts = con.execute(
            f"SELECT target_date, horizon_day, count(*) n FROM forecast_log_v "
            f"WHERE edas_id = ? AND target_date >= ? AND horizon_day IN ({placeholders}) "
            f"GROUP BY 1, 2",
            [config.edas_id, cutoff, *horizon_labels],
        ).df()
        system_start_issue = _system_start_issue_date(config, con, horizon_labels)
    finally:
        con.close()

    existing = {(row.target_date, row.horizon_day): row.n for row in counts.itertuples()}
    archive_candidates = _scan_archive_coverage(config)

    gaps = []
    for k in range(lookback_days):
        tdate_d = date.today() - timedelta(days=k)
        tdate = tdate_d.isoformat()
        for label in horizon_labels:
            n = existing.get((tdate, label), 0)
            if n >= 24:
                continue
            hdays = _horizon_days_for_label(config, label)
            if (tdate, hdays) in archive_candidates:
                continue  # arşiv var — heal bir sonraki run'da bunu dolduracak, gerçek kayıp değil
            needed_issue = tdate_d - timedelta(days=hdays)
            if system_start_issue is not None and needed_issue < system_start_issue:
                continue  # sistem o gün henüz çalışmıyordu — gerçek kayıp değil, yapısal imkânsızlık
            gaps.append({
                "target_date": tdate,
                "horizon_day": label,
                "rows_found": n,
                "archive_available": False,
            })
    return gaps


def _write_gap_report(config: TenantConfig, gaps: list[dict]) -> None:
    config.gaps_dir.mkdir(parents=True, exist_ok=True)
    path = config.gaps_dir / f"{date.today().isoformat()}.json"
    path.write_text(json.dumps(gaps, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.getLogger(config.logger_name).warning(
        f"[Reconcile] {len(gaps)} hücre GERÇEK KAYIP (arşiv de yok) -> {path}")


def reconcile(config: TenantConfig, lookback_days: int = DEFAULT_LOOKBACK_DAYS,
              horizon_labels: tuple[str, ...] = DEFAULT_HORIZON_LABELS) -> dict:
    """run_daily.py'nin her run sonunda çağırdığı tek fonksiyon: heal + tamlık
    kontrolü. Dönen dict: {"status", "heal": {...}, "gaps": [...]}."""
    heal_result = heal_forecast_log_gaps(config)
    gaps = _check_completeness(config, lookback_days, horizon_labels)
    if gaps:
        _write_gap_report(config, gaps)
    return {
        "status": "gaps_found" if gaps else "ok",
        "heal": heal_result,
        "gaps": gaps,
    }
