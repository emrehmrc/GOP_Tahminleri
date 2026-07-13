"""
tenant_config.py — iki tenant'ı (ADM/GDZ) parametrize eden tek dataclass.

Her tenant'ın kendi config_live*.py'si bir TenantConfig instance'ı export eder
(bkz. adm live/config_live.py:TENANT, gdz talep/live/config_live_gdz.py:TENANT).
monitoring/ paketindeki paylaşımlı fonksiyonlar `import config_live as C` YERİNE
bu instance'ı parametre olarak alır — tek kod, iki farklı config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TenantConfig:
    edas_id: str
    logger_name: str

    log_root: Path
    archive_dir: Path
    weather_history_parquet: Path
    known_events_csv: Path
    alerts_dir: Path
    log_backup_dir: Path

    weather_station_temp_cols: list[str] = field(default_factory=list)

    # Faz 2b-4 (2026-07-13): sadece BETİMLEYİCİ metadata — hangi tenant'ın
    # feature mühendisliği hangi sinyale daha çok ağırlık veriyor (bkz. 12 Temmuz
    # post-mortem bulguları). Şu an hiçbir koda davranışsal etkisi YOK; ADM ve
    # GDZ'nin pipeline/03_build_features.py + 04_predict_48h.py'si hâlâ tamamen
    # ayrı codebase (bkz. MASTER_PLAN.md Faz 2b-4 notu) — bu alan sadece Faz 3
    # ortaklaştırması sırasında referans/dokümantasyon amaçlı.
    feature_profile: str = "unspecified"

    # GDZ'de "T1"=issue günü (horizon_days=0), ADM'de "T+1"=issue+1 (horizon_days=1).
    # horizon_day STRING alanı (geriye-uyumluluk) bu farkla üretilir:
    # label = f"T+{horizon_days + horizon_day_label_offset}".
    horizon_day_label_offset: int = 0

    z_baseline_window_days: int = 30
    z_warmup_min_days: int = 30
    z_threshold: float = 3.0
    scorecard_rebuild_window_days: int = 400
    scorecard_windows: tuple[int, ...] = (7, 30, 365)
    headline_horizon: str = "T+2"

    @property
    def forecast_log_dir(self) -> Path:
        return self.log_root / "forecast_log"

    @property
    def actuals_log_dir(self) -> Path:
        return self.log_root / "actuals_log"

    @property
    def monitoring_db(self) -> Path:
        return self.log_root / "monitoring.duckdb"

    @property
    def gaps_dir(self) -> Path:
        """Faz 3: reconcile()'ın gerçek-kayıp (arşiv de yok) raporları — <date>.json."""
        return self.log_root / "gaps"
