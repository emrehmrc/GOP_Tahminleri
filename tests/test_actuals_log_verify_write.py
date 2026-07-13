"""Faz 2 (2026-07-13) — actuals_log write-then-verify testleri.

Kök neden bulunamadı ama gözlem: 2026-07-11/07-12 için ActualsLog "N satır upsert
edildi" logu BAŞARI gösterdi (exception yok), dosya sonradan diskte yoktu.
`upsert_actuals` artık forecast_log'daki gibi yazımdan hemen sonra diskten geri
okuyup satır sayısını doğruluyor -- "exception fırlamadı" tek başına yeterli
güvence değil.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from monitoring.forecast_logger import upsert_actuals, upsert_by_date
from monitoring.tenant_config import TenantConfig


def _config(tmp_path: Path) -> TenantConfig:
    return TenantConfig(
        edas_id="TST", logger_name="test_adm_live", log_root=tmp_path,
        archive_dir=tmp_path / "archive", weather_history_parquet=tmp_path / "wh.parquet",
        known_events_csv=tmp_path / "known_events.csv", alerts_dir=tmp_path / "alerts",
        log_backup_dir=tmp_path / "backup",
    )


def _updates(target_date: str, n: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "target_ts": pd.date_range(f"{target_date} 00:00", periods=n, freq="h"),
        "y_actual": [1000.0 + i for i in range(n)],
        "data_quality_flag": [""] * n,
        "known_event": [None] * n,
    })


def test_upsert_actuals_happy_path_writes_and_verifies(tmp_path):
    config = _config(tmp_path)
    upsert_actuals(config, "2026-07-11", _updates("2026-07-11"))

    path = config.actuals_log_dir / "edas_id=TST" / "target_date=2026-07-11.parquet"
    assert path.exists()
    assert len(pd.read_parquet(path)) == 24


def test_upsert_actuals_raises_when_write_silently_lost(tmp_path, monkeypatch):
    """_write_typed_parquet çağrısı 'başarılı' dönse bile dosya diskte yoksa
    (tam da 07-11/07-12 olayının gözlemi) artık sessizce geçilmiyor."""
    config = _config(tmp_path)

    import monitoring.forecast_logger as fl
    monkeypatch.setattr(fl, "_write_typed_parquet", lambda df, schema, path: None)

    with pytest.raises(RuntimeError, match="yazım başarısız"):
        upsert_actuals(config, "2026-07-11", _updates("2026-07-11"))


def test_upsert_actuals_raises_when_row_count_mismatch(tmp_path, monkeypatch):
    """Dosya oluşuyor ama beklenenden az satır içeriyorsa da hata fırlatılmalı."""
    config = _config(tmp_path)

    import monitoring.forecast_logger as fl
    real_write = fl._write_typed_parquet

    def _truncated_write(df, schema, path):
        real_write(df.iloc[:1], schema, path)

    monkeypatch.setattr(fl, "_write_typed_parquet", _truncated_write)

    with pytest.raises(RuntimeError, match="doğrulama hatası"):
        upsert_actuals(config, "2026-07-11", _updates("2026-07-11"))


def test_upsert_by_date_propagates_verification_failure(tmp_path, monkeypatch):
    config = _config(tmp_path)

    import monitoring.forecast_logger as fl
    monkeypatch.setattr(fl, "_write_typed_parquet", lambda df, schema, path: None)

    updates = _updates("2026-07-11")
    with pytest.raises(RuntimeError):
        upsert_by_date(config, updates)
