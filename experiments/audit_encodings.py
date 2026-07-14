# -*- coding: utf-8 -*-
"""
audit_encodings.py — F0.2: Encoding & veri bütünlüğü audit'i (READ-ONLY)
=========================================================================
MODEL_AUDIT_VE_OPTIMIZASYON_PLANI_2026-07-13.md §E-F0.2 spesifikasyonu.

Hiçbir dosyayı değiştirmez. Sadece okur, karşılaştırır, logs/audit_encodings_report.json'a
ve konsola özet yazar. ADM (`data/master.parquet`, `data/weather_history.parquet`) ve GDZ
(`../gdz talep/live/data/feature_matrix.parquet`) üzerinde çalışır.

Kontroller (plan §E-F0.2):
  1. Takvim yeniden-hesap diff'i (ADM): Yıl/Ay/Gün/Ramazan_Bayram/Kurban_Bayram/
     Milli_Bayram/Yilbasi/Secim_Gunu/is_religional_holiday <- Tarih+Saat(0->24) ile
     src/holiday_calendar üzerinden yeniden üretilip yıl-bazında uyumsuzluk raporlanır.
  2. Saat konvansiyonu hizası: sıcak günlerde yük piki <-> sıcaklık piki saat farkı.
  3. Is_Semester/Is_Summer_Break (GDZ) - hardcoded aralıkların iç-tutarlılık kontrolü
     (MEB'in resmi takvimiyle web karşılaştırması bu script'in kapsamı dışında -> not düşülür).
  4. dtype tutarlılığı, duplicate (Tarih,Saat), weather NaN adacıkları.
  5. GDZ: Week yıl-sınırı (ISO hafta 52/53 <-> takvim yılı uyumsuzluğu), hedef kolon
     rename zincirinin non-null bütünlüğü.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ADM_ROOT = Path(__file__).resolve().parent.parent
GDZ_ROOT = ADM_ROOT.parent / "gdz talep"

sys.path.insert(0, str(ADM_ROOT))
from src.holiday_calendar import build_holiday_calendar  # noqa: E402

REPORT: dict = {"generated_at": pd.Timestamp.now().isoformat(), "checks": {}}

# Seçim günleri — ADM holiday_calendar.py'de yok; GDZ src/calendar_features.py'den
# ödünç alındı (aynı kaynak listesi, iki tenant'ta da hardcoded).
ELECTION_DATES = [
    date(2018, 6, 24),
    date(2019, 3, 31),
    date(2023, 5, 14),
    date(2023, 5, 28),
    date(2024, 3, 31),
]


def _recon_datetime(df: pd.DataFrame, date_col: str, hour_col: str) -> pd.Series:
    """Tarih.normalize() + Saat(0->24) düzeltmesi — 03_build_features._backfill_calendar_columns
    ile aynı rekonstrüksiyon mantığı."""
    corrected_hours = df[hour_col].replace(0, 24)
    return pd.to_datetime(df[date_col]).dt.normalize() + pd.to_timedelta(corrected_hours, unit="h")


def check1_calendar_recompute(master: pd.DataFrame) -> dict:
    print("\n[1/5] Takvim yeniden-hesap diff'i (ADM)...")
    recon_dt = _recon_datetime(master, "Tarih", "Saat")
    years = sorted(set(recon_dt.dt.year) | {int(recon_dt.dt.year.min()) - 1, int(recon_dt.dt.year.max()) + 1})
    cal = build_holiday_calendar(years)

    yil_r = recon_dt.dt.year
    ay_r = recon_dt.dt.month
    gun_r = recon_dt.dt.day

    ramazan_r, kurban_r, milli_r, yilbasi_r, secim_r = [], [], [], [], []
    for d in recon_dt.dt.date:
        meta = cal.get(d)
        if meta and meta["category"] == "Ramazan_Bayram":
            ramazan_r.append(meta["holiday_day_number"])
            kurban_r.append(0)
        elif meta and meta["category"] == "Kurban_Bayram":
            ramazan_r.append(0)
            kurban_r.append(meta["holiday_day_number"])
        else:
            ramazan_r.append(0)
            kurban_r.append(0)

        is_official = bool(meta and meta["holiday_type"] == "official")
        milli_r.append(1 if is_official else 0)
        yilbasi_r.append(1 if (meta and meta["holiday_name"] == "Yilbasi") else 0)
        secim_r.append(1 if d in ELECTION_DATES else 0)

    ramazan_r = np.array(ramazan_r)
    kurban_r = np.array(kurban_r)
    is_relig_r = ((ramazan_r > 0) | (kurban_r > 0)).astype(int)

    recomputed = pd.DataFrame({
        "Yıl_r": yil_r.values, "Ay_r": ay_r.values, "Gün_r": gun_r.values,
        "Ramazan_Bayram_r": ramazan_r, "Kurban_Bayram_r": kurban_r,
        "Milli_Bayram_r": milli_r, "Yilbasi_r": yilbasi_r,
        "Secim_Gunu_r": secim_r, "is_religional_holiday_r": is_relig_r,
    }, index=master.index)

    pairs = [
        ("Yıl", "Yıl_r"), ("Ay", "Ay_r"), ("Gün", "Gün_r"),
        ("Ramazan_Bayram", "Ramazan_Bayram_r"), ("Kurban_Bayram", "Kurban_Bayram_r"),
        ("Milli_Bayram", "Milli_Bayram_r"), ("Yilbasi", "Yilbasi_r"),
        ("Secim_Gunu", "Secim_Gunu_r"), ("is_religional_holiday", "is_religional_holiday_r"),
    ]

    result = {"per_column": {}, "per_year_mismatch_rate": {}}
    yil_col = master["Yıl"] if "Yıl" in master.columns else pd.Series(yil_r.values, index=master.index)

    for orig_col, recomp_col in pairs:
        if orig_col not in master.columns:
            result["per_column"][orig_col] = {"status": "MISSING_IN_MASTER"}
            continue
        orig = pd.to_numeric(master[orig_col], errors="coerce")
        recomp = recomputed[recomp_col]
        mismatch = (orig.fillna(-999) != recomp).values
        n_mismatch = int(mismatch.sum())
        n_nan = int(orig.isna().sum())
        result["per_column"][orig_col] = {
            "n_rows": len(orig), "n_mismatch": n_mismatch, "n_nan": n_nan,
            "mismatch_rate": round(n_mismatch / len(orig), 5),
        }
        if n_mismatch > 0:
            by_year = pd.DataFrame({"year": recon_dt.dt.year.values, "mismatch": mismatch}).groupby("year")["mismatch"].mean()
            bad_years = by_year[by_year > 0.01].round(4).to_dict()
            if bad_years:
                result["per_year_mismatch_rate"][orig_col] = {int(k): float(v) for k, v in bad_years.items()}

    print("  Kolon bazlı uyumsuzluk oranları:")
    for col, stats in result["per_column"].items():
        if stats.get("status") == "MISSING_IN_MASTER":
            print(f"    {col}: master'da YOK")
        else:
            print(f"    {col}: mismatch={stats['n_mismatch']}/{stats['n_rows']} ({stats['mismatch_rate']:.3%}), NaN={stats['n_nan']}")
    if result["per_year_mismatch_rate"]:
        print("  Yıl-bazında >%1 uyumsuzluk taşıyan kolon/yıllar:")
        for col, years_d in result["per_year_mismatch_rate"].items():
            print(f"    {col}: {years_d}")
    else:
        print("  Yıl-bazında >%1 uyumsuzluk yok.")

    return result


def check2_hour_convention(master: pd.DataFrame, weather: pd.DataFrame) -> dict:
    print("\n[2/5] Saat konvansiyonu hizası (yük piki <-> sıcaklık piki)...")
    target_col = "ADM_Dağıtılan_Enerji_(MWh)"
    temp_cols = [c for c in weather.columns if c.endswith("_app_temp_actual")]
    weather = weather.copy()
    weather["_mean_temp"] = weather[temp_cols].mean(axis=1)

    m = master[["Tarih", "Saat", target_col]].dropna()
    m["date"] = m["Tarih"].dt.date
    w = weather[["Tarih", "Saat", "_mean_temp"]].dropna()
    w["date"] = w["Tarih"].dt.date

    # Isı sinyalinin en güçlü olduğu 3 gün: yıl-ortalama üstü mean_temp'e sahip, load ile
    # ölçülebilir korelasyonu olan yaz günleri arasından en sıcak 3'ü seç.
    daily_temp = w.groupby("date")["_mean_temp"].mean().sort_values(ascending=False)
    candidate_days = daily_temp.head(10).index.tolist()

    offsets = []
    detail = []
    for d in candidate_days[:5]:
        md = m[m["date"] == d].sort_values("Saat")
        wd = w[w["date"] == d].sort_values("Saat")
        if len(md) < 20 or len(wd) < 20:
            continue
        load_peak_hour = int(md.loc[md[target_col].idxmax(), "Saat"])
        temp_peak_hour = int(wd.loc[wd["_mean_temp"].idxmax(), "Saat"])
        offset = load_peak_hour - temp_peak_hour
        offsets.append(offset)
        detail.append({"date": str(d), "load_peak_hour": load_peak_hour, "temp_peak_hour": temp_peak_hour, "offset_h": offset})

    result = {
        "n_days_checked": len(detail),
        "days": detail,
        "median_offset_h": float(np.median(offsets)) if offsets else None,
        "note": ("Pozitif offset yük pikinin sıcaklık pikinden SONRA geldiğini gösterir "
                 "(normal, AC gecikmesi ~1-3h beklenir). |offset|>=1h sistematik ise ve TÜM "
                 "günlerde AYNI yönde ise saat kayması şüphesi güçlenir."),
    }
    print(f"  Kontrol edilen gün sayısı: {len(detail)}")
    for row in detail:
        print(f"    {row['date']}: yük piki saat={row['load_peak_hour']}, sıcaklık piki saat={row['temp_peak_hour']}, offset={row['offset_h']}h")
    if offsets:
        print(f"  Medyan offset: {result['median_offset_h']}h")
    return result


def check3_semester_ranges() -> dict:
    print("\n[3/5] Is_Semester/Is_Summer_Break (GDZ) iç-tutarlılık kontrolü...")
    calendar_features_path = GDZ_ROOT / "src" / "calendar_features.py"
    result = {"status": "SKIPPED", "reason": "MEB resmi takvimiyle web karşılaştırması bu script kapsamında yapılmadı (internet erişimi gerektirir)."}
    if not calendar_features_path.exists():
        result["status"] = "FILE_NOT_FOUND"
        print(f"  {calendar_features_path} bulunamadı, atlanıyor.")
        return result

    import re
    src = calendar_features_path.read_text(encoding="utf-8")
    sem_match = re.search(r"semester_ranges\s*=\s*\[(.*?)\]", src, re.S)
    sum_match = re.search(r"summer_ranges\s*=\s*\[(.*?)\]", src, re.S)

    def _parse_ranges(block: str) -> list[tuple[str, str]]:
        return re.findall(r"\('([\d-]+)',\s*'([\d-]+)'\)", block)

    sem_ranges = _parse_ranges(sem_match.group(1)) if sem_match else []
    sum_ranges = _parse_ranges(sum_match.group(1)) if sum_match else []

    issues = []
    for start, end in sem_ranges + sum_ranges:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e < s:
            issues.append(f"ters aralık: {start}->{end}")
        dur = (e - s).days
        if dur > 100:
            issues.append(f"olağandışı uzun aralık ({dur} gün): {start}->{end}")

    # Yıllar arası kapsama kontrolü: her yılda tam 1 semester + 1 summer aralığı var mı
    sem_years = sorted({int(s[:4]) for s, _ in sem_ranges})
    sum_years = sorted({int(s[:4]) for s, _ in sum_ranges})

    result = {
        "status": "OK" if not issues else "ISSUES_FOUND",
        "semester_ranges_found": len(sem_ranges),
        "summer_ranges_found": len(sum_ranges),
        "semester_years_covered": sem_years,
        "summer_years_covered": sum_years,
        "issues": issues,
        "note": "MEB resmi takvimiyle web karşılaştırması YAPILMADI — sadece iç-tutarlılık (aralık sırası, süre, yıl kapsaması) kontrol edildi.",
    }
    print(f"  semester_ranges: {len(sem_ranges)} adet, summer_ranges: {len(sum_ranges)} adet")
    print(f"  İç-tutarlılık sorunları: {issues if issues else 'yok'}")
    print("  NOT: MEB resmi takvimiyle web karşılaştırması bu koşuda YAPILMADI.")
    return result


def check4_integrity(master: pd.DataFrame, weather: pd.DataFrame) -> dict:
    print("\n[4/5] dtype tutarlılığı, duplicate, weather NaN adacıkları...")
    result = {}

    dup_mask = master.duplicated(subset=["Tarih", "Saat"], keep=False)
    result["n_duplicate_tarih_saat_rows"] = int(dup_mask.sum())
    if dup_mask.sum() > 0:
        dup_sample = master.loc[dup_mask, ["Tarih", "Saat"]].drop_duplicates().head(10)
        result["duplicate_sample"] = dup_sample.astype(str).to_dict("records")

    target_col = "ADM_Dağıtılan_Enerji_(MWh)"
    actual_rows = master[master[target_col].notna()]
    dtype_issues = {}
    for col in actual_rows.columns:
        if actual_rows[col].dtype == object and col not in ("ÖzelGün_Adı",):
            dtype_issues[col] = str(actual_rows[col].dtype)
    result["unexpected_object_dtype_cols"] = dtype_issues

    weather_cols = [c for c in weather.columns if c not in ("Tarih", "Saat")]
    nan_islands = {}
    for col in weather_cols:
        s = weather[col]
        is_nan = s.isna()
        if not is_nan.any():
            continue
        # ardışık NaN blokları
        grp = (is_nan != is_nan.shift()).cumsum()
        blocks = is_nan.groupby(grp).sum()
        blocks = blocks[blocks > 0]
        max_block = int(blocks.max()) if len(blocks) else 0
        total_nan = int(is_nan.sum())
        if total_nan > 0:
            nan_islands[col] = {"total_nan": total_nan, "max_consecutive": max_block, "n_blocks": int(len(blocks))}
    result["weather_nan_islands"] = nan_islands

    print(f"  Duplicate (Tarih,Saat) satır: {result['n_duplicate_tarih_saat_rows']}")
    print(f"  Beklenmeyen object dtype kolonlar (actual satırlarda): {list(dtype_issues.keys()) if dtype_issues else 'yok'}")
    if nan_islands:
        worst = sorted(nan_islands.items(), key=lambda kv: -kv[1]["total_nan"])[:5]
        print(f"  Weather NaN'i en yüksek 5 kolon: {[(k, v['total_nan'], v['max_consecutive']) for k, v in worst]}")
    else:
        print("  Weather NaN adacığı yok.")
    return result


def check5_gdz(feature_matrix: pd.DataFrame | None) -> dict:
    print("\n[5/5] GDZ Week yıl-sınırı + hedef kolon rename zinciri...")
    result = {}
    if feature_matrix is None:
        result["status"] = "FILE_NOT_FOUND"
        print("  GDZ feature_matrix.parquet bulunamadı, atlanıyor.")
        return result

    date_col = None
    for cand in ("Tarih", "Datetime", "index"):
        if cand in feature_matrix.columns:
            date_col = cand
            break
    if date_col is None and isinstance(feature_matrix.index, pd.DatetimeIndex):
        dt_index = feature_matrix.index
    elif date_col:
        dt_index = pd.to_datetime(feature_matrix[date_col])
    else:
        dt_index = None

    if dt_index is not None and "Week" in feature_matrix.columns:
        iso_year = dt_index.isocalendar().year.values if hasattr(dt_index, "isocalendar") else None
        cal_year = dt_index.year.values if hasattr(dt_index, "year") else None
        week_vals = feature_matrix["Week"].values
        month_vals = feature_matrix["Month"].values if "Month" in feature_matrix.columns else None

        boundary_issue_rows = 0
        if month_vals is not None:
            # Ocak ayında Week>=52 VEYA Aralık ayında Week==1 -> yıl-sınırı karışıklığı adayı
            jan_mask = (month_vals == 1) & (week_vals >= 52)
            dec_mask = (month_vals == 12) & (week_vals == 1)
            boundary_issue_rows = int(jan_mask.sum() + dec_mask.sum())

        result["week_year_boundary_rows"] = boundary_issue_rows
        result["week_range"] = [int(np.nanmin(week_vals)), int(np.nanmax(week_vals))]
        result["note"] = ("Week = df.index.isocalendar().week (ISO hafta), Year kolonu ise "
                           "takvim yılı (calendar year) — ISO yıl DEĞİL. Ocak başında Week>=52 "
                           "veya Aralık sonunda Week==1 görülen satırlar yıl-sınırında ISO hafta "
                           "ile takvim yılı arasında tutarsızlık taşır (feature olarak ham "
                           "kullanılıyorsa model bunu 'yıl sonu' ile karıştırabilir).")
        print(f"  Week aralığı: {result['week_range']}, yıl-sınırı şüpheli satır sayısı: {boundary_issue_rows}")
    else:
        result["status"] = "Week veya tarih kolonu bulunamadı"
        print("  Week veya tarih kolonu bulunamadı.")

    target_candidates = [c for c in feature_matrix.columns if "Dağıtılan" in c or "Enerji" in c or c.lower() in ("target", "y")]
    result["target_column_candidates"] = target_candidates
    if target_candidates:
        tcol = target_candidates[0]
        n_nan = int(feature_matrix[tcol].isna().sum())
        result["target_col_used"] = tcol
        result["target_n_nan"] = n_nan
        result["target_n_rows"] = len(feature_matrix)
        print(f"  Hedef kolon adayı: {tcol}, NaN: {n_nan}/{len(feature_matrix)}")
    else:
        print("  Hedef kolon adayı feature_matrix.parquet'te bulunamadı (postprocess sonrası ayrılmış olabilir).")

    return result


def main():
    print("=" * 70)
    print("F0.2 — Encoding & Veri Bütünlüğü Audit'i (READ-ONLY)")
    print("=" * 70)

    master = pd.read_parquet(ADM_ROOT / "data" / "master.parquet")
    weather = pd.read_parquet(ADM_ROOT / "data" / "weather_history.parquet")

    REPORT["checks"]["1_calendar_recompute_adm"] = check1_calendar_recompute(master)
    REPORT["checks"]["2_hour_convention"] = check2_hour_convention(master, weather)
    REPORT["checks"]["3_semester_ranges_gdz"] = check3_semester_ranges()
    REPORT["checks"]["4_integrity"] = check4_integrity(master, weather)

    gdz_fm_path = GDZ_ROOT / "live" / "data" / "feature_matrix.parquet"
    feature_matrix = pd.read_parquet(gdz_fm_path) if gdz_fm_path.exists() else None
    REPORT["checks"]["5_gdz"] = check5_gdz(feature_matrix)

    out_path = ADM_ROOT / "logs" / "audit_encodings_report.json"
    out_path.parent.mkdir(exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)

    out_path.write_text(json.dumps(REPORT, indent=2, default=_default, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapor yazıldı: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
