# -*- coding: utf-8 -*-
"""
thermal_features.py
===================
Termal adaptasyon, nem etkileşimi ve rüzgar soğutma bastırma feature'ları.

Grup A — Termal Adaptasyon (app_temp bazlı, tamamen implemente)
Grup B — Nem Etkileşimi (RH kolonu varsa tam, yoksa cloud-proxy ile çalışır)
Grup C — Rüzgar Soğutma Bastırma (wind_speed kolonu varsa aktif, yoksa skip)

Tüm rolling hesaplamalar shift(1) kullanır — leakage yok.
data_manager.py'dan çağrılır:
    from src.thermal_features import add_thermal_features
    df = add_thermal_features(df)
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Yardımcı fonksiyonlar
# ═══════════════════════════════════════════════════════════════════════════════

def _streak(binary: pd.Series) -> pd.Series:
    """
    Vectorized art arda True sayaci. Sifirlanir False goruldugunde.
    Ornek: [F,T,T,T,F,T,T] -> [0,1,2,3,0,1,2]
    """
    groups = (~binary).cumsum()
    return (binary.groupby(groups).cumsum() * binary).astype(int)


def _daily_to_hourly(daily_series: pd.Series, hourly_index: pd.DatetimeIndex) -> pd.Series:
    """
    Gunluk index'li seriyi saatlik index'e map'ler.
    daily_series index'i midnight (00:00) olmali (resample('D') cikisi gibi).
    """
    return hourly_index.floor("D").map(daily_series)


# ═══════════════════════════════════════════════════════════════════════════════
# Grup A — Termal Adaptasyon
# ═══════════════════════════════════════════════════════════════════════════════

def _add_group_a(df: pd.DataFrame, base_col: str,
                 hdd_col: str, cdd_col: str, cfg) -> tuple:
    """
    Termal adaptasyon feature'lari.
    Returns: (df, list_of_added_feature_names)
    """
    added = []

    if base_col not in df.columns:
        print(f"[ThermalFeatures-A] UYARI: {base_col} bulunamadi, Grup A atlanıyor.")
        return df, added

    T = df[base_col]  # saatlik apparent temperature (Mugla il ort.)
    HDD = df[hdd_col] if hdd_col in df.columns else None
    CDD = df[cdd_col] if cdd_col in df.columns else None

    # ── A1: 30 günlük sıcaklık anomalisi ────────────────────────────────────
    # Bugünkü T - son 30 günün saatlik ortalaması (shift ile leakage yok)
    rolling_mean_30d = T.shift(1).rolling(window=30 * 24, min_periods=168).mean()
    df["Temp_Anomaly_30d"] = T - rolling_mean_30d
    added.append("Temp_Anomaly_30d")

    # ── A2 & A3: Kümülatif CDD/HDD son 72 saat ──────────────────────────────
    if CDD is not None:
        df["CDD_72h_Cum"] = CDD.shift(1).rolling(window=72, min_periods=24).sum()
        added.append("CDD_72h_Cum")

    if HDD is not None:
        df["HDD_72h_Cum"] = HDD.shift(1).rolling(window=72, min_periods=24).sum()
        added.append("HDD_72h_Cum")

    # ── A4: Art arda sıcak/soğuk gün sayısı ─────────────────────────────────
    if CDD is not None:
        daily_cdd = CDD.resample("D").sum()                      # günlük toplam CDD
        hot_flag  = (daily_cdd.shift(1) > cfg.CDD_STREAK_THRESHOLD)  # dün CDD > esik?
        streak_hot = _streak(hot_flag)

        df["Consecutive_Hot_Days"]  = _daily_to_hourly(streak_hot, df.index)
        added.append("Consecutive_Hot_Days")

    if HDD is not None:
        daily_hdd  = HDD.resample("D").sum()
        cold_flag  = (daily_hdd.shift(1) > cfg.HDD_STREAK_THRESHOLD)
        streak_cold = _streak(cold_flag)

        df["Consecutive_Cold_Days"] = _daily_to_hourly(streak_cold, df.index)
        added.append("Consecutive_Cold_Days")

    # ── A5: Sezonal başlangıç dummy (ilk CDD>5 gününden sonraki 7 gün) ──────
    if CDD is not None:
        daily_cdd_shifted = CDD.resample("D").sum().shift(1)  # leakage-free günlük CDD

        onset_flag = pd.Series(0, index=daily_cdd_shifted.index, dtype="int8")
        years = daily_cdd_shifted.index.year.unique()
        for year in years:
            year_series = daily_cdd_shifted[daily_cdd_shifted.index.year == year]
            hot_days = year_series[year_series > cfg.CDD_STREAK_THRESHOLD]
            if len(hot_days) > 0:
                onset_date = hot_days.index[0]
                window = pd.date_range(onset_date, periods=7, freq="D")
                mask = onset_flag.index.isin(window)
                onset_flag.loc[mask] = 1

        df["Season_Onset_Cooling"] = _daily_to_hourly(onset_flag, df.index).fillna(0).astype("int8")
        added.append("Season_Onset_Cooling")

    # ── A6: Tropikal gece flag ───────────────────────────────────────────────
    # Onceki gecenin (00:00-06:00) min sicakligi esigi astiysa = 1
    night_mask = df.index.hour <= 6
    night_min_daily = df.loc[night_mask, base_col].resample("D").min()
    night_min_yesterday = night_min_daily.shift(1)
    tropical = (night_min_yesterday > cfg.TROPICAL_NIGHT_THRESHOLD).astype("int8")
    df["Tropical_Night"] = _daily_to_hourly(tropical, df.index).fillna(0).astype("int8")
    added.append("Tropical_Night")

    # ── A7: Günlük sıcaklık genliği (dünün Tmax - Tmin) ─────────────────────
    daily_max = T.resample("D").max().shift(1)
    daily_min = T.resample("D").min().shift(1)
    daily_range = daily_max - daily_min
    df["Daily_Temp_Range"] = _daily_to_hourly(daily_range, df.index)
    added.append("Daily_Temp_Range")

    return df, added


# ═══════════════════════════════════════════════════════════════════════════════
# Grup B — Nem Etkileşimi
# ═══════════════════════════════════════════════════════════════════════════════

def _add_group_b(df: pd.DataFrame, base_col: str, cdd_col: str, cfg) -> tuple:
    """
    Nem etkileşim feature'lari.

    RH (relative humidity) kolonu varsa: dew_point, wet_bulb, CDD*RH vb.
    Yoksa: bulut örtüsü tabanlı proxy (Cloud_pct mevcut olduğunda).
    """
    added = []

    T = df.get(base_col)
    CDD = df.get(cdd_col)

    # Nem kolonu ara — çeşitli isim formatlarını kontrol et
    rh_candidates = [
        "RH", "Humidity", "relative_humidity", "humidity",
        "MUGLA_MenteseCenter_humidity", "Hissedilen_Nem",
    ]
    rh_col = next((c for c in rh_candidates if c in df.columns), None)

    # Bulut örtüsü ortalaması (proxy)
    cloud_cols = [c for c in df.columns if c.endswith("_cloud_actual")]
    cloud_mean = df[cloud_cols].mean(axis=1) if cloud_cols else None

    has_rh = rh_col is not None

    if has_rh:
        # ── B1: Dew point (Magnus yaklaşımı) ────────────────────────────────
        RH = df[rh_col]
        df["Dew_Point"] = T - ((100 - RH) / 5.0)
        added.append("Dew_Point")

        # ── B2: Wet-bulb temperature (Stull 2011 yaklaşımı) ─────────────────
        # Tw = T*atan(0.151977*(RH+8.313659)^0.5) + atan(T+RH)
        #      - atan(RH-1.676331) + 0.00391838*RH^1.5 * atan(0.023101*RH) - 4.686035
        import math
        Tarr  = T.values
        RHarr = RH.values
        wb = (
            Tarr * np.arctan(0.151977 * (RHarr + 8.313659) ** 0.5)
            + np.arctan(Tarr + RHarr)
            - np.arctan(RHarr - 1.676331)
            + 0.00391838 * RHarr ** 1.5 * np.arctan(0.023101 * RHarr)
            - 4.686035
        )
        df["Wet_Bulb_Temp"] = pd.Series(wb, index=df.index)
        added.append("Wet_Bulb_Temp")

        # ── B3: CDD × nem etkileşim ──────────────────────────────────────────
        if CDD is not None:
            df["CDD_x_Humidity"] = CDD * RH
            added.append("CDD_x_Humidity")

            df["CDD_x_DewPoint"] = CDD * df["Dew_Point"]
            added.append("CDD_x_DewPoint")

        # ── B4: Yüksek nem + sıcak flag ─────────────────────────────────────
        df["High_Humidity_Hot"] = (
            (RH > cfg.HIGH_HUMIDITY_THRESHOLD) & (T > cfg.HOT_THRESHOLD)
        ).astype("int8")
        added.append("High_Humidity_Hot")

    else:
        # ── B-proxy: Bulut örtüsü tabanlı ────────────────────────────────────
        # Yüksek bulut + yüksek sıcaklık = güneş yok ama boğucu sıcak
        if cloud_mean is not None and T is not None:
            df["Cloud_x_Temp"] = cloud_mean * T
            added.append("Cloud_x_Temp")

            if CDD is not None:
                df["CDD_x_Cloud"] = CDD * cloud_mean
                added.append("CDD_x_Cloud")

            df["Overcast_Hot"] = (
                (cloud_mean > 70) & (T > cfg.HOT_THRESHOLD)
            ).astype("int8")
            added.append("Overcast_Hot")

        if T is not None and CDD is not None:
            # GHI-bazli sıcaklık etkileşim (GHI düşük + sıcak = sisli/nemli)
            for ghi_col in ["GHI_ADM_Weighted", "GHI_Mugla_W_m2"]:
                if ghi_col in df.columns:
                    # Normalize GHI 0-1 arası (max 1000 W/m2 referans)
                    ghi_norm = df[ghi_col].clip(0, 1000) / 1000.0
                    df["CDD_x_LowGHI"] = CDD * (1.0 - ghi_norm)
                    added.append("CDD_x_LowGHI")
                    break  # Sadece bir tane ekle

        if not added:
            print("[ThermalFeatures-B] RH kolonu yok, bulut proxy da hesplanamadı — Grup B atlandı.")

    return df, added


# ═══════════════════════════════════════════════════════════════════════════════
# Grup C — Rüzgar Soğutma Bastırma
# ═══════════════════════════════════════════════════════════════════════════════

def _add_group_c(df: pd.DataFrame, cdd_col: str, cfg) -> tuple:
    """
    Rüzgar soğutma bastırma feature'lari.
    wind_speed kolonu yoksa gracefully skip edilir.
    """
    added = []

    wind_candidates = [
        "wind_speed", "WindSpeed", "wind_speed_10m",
        "MUGLA_MenteseCenter_wind_speed", "Ruzgar_Hizi",
    ]
    wind_col = next((c for c in wind_candidates if c in df.columns), None)

    if wind_col is None:
        print("[ThermalFeatures-C] wind_speed kolonu bulunamadi - Grup C atlandi.")
        print("   Grup C icin Open-Meteo'dan wind_speed_10m cekilip Excel'e eklenmeli.")
        return df, added

    W = df[wind_col]
    T = df.get("Hissedilen_Sıcaklık_Mean_MUGLA")
    CDD = df.get(cdd_col)

    # ── C1: Rüzgar × CDD etkileşim ──────────────────────────────────────────
    if CDD is not None:
        df["Wind_x_CDD"] = W * CDD
        added.append("Wind_x_CDD")

    # ── C2: Durgun sıcak hava flag ───────────────────────────────────────────
    if T is not None:
        df["Still_Hot_Air"] = (
            (T > cfg.HOT_THRESHOLD) & (W < cfg.WIND_STILL_THRESHOLD)
        ).astype("int8")
        added.append("Still_Hot_Air")

    # ── C3: Rüzgar lag'i (rüzgar hızı kendi başına geciktirilmiş) ───────────
    df["Wind_Lag3h"]  = W.shift(3)
    df["Wind_Lag24h"] = W.shift(24)
    added.extend(["Wind_Lag3h", "Wind_Lag24h"])

    return df, added


# ═══════════════════════════════════════════════════════════════════════════════
# Ana giriş noktası
# ═══════════════════════════════════════════════════════════════════════════════

def add_thermal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thermal feature engineering ana fonksiyonu.
    config.py'dan flag'leri okur; her grup ayrı ayrı kontrol edilir.

    Cagri:
        from src.thermal_features import add_thermal_features
        df = add_thermal_features(df)
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import config_live as cfg
    except ImportError:
        print("[ThermalFeatures] config.py import hatası — tüm gruplar atlandı.")
        return df

    if not getattr(cfg, "ENABLE_THERMAL_FEATURES", False):
        return df

    print("[ThermalFeatures] Feature engineering başlıyor...")

    base_col = "Hissedilen_Sıcaklık_Mean_MUGLA"
    hdd_col  = "HDD_Heating_Stress"
    cdd_col  = "CDD_Cooling_Stress"
    total_added = []

    if getattr(cfg, "ENABLE_THERMAL_ADAPTATION", True):
        df, feats = _add_group_a(df, base_col, hdd_col, cdd_col, cfg)
        total_added.extend(feats)
        print(f"[ThermalFeatures-A] {len(feats)} özellik eklendi: {feats}")

    if getattr(cfg, "ENABLE_HUMIDITY_FEATURES", True):
        df, feats = _add_group_b(df, base_col, cdd_col, cfg)
        total_added.extend(feats)
        print(f"[ThermalFeatures-B] {len(feats)} özellik eklendi: {feats}")

    if getattr(cfg, "ENABLE_WIND_INTERACTION", True):
        df, feats = _add_group_c(df, cdd_col, cfg)
        total_added.extend(feats)
        if feats:
            print(f"[ThermalFeatures-C] {len(feats)} özellik eklendi: {feats}")

    print(f"[ThermalFeatures] Toplam {len(total_added)} yeni özellik eklendi.")
    return df
