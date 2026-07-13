# -*- coding: utf-8 -*-
import os, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.max_open_warning": 0, "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11})

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ADM_MASTER = os.path.join(BASE, "data", "master.parquet")
GDZ_MASTER = os.path.join(BASE, "..", "gdz talep", "Input", "GDZ_MASTER.parquet")
GDZ_WEATHER = os.path.join(BASE, "..", "gdz talep", "Output", "GDZ_WEATHER.parquet")
GDZ_PREPARED = os.path.join(BASE, "..", "gdz talep", "Output", "GDZ_PREPARED.parquet")
OUTPUT_DIR = os.path.join(BASE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Station groups ─────────────────────────────────────────────────────────
ADM_STATIONS = [
    "MUGLA_MenteseCenter", "MUGLA_MilasIndustrial", "MUGLA_YataganIndustrial",
    "MUGLA_SandrasHighAlt", "MUGLA_DalamanPlain", "MUGLA_BodrumCenter",
    "DENIZLI_Honaz", "DENIZLI_OSB", "DENIZLI_Merkez", "DENIZLI_IsikliCivril",
    "AYDIN_Merkez", "AYDIN_OSB", "AYDIN_BuyukMenderes", "AYDIN_BozdoganMadran",
]
ADM_TEMP_COLS = [f"{s}_app_temp_actual" for s in ADM_STATIONS]
MUGLA_STATIONS = [s for s in ADM_STATIONS if s.startswith("MUGLA")]
DENIZLI_STATIONS = [s for s in ADM_STATIONS if s.startswith("DENIZLI")]
AYDIN_STATIONS = [s for s in ADM_STATIONS if s.startswith("AYDIN")]

# GDZ station groups (19 stations: Izmir 12 + Manisa 7)
GDZ_STATIONS = [
    "IZMIR_Merkez", "IZMIR_Aliaga_Sanayi", "IZMIR_Bornova", "IZMIR_Cigli_AOSB",
    "IZMIR_Gaziemir", "IZMIR_Torbali_Sanayi", "IZMIR_Kemalpasa_Sanayi",
    "IZMIR_Cesme_Kiyi", "IZMIR_Bergama", "IZMIR_Odemis_Ic", "IZMIR_Menemen",
    "IZMIR_Karsiyaka", "MANISA_Merkez", "MANISA_Turgutlu", "MANISA_Salihli",
    "MANISA_Akhisar", "MANISA_Soma", "MANISA_Alasehir", "MANISA_OSB",
]
GDZ_TEMP_COLS = [f"{s}_app_temp_actual" for s in GDZ_STATIONS]
IZMIR_STATIONS = [s for s in GDZ_STATIONS if s.startswith("IZMIR")]
MANISA_STATIONS = [s for s in GDZ_STATIONS if s.startswith("MANISA")]


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
def load_adm():
    df = pd.read_parquet(ADM_MASTER)
    df["datetime"] = pd.to_datetime(df["Tarih"]) + pd.to_timedelta(df["Saat"], unit="h")
    df = df.set_index("datetime").sort_index()
    df["load_mwh"] = df["ADM_Dağıtılan_Enerji_(MWh)"]
    df["temp_mean"] = df[ADM_TEMP_COLS].mean(axis=1)
    for reg, cols in [("MUGLA", MUGLA_STATIONS), ("DENIZLI", DENIZLI_STATIONS), ("AYDIN", AYDIN_STATIONS)]:
        reg_cols = [f"{s}_app_temp_actual" for s in cols]
        df[f"temp_{reg}"] = df[reg_cols].mean(axis=1)
    return df


def load_gdz():
    prep = pd.read_parquet(GDZ_PREPARED)
    if "Saat" not in prep.columns and "Hour" in prep.columns:
        prep = prep.rename(columns={"Hour": "Saat"})
    prep["load_mwh"] = prep["GDZ_Dagitilan_Enerji_MWh"]
    prep["temp_mean"] = prep["GDZ_Mean_app_temp_actual"]
    prep["temp_IZMIR"] = prep["IZMIR_app_temp_actual"]
    prep["temp_MANISA"] = prep["MANISA_app_temp_actual"]
    if "Weekday" in prep.columns:
        prep["Haftanın_Günü"] = prep["Weekday"]
    if "Is_Weekend" in prep.columns:
        prep["is_weekend"] = prep["Is_Weekend"].astype(bool)
    return prep.sort_index()


# ═══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
def daily_agg(df, load_col="load_mwh", temp_col="temp_mean"):
    daily = df.resample("D").agg({load_col: "mean", temp_col: "mean"})
    daily["load_sum"] = df.resample("D")[load_col].sum()
    return daily.dropna()


def lowess_smooth(x, y, frac=0.15):
    n = len(x)
    if n < 10:
        return y.copy()
    k = max(3, int(n * frac))
    k = min(k, n)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    smoothed = uniform_filter1d(ys.astype(float), size=k, mode="reflect")
    inv_order = np.argsort(order)
    return smoothed[inv_order]


def piecewise_breakpoint(x, y, min_temp=10, max_temp=35):
    mask = (x >= min_temp) & (x <= max_temp)
    xx, yy = x[mask], y[mask]
    if len(xx) < 20:
        return None, None, None
    sort_idx = np.argsort(xx)
    xx_s, yy_s = xx[sort_idx], yy[sort_idx]

    def mse_at(breakpoint):
        left = yy_s[xx_s <= breakpoint]
        right = yy_s[xx_s > breakpoint]
        if len(left) < 5 or len(right) < 5:
            return 1e12
        left_mean, right_mean = np.mean(left), np.mean(right)
        return np.sum((left - left_mean) ** 2) + np.sum((right - right_mean) ** 2)

    res = minimize_scalar(mse_at, bounds=(min_temp + 3, max_temp - 3), method="bounded")
    bp = res.x
    left_slope = np.polyfit(xx_s[xx_s <= bp], yy_s[xx_s <= bp], 1)[0] if sum(xx_s <= bp) > 5 else 0
    right_slope = np.polyfit(xx_s[xx_s > bp], yy_s[xx_s > bp], 1)[0] if sum(xx_s > bp) > 5 else 0
    return bp, left_slope, right_slope


def cdd(series, threshold):
    return np.maximum(0, series - threshold)


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1: Temperature-Consumption Relationship + Piecewise
# ═══════════════════════════════════════════════════════════════════════════
def section_1_temp_consumption(adm, gdz):
    print("=" * 60)
    print("SECTION 1: SICAKLIK-TUKETIM ILISKISI (Scatter + Piecewise)")
    print("=" * 60)
    results = {}

    for label, df in [("ADM", adm), ("GDZ", gdz)]:
        daily = daily_agg(df)
        x, y = daily["temp_mean"].values, daily["load_mwh"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        if len(x) == 0:
            continue

        bp, ls, rs = piecewise_breakpoint(x, y)
        results[label] = {"breakpoint": bp, "left_slope": ls, "right_slope": rs}
        if bp:
            results[label]["sensitivity_MWh_per_C"] = rs
            print(f"  {label}: Breakpoint={bp:.1f}C, Left slope={ls:.3f}, Right slope={rs:.3f} (cooling sensitivity)")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x, y, s=3, alpha=0.3, c="#2c7bb6", label="Günlük veri")
        smooth = lowess_smooth(x, y, frac=0.12)
        order = np.argsort(x)
        ax.plot(x[order], smooth[order], color="#d7191c", lw=2.5, label="LOWESS trend")
        if bp:
            ax.axvline(bp, color="green", ls="--", lw=1.5, label=f"Kırılım noktası = {bp:.1f}°C")
        ax.set_xlabel("Ortalama Hissedilen Sıcaklık (°C)")
        ax.set_ylabel("Günlük Ortalama Tüketim (MWh)")
        ax.set_title(f"{label} — Sıcaklık vs Tüketim")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        savefig(f"sec1_scatter_{label}.png")
        print(f"    -> output/sec1_scatter_{label}.png")

        # Cooling sensitivity by temp bins
        bins = np.arange(0, 45, 2)
        daily["temp_bin"] = pd.cut(daily["temp_mean"], bins)
        bin_stats = daily.groupby("temp_bin", observed=False)["load_mwh"].agg(["mean", "std", "count"])
        bin_centers = [interval.mid for interval in bin_stats.index]
        bin_stats["temp_center"] = bin_centers
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(bin_stats["temp_center"], bin_stats["mean"], yerr=bin_stats["std"],
                     fmt="o-", capsize=3, color="#2c7bb6", markersize=4)
        ax.set_xlabel("Sıcaklık (°C)")
        ax.set_ylabel("Ortalama Tüketim (MWh)")
        ax.set_title(f"{label} — Sıcaklık Dilimi Ortalama Tüketim")
        ax.grid(True, alpha=0.3)
        savefig(f"sec1_temp_bin_{label}.png")
        print(f"    -> output/sec1_temp_bin_{label}.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2: Hourly Cooling Load Profile
# ═══════════════════════════════════════════════════════════════════════════
def section_2_cooling_profile(adm, gdz):
    print("\n" + "=" * 60)
    print("SECTION 2: SAATLIK KLIMA PROFILI (Cooling Load Shape)")
    print("=" * 60)
    results = {}

    for label, df in [("ADM", adm), ("GDZ", gdz)]:
        daily_mean_temp = df["temp_mean"].resample("D").mean()
        hot_days = daily_mean_temp[daily_mean_temp > 24].index.date
        cool_days = daily_mean_temp[daily_mean_temp < 20].index.date

        if len(hot_days) < 3 or len(cool_days) < 3:
            print(f"  {label}: Yetersiz sıcak/serin gün sayısı (hot={len(hot_days)}, cool={len(cool_days)})")
            continue

        df["date"] = df.index.date
        hot_profile = df[df["date"].isin(hot_days)].groupby("Saat")["load_mwh"].mean()
        cool_profile = df[df["date"].isin(cool_days)].groupby("Saat")["load_mwh"].mean()
        cooling_load = hot_profile - cool_profile

        results[label] = {"hot_profile": hot_profile, "cool_profile": cool_profile, "cooling_load": cooling_load}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(hot_profile.index, hot_profile.values, "o-", color="#d7191c", lw=2, label=f"Sıcak günler (>24°C, n={len(hot_days)})")
        ax.plot(cool_profile.index, cool_profile.values, "o-", color="#2c7bb6", lw=2, label=f"Serin günler (<20°C, n={len(cool_days)})")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Ortalama Tüketim (MWh)")
        ax.set_title(f"{label} — Sıcak vs Serin Gün Saatlik Profili")
        ax.legend(fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.bar(cooling_load.index, cooling_load.values, color="#fdae61", width=0.8, alpha=0.85)
        ax.set_xlabel("Saat")
        ax.set_ylabel("Klima Yükü (MWh) = Sıcak - Serin")
        ax.set_title(f"{label} — Klima Profili (Cooling Load Shape)")
        ax.set_xticks(range(0, 24, 2))
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(True, alpha=0.3)

        peak_hour = cooling_load.idxmax()
        peak_val = cooling_load.max()
        ax.annotate(f"Tepe: {peak_hour}:00 ({peak_val:.0f} MWh)",
                    xy=(peak_hour, peak_val), fontsize=9, color="darkred",
                    xytext=(peak_hour + 2, peak_val * 0.85),
                    arrowprops=dict(arrowstyle="->", color="darkred"))
        print(f"  {label} klima profili tepe saati: {peak_hour}:00 ({peak_val:.1f} MWh)")
        savefig(f"sec2_hourly_profile_{label}.png")
        print(f"    -> output/sec2_hourly_profile_{label}.png")

        # Save cooling load vector
        cooling_vec = cooling_load.to_frame(name="cooling_load_mwh")
        cooling_vec.index.name = "hour"
        cooling_vec.to_csv(os.path.join(OUTPUT_DIR, f"klima_profili_{label}.csv"))
        print(f"    -> Klima profili vektörü kaydedildi: output/klima_profili_{label}.csv")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3: Regional Temperature Effect
# ═══════════════════════════════════════════════════════════════════════════
def section_3_regional_effect(adm, gdz):
    print("\n" + "=" * 60)
    print("SECTION 3: BOLGESEL SICAKLIK ETKISI")
    print("=" * 60)
    results = {}

    # ADM regional analysis
    print("  ADM Bölgesel:")
    daily_adm = adm.resample("D").agg({c: "mean" for c in ADM_TEMP_COLS + ["load_mwh"]})
    corrs = {}
    for s in ADM_STATIONS:
        col = f"{s}_app_temp_actual"
        if col in daily_adm.columns:
            c = daily_adm[col].corr(daily_adm["load_mwh"])
            corrs[s] = c
    sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
    for station, corr_val in sorted_corrs[:5]:
        print(f"    {station}: r={corr_val:.4f}")
    results["ADM_station_corrs"] = dict(sorted_corrs)

    # Regional averages for ADM
    for region, stations in [("MUGLA", MUGLA_STATIONS), ("DENIZLI", DENIZLI_STATIONS), ("AYDIN", AYDIN_STATIONS)]:
        cols = [f"{s}_app_temp_actual" for s in stations]
        cols = [c for c in cols if c in daily_adm.columns]
        if cols:
            daily_adm[f"temp_{region}"] = daily_adm[cols].mean(axis=1)

    idx_name = daily_adm.index.name or "datetime"
    daily_adm_long = daily_adm.reset_index().melt(
        id_vars=[idx_name, "load_mwh"],
        value_vars=["temp_MUGLA", "temp_DENIZLI", "temp_AYDIN"],
        var_name="region", value_name="temp"
    )
    daily_adm_long["region"] = daily_adm_long["region"].str.replace("temp_", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    for region in ["MUGLA", "DENIZLI", "AYDIN"]:
        sub = daily_adm_long[daily_adm_long["region"] == region]
        ax.scatter(sub["temp"], sub["load_mwh"], s=2, alpha=0.2, label=region)
    ax.set_xlabel("Bölgesel Ortalama Sıcaklık (°C)")
    ax.set_ylabel("Günlük Tüketim (MWh)")
    ax.set_title("ADM — Bölgesel Sıcaklık vs Tüketim")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    stations_short = [s.replace("_app_temp_actual", "").replace("_", "\n") for s, _ in sorted_corrs[:14]]
    corr_vals = [v for _, v in sorted_corrs[:14]]
    colors = ["#d7191c" if abs(v) > 0.3 else "#fdae61" for v in corr_vals]
    ax.barh(range(len(stations_short)), corr_vals, color=colors, alpha=0.8)
    ax.set_yticks(range(len(stations_short)))
    ax.set_yticklabels(stations_short, fontsize=7)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Korelasyon (r)")
    ax.set_title("ADM — İstasyon Bazında Sıcaklık-Tüketim Korelasyonu")
    ax.grid(True, alpha=0.3, axis="x")
    savefig("sec3_regional_adm.png")
    print("    -> output/sec3_regional_adm.png")

    # GHI effect (ADM)
    daily_adm_ghi = adm.resample("D").agg({c: "mean" for c in ["GHI_ADM_Weighted", "load_mwh", "temp_mean"]})
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(daily_adm_ghi["temp_mean"], daily_adm_ghi["load_mwh"],
                    c=daily_adm_ghi["GHI_ADM_Weighted"], s=8, alpha=0.5, cmap="viridis")
    cbar = plt.colorbar(sc, ax=ax, label="GHI (W/m²)")
    ax.set_xlabel("Ortalama Sıcaklık (°C)")
    ax.set_ylabel("Günlük Tüketim (MWh)")
    ax.set_title("ADM — Sıcaklık, GHI ve Tüketim İlişkisi")
    ax.grid(True, alpha=0.3)
    savefig("sec3_ghi_effect_adm.png")
    print("    -> output/sec3_ghi_effect_adm.png")

    # Partial correlation: GHI effect controlling for temp
    resid_temp = daily_adm_ghi["load_mwh"] - lowess_smooth(
        daily_adm_ghi["temp_mean"].values, daily_adm_ghi["load_mwh"].values, frac=0.2
    )
    ghi_resid_corr = pd.Series(resid_temp).corr(daily_adm_ghi["GHI_ADM_Weighted"])
    print(f"    GHI'nin sıcaklık-sonrası rezidüel korelasyonu: {ghi_resid_corr:.4f}")
    results["ADM_GHI_partial_corr"] = ghi_resid_corr

    # GDZ regional (Izmir vs Manisa)
    if gdz is not None and "temp_mean" in gdz.columns:
        print("  GDZ Bölgesel:")
        daily_gdz = gdz.resample("D").agg({c: "mean" for c in ["IZMIR_app_temp_actual", "MANISA_app_temp_actual", "load_mwh"] if c in gdz.columns})
        corrs_gdz = {}
        for region in ["IZMIR", "MANISA"]:
            col = f"{region}_app_temp_actual"
            if col in daily_gdz.columns:
                cv = daily_gdz[col].corr(daily_gdz["load_mwh"])
                corrs_gdz[region] = cv
                print(f"    {region}: r={cv:.4f}")
        results["GDZ_region_corrs"] = corrs_gdz

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4: Weekday / Weekend AC Difference
# ═══════════════════════════════════════════════════════════════════════════
def section_4_weekday_weekend(adm, gdz):
    print("\n" + "=" * 60)
    print("SECTION 4: HAFTAICI / HAFTASONU KLIMA FARKI")
    print("=" * 60)
    results = {}

    for label, df in [("ADM", adm), ("GDZ", gdz)]:
        daily_mean_temp = df["temp_mean"].resample("D").mean()
        hot_days = daily_mean_temp[daily_mean_temp > 24].index.date
        if len(hot_days) < 5:
            print(f"  {label}: Yetersiz sıcak gün ({len(hot_days)}), atlanıyor.")
            continue

        df["date"] = df.index.date
        df["is_weekend"] = df.index.dayofweek >= 5  # DatetimeIndex: 5=Cmt, 6=Paz

        hot = df[df["date"].isin(hot_days)]
        wd = hot[~hot["is_weekend"]].groupby("Saat")["load_mwh"].mean()
        we = hot[hot["is_weekend"]].groupby("Saat")["load_mwh"].mean()
        we_diff = we - wd

        if len(wd) == 0 or len(we) == 0:
            print(f"  {label}: Haftaici veya haftasonu verisi yok.")
            continue

        results[label] = {"wd": wd, "we": we, "we_diff": we_diff}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(wd.index, wd.values, "o-", color="#2c7bb6", lw=2, label="Hafta içi")
        ax.plot(we.index, we.values, "o-", color="#d7191c", lw=2, label="Hafta sonu")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Ortalama Tüketim (MWh)")
        ax.set_title(f"{label} — Sıcak Günlerde Hafta içi vs Hafta sonu")
        ax.legend(fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.bar(we_diff.index, we_diff.values, color="#fdae61" if we_diff.mean() < 0 else "#abdda4", width=0.8, alpha=0.85)
        ax.set_xlabel("Saat")
        ax.set_ylabel("Fark (MWh) = Hafta sonu - Hafta içi")
        ax.set_title(f"{label} — Hafta Sonu Fark Profili")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        savefig(f"sec4_weekday_weekend_{label}.png")
        print(f"    -> output/sec4_weekday_weekend_{label}.png")

        # Saturday vs Sunday breakdown
        hot = hot.copy()
        hot["dayname"] = hot.index.dayofweek.map({5: "Cumartesi", 6: "Pazar"}).fillna("Haftaici")
        results[label + "_wd_mean"] = wd.mean()
        results[label + "_we_mean"] = we.mean()
        print(f"  {label}: Hafta içi ort={wd.mean():.0f} MWh, Hafta sonu ort={we.mean():.0f} MWh, Fark={we_diff.mean():.1f} MWh")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5: CDD Analysis
# ═══════════════════════════════════════════════════════════════════════════
def section_5_cdd(adm, gdz):
    print("\n" + "=" * 60)
    print("SECTION 5: CDD (Cooling Degree Days) ANALIZI")
    print("=" * 60)
    results = {}

    for label, df in [("ADM", adm), ("GDZ", gdz)]:
        daily = daily_agg(df)
        x = daily["temp_mean"].values
        y = daily["load_mwh"].values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        if len(x) == 0:
            continue

        corrs = {}
        for threshold in [18, 20, 22, 24]:
            cdd_vals = cdd(x, threshold)
            cdd_sum = cdd_vals * 24  # approximate daily CDD
            c = np.corrcoef(cdd_vals, y)[0, 1]
            corrs[threshold] = c
        best_threshold = max(corrs, key=corrs.get)
        results[label] = {"corrs": corrs, "best_threshold": best_threshold}
        print(f"  {label} CDD korelasyonları: ", {k: f"{v:.4f}" for k, v in corrs.items()})
        print(f"    En iyi eşik: {best_threshold}°C (r={corrs[best_threshold]:.4f})")

        # Scatter: best CDD vs consumption
        best_cdd = cdd(x, best_threshold)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(best_cdd, y, s=5, alpha=0.3, c="#2c7bb6")
        slope, intercept = np.polyfit(best_cdd, y, 1)
        fit_line = slope * best_cdd + intercept
        order = np.argsort(best_cdd)
        ax.plot(best_cdd[order], fit_line[order], color="#d7191c", lw=2,
                label=f"r={corrs[best_threshold]:.4f}, slope={slope:.2f}")
        ax.set_xlabel(f"CDD (eşik={best_threshold}°C)")
        ax.set_ylabel("Günlük Ortalama Tüketim (MWh)")
        ax.set_title(f"{label} — CDD vs Tüketim (en iyi eşik)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        savefig(f"sec5_cdd_{label}.png")
        print(f"    -> output/sec5_cdd_{label}.png")

        # Monthly CDD trend
        daily_cdd_all = x.copy()
        for t in [18, 20, 22, 24]:
            daily_cdd_all = np.column_stack((daily_cdd_all, np.maximum(0, x - t))) if daily_cdd_all.ndim > 1 else np.column_stack((x, np.maximum(0, x - t)))
        daily["cdd_20"] = cdd(x, 20)
        daily.index = pd.to_datetime(daily.index)
        monthly_cdd = daily["cdd_20"].resample("ME").sum()
        monthly_load = daily["load_mwh"].resample("ME").mean()

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(monthly_cdd.index, monthly_cdd.values, color="#fdae61", alpha=0.7, width=20, label="Aylık CDD (20°C)")
        ax1.set_ylabel("CDD (derece-gün)")
        ax2 = ax1.twinx()
        ax2.plot(monthly_load.index, monthly_load.values, "o-", color="#d7191c", lw=2, label="Aylık Ort. Tüketim")
        ax2.set_ylabel("Tüketim (MWh)")
        ax1.set_title(f"{label} — Aylık CDD ve Tüketim Trendi")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
        ax1.grid(True, alpha=0.3)
        savefig(f"sec5_monthly_cdd_{label}.png")
        print(f"    -> output/sec5_monthly_cdd_{label}.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6: ADM vs GDZ Comparison
# ═══════════════════════════════════════════════════════════════════════════
def section_6_adm_vs_gdz(adm, gdz):
    print("\n" + "=" * 60)
    print("SECTION 6: ADM vs GDZ KARSILASTIRMASI")
    print("=" * 60)
    results = {}

    daily_a = daily_agg(adm)
    daily_g = daily_agg(gdz)

    daily_a = daily_a.rename(columns={"load_mwh": "ADM_load", "load_sum": "ADM_sum", "temp_mean": "ADM_temp"})
    daily_g = daily_g.rename(columns={"load_mwh": "GDZ_load", "load_sum": "GDZ_sum", "temp_mean": "GDZ_temp"})

    daily_a.index = pd.to_datetime(daily_a.index)
    daily_g.index = pd.to_datetime(daily_g.index)
    daily_a.index = daily_a.index.tz_localize(None) if hasattr(daily_a.index, 'tz') else daily_a.index
    daily_g.index = daily_g.index.tz_localize(None) if hasattr(daily_g.index, 'tz') else daily_g.index

    combined = daily_a[["ADM_load", "ADM_temp"]].join(daily_g[["GDZ_load", "GDZ_temp"]], how="inner").dropna()

    if len(combined) < 30:
        print(f"  Ortak gün sayısı yetersiz: {len(combined)}")
        return results

    results["n_days"] = len(combined)
    results["adm_mean_load"] = combined["ADM_load"].mean()
    results["gdz_mean_load"] = combined["GDZ_load"].mean()
    print(f"  Ortak gün sayısı: {len(combined)}")
    print(f"  ADM ortalama günlük tüketim: {combined['ADM_load'].mean():.0f} MWh")
    print(f"  GDZ ortalama günlük tüketim: {combined['GDZ_load'].mean():.0f} MWh")

    # Cooling sensitivity comparison
    # ADM
    bp_a, ls_a, rs_a = piecewise_breakpoint(combined["ADM_temp"].values, combined["ADM_load"].values)
    bp_g, ls_g, rs_g = piecewise_breakpoint(combined["GDZ_temp"].values, combined["GDZ_load"].values)
    results["ADM_breakpoint"] = bp_a
    results["GDZ_breakpoint"] = bp_g
    print(f"  ADM kırılım noktası: {bp_a:.1f}°C, soğutma hassasiyeti: {rs_a:.3f} MWh/°C" if bp_a else "  ADM kırılım bulunamadı")
    print(f"  GDZ kırılım noktası: {bp_g:.1f}°C, soğutma hassasiyeti: {rs_g:.3f} MWh/°C" if bp_g else "  GDZ kırılım bulunamadı")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(combined["ADM_temp"], combined["ADM_load"], s=5, alpha=0.3, c="#2c7bb6", label="ADM")
    ax.scatter(combined["GDZ_temp"], combined["GDZ_load"], s=5, alpha=0.3, c="#d7191c", label="GDZ")
    order_a = np.argsort(combined["ADM_temp"])
    order_g = np.argsort(combined["GDZ_temp"])
    smooth_a = lowess_smooth(combined["ADM_temp"].values, combined["ADM_load"].values, frac=0.15)
    smooth_g = lowess_smooth(combined["GDZ_temp"].values, combined["GDZ_load"].values, frac=0.15)
    ax.plot(combined["ADM_temp"].values[order_a], smooth_a[order_a], color="#2c7bb6", lw=2.5, label="ADM LOWESS")
    ax.plot(combined["GDZ_temp"].values[order_g], smooth_g[order_g], color="#d7191c", lw=2.5, label="GDZ LOWESS")
    ax.set_xlabel("Ortalama Sıcaklık (°C)")
    ax.set_ylabel("Günlük Tüketim (MWh)")
    ax.set_title("ADM vs GDZ — Sıcaklık-Tüketim LOWESS Karşılaştırması")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    adm_hot = combined[combined["ADM_temp"] > 24]
    gdz_hot = combined[combined["GDZ_temp"] > 24]
    adm_dates_set = set(d.strftime("%Y-%m-%d") for d in adm_hot.index.date)
    gdz_dates_set = set(d.strftime("%Y-%m-%d") for d in gdz_hot.index.date)
    adm_hourly = adm[adm.index.to_series().dt.strftime("%Y-%m-%d").isin(adm_dates_set)]
    gdz_hourly = gdz[gdz.index.to_series().dt.strftime("%Y-%m-%d").isin(gdz_dates_set)]
    if len(adm_hourly) > 0 and len(gdz_hourly) > 0:
        adm_hp = adm_hourly.groupby("Saat")["load_mwh"].mean()
        gdz_hp = gdz_hourly.groupby("Saat")["load_mwh"].mean()
        # Normalize by mean for shape comparison
        adm_hp_norm = adm_hp / adm_hp.mean()
        gdz_hp_norm = gdz_hp / gdz_hp.mean()
        ax.plot(adm_hp_norm.index, adm_hp_norm.values, "o-", color="#2c7bb6", lw=2, label="ADM (norm.)")
        ax.plot(gdz_hp_norm.index, gdz_hp_norm.values, "o-", color="#d7191c", lw=2, label="GDZ (norm.)")
        ax.set_xlabel("Saat")
        ax.set_ylabel("Normalize Tüketim (ort=1)")
        ax.set_title("Sıcak Günlerde Saatlik Profil Karşılaştırması (norm.)")
        ax.legend(fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
    savefig("sec6_adm_vs_gdz.png")
    print("    -> output/sec6_adm_vs_gdz.png")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7: STLF Feature Recommendations
# ═══════════════════════════════════════════════════════════════════════════
def section_7_feature_recommendations(sec1_results, sec2_results, sec3_results, sec4_results, sec5_results):
    print("\n" + "=" * 60)
    print("SECTION 7: STLF MODELI ICIN FEATURE ONERILERI")
    print("=" * 60)

    recommendations = []

    # 1. CDD threshold optimization
    if sec5_results and "ADM" in sec5_results:
        best = sec5_results["ADM"].get("best_threshold", 22)
        recommendations.append(f"CDD_Cooling_Stress eşik değeri {best}°C olarak optimize edilmeli "
                               f"(mevcut değer kontrol edilmeli)")

    # 2. GHI*Temp interaction
    if sec3_results and sec3_results.get("ADM_GHI_partial_corr", 0) > 0.05:
        recs = ("GHI * Sıcaklık etkileşim feature'ı (GHI_ADM_Weighted * temp_mean): "
                "güneşli ve sıcak günlerde ek klima yükü var")
        recommendations.append(recs)

    # 3. Rolling 3-day temp average
    recommendations.append("Rolling 3-gün sıcaklık ortalaması (ısı birikimi etkisi): "
                           "sıcak dalgası feature'ı olarak modele eklenebilir")

    # 4. Temp ramp rate
    recommendations.append("Sıcaklık rampa hızı (dT/dt): günden güne sıcaklık değişim hızı")

    # 5. Weekday-weekend cooling shape
    if sec2_results and "ADM" in sec2_results:
        peak_hour = sec2_results["ADM"]["cooling_load"].idxmax() if hasattr(sec2_results["ADM"]["cooling_load"], 'idxmax') else "? "
        recommendations.append(f"Saatlik klima profili (tepe saat {peak_hour}:00): "
                               f"tepe saatlerde ek regresör olarak kullanılabilir")

    # 6. Weekend AC scaling
    if sec4_results:
        recommendations.append("Hafta sonu klima kullanım farkı: hafta sonu günlerinde "
                               "cooling feature scale faktörü uygulanabilir")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    return recommendations


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("#" * 60)
    print("# KLIMA ETKISI STLF ANALIZI — ADM & GDZ KARSILASTIRMALI")
    print("#" * 60)
    print()

    # Load
    print("Veri yükleniyor...")
    adm = load_adm()
    gdz = load_gdz()
    print(f"  ADM: {len(adm)} saatlik kayıt ({adm.index[0]} -> {adm.index[-1]})")
    print(f"  GDZ: {len(gdz)} saatlik kayıt ({gdz.index[0]} -> {gdz.index[-1]})")
    print()

    # Run all sections
    sec1 = section_1_temp_consumption(adm, gdz)
    sec2 = section_2_cooling_profile(adm, gdz)
    sec3 = section_3_regional_effect(adm, gdz)
    sec4 = section_4_weekday_weekend(adm, gdz)
    sec5 = section_5_cdd(adm, gdz)
    sec6 = section_6_adm_vs_gdz(adm, gdz)

    print("\n" + "#" * 60)
    sec7 = section_7_feature_recommendations(sec1, sec2, sec3, sec4, sec5)

    # ── Summary ──
    print()
    print("#" * 60)
    print("# OZET RAPOR")
    print("#" * 60)

    # Print summary of key findings
    if sec1:
        for label in ["ADM", "GDZ"]:
            if label in sec1 and sec1[label].get("breakpoint"):
                bp = sec1[label]["breakpoint"]
                sens = sec1[label].get("sensitivity_MWh_per_C", 0)
                print(f"  {label}: Klima devreye giriş ~{bp:.1f}°C, her 1°C artışta ~{sens:.1f} MWh/gün ek yük")

    if sec2 and "ADM" in sec2:
        cl = sec2["ADM"]["cooling_load"]
        print(f"  ADM Klima Profili: tepe={cl.idxmax()}:00 ({cl.max():.0f} MWh), "
              f"toplam günlük klima yükü={cl.sum():.0f} MWh")

    if sec5 and "ADM" in sec5:
        best_t = sec5["ADM"].get("best_threshold", "?")
        print(f"  En iyi CDD eşiği: {best_t}°C")

    if sec6:
        if "adm_mean_load" in sec6:
            ratio = sec6["gdz_mean_load"] / sec6["adm_mean_load"] if sec6["adm_mean_load"] else 0
            print(f"  GDZ/ADM ortalama tüketim oranı: {ratio:.2f}x")

    print()
    print(f"Tüm grafikler output/ klasörüne kaydedildi.")
    print(f"Klima profili vektörleri: output/klima_profili_ADM.csv, output/klima_profili_GDZ.csv")


if __name__ == "__main__":
    main()
