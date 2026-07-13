"""
diagnostic_core.py — ADM + GDZ ortak interaktif diagnostic motoru
==================================================================
TEK KAYNAK: hem veri hesabı hem HTML/JS render burada. ADM ve GDZ
scriptleri sadece kendi kolonlarını "kanonik" isimlere map edip
compute() + render() çağırır. Böylece iki EDAŞ hiçbir zaman ayrışmaz
(eski "ADM template'ini string-splice et" yaklaşımının GDZ'yi bozması
bu modülle kökten çözülür).

Kanonik `merged` kolonları (wrapper doldurur):
    dt      : normalize edilmiş gün (datetime)
    h       : saat 0-23 (int)
    load    : hedef tüketim (MWh)
    temp    : hissedilen sıcaklık (°C)
    ghi     : global ışınım (W/m²)   [opsiyonel]
    cloud   : bulutluluk (%)          [opsiyonel]
    humidity: nem (%)                 [opsiyonel]
    wind    : rüzgar (m/s veya km/s)  [opsiyonel]
    precip  : yağış                  [opsiyonel]
    special : özel gün adı (str) ya da "Değil"/None  [opsiyonel]

fc      : D+2 tahmini, 24 değerli liste
fc_wx   : temp/ghi/cloud/humidity/wind/precip 24 saatlik tahmin havası
"""
import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Sabitler ──────────────────────────────────────────────────────────
SEASONS = [("Kis", [12, 1, 2]), ("Ilkbahar", [3, 4, 5]),
           ("Yaz", [6, 7, 8]), ("Sonbahar", [9, 10, 11])]
TEMP_BINS = [(-10, 5), (5, 10), (10, 15), (15, 20),
             (20, 25), (25, 30), (30, 35), (35, 50)]
HOUR_GROUPS = [("Gece (00-06)", range(0, 7)), ("Sabah (07-10)", range(7, 11)),
               ("Ogle (11-16)", range(11, 17)), ("Aksam (17-23)", range(17, 24))]
CMP_PAIRS = [(7, "1 hafta once"), (14, "2 hafta once"),
             (364, "1 yil once"), (371, "1 yil + 1 hafta once")]


def _f(x):
    """numpy/nan güvenli float→JSON."""
    try:
        v = float(x)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 3)
    except (TypeError, ValueError):
        return None


def _slope(x, y):
    """Basit lineer eğim (MW/°C). Yetersiz/degenerate veride None."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 15 or np.std(x) < 0.8:
        return None
    b = np.polyfit(x, y, 1)[0]
    return _f(b)


FEATURE_LABELS = {
    "temp": "çok-istasyon sıcaklık",
    "ghi": "güneş ışınımı (GHI)",
    "cloud": "bulutluluk",
    "humidity": "nem",
    "wind": "rüzgâr",
    "precip": "yağış",
    "lag1": "D-1 aynı saat yükü",
    "lag2": "D-2 aynı saat yükü",
    "lag7": "D-7 aynı saat yükü",
    "lag14": "D-14 aynı saat yükü",
    "asof_d": "son erişilebilir D yükü",
    "asof_d1": "son erişilebilir D-1 yükü",
    "asof_d7": "son erişilebilir D-7 yükü",
    "roll7": "son 7 gün yük seviyesi",
    "roll14": "son 14 gün yük seviyesi",
    "weekly_profile": "son haftaların aynı saat profili",
    "profile_prev": "son erişilebilir gün komşu saat",
    "profile_next": "son erişilebilir gün komşu saat",
    "dow_sin": "haftanın günü",
    "dow_cos": "haftanın günü",
    "is_saturday": "cumartesi etkisi",
    "is_sunday": "pazar etkisi",
    "is_special": "tatil/özel gün",
    "trend": "uzun dönem yük trendi",
}


def _lookup_load(grid, dates, hours):
    parsed = pd.to_datetime(dates)
    normalized = parsed.dt.normalize() if isinstance(parsed, pd.Series) else parsed.normalize()
    keys = pd.MultiIndex.from_arrays([normalized, hours])
    return grid.reindex(keys).to_numpy(dtype=float)


def _add_consolidated_features(merged):
    """Yalnizca gecmis bilgiyi kullanan takvim, lag ve profil feature'lari."""
    out = merged.sort_values(['dt', 'h']).drop_duplicates(['dt', 'h'], keep='last').copy()
    out['dt'] = pd.to_datetime(out['dt']).dt.normalize()
    out['h'] = out['h'].astype(int)
    grid = out.set_index(['dt', 'h'])['load'].astype(float).sort_index()
    for lag in range(1, 16):
        out[f"_lag{lag}"] = _lookup_load(grid, out['dt'] - pd.Timedelta(days=lag), out['h'])
    for lag in (21, 28):
        out[f"_lag{lag}"] = _lookup_load(grid, out['dt'] - pd.Timedelta(days=lag), out['h'])
    out['lag1'] = out['_lag1']; out['lag2'] = out['_lag2']
    out['lag7'] = out['_lag7']; out['lag14'] = out['_lag14']
    # Canli veri bir gun gecikmeli gelir: hedef-2 = issue gununde bilinen son D.
    out['asof_d'] = out['_lag2']; out['asof_d1'] = out['_lag3']; out['asof_d7'] = out['_lag9']
    out['roll7'] = out[[f"_lag{i}" for i in range(2, 9)]].mean(axis=1, skipna=True)
    out['roll14'] = out[[f"_lag{i}" for i in range(2, 16)]].mean(axis=1, skipna=True)
    out['weekly_profile'] = out[["_lag7", "_lag14", "_lag21", "_lag28"]].mean(axis=1, skipna=True)
    prev_hours = (out['h'] - 1).where(out['h'] > 0, np.nan)
    next_hours = (out['h'] + 1).where(out['h'] < 23, np.nan)
    out['profile_prev'] = _lookup_load(grid, out['dt'] - pd.Timedelta(days=2), prev_hours)
    out['profile_next'] = _lookup_load(grid, out['dt'] - pd.Timedelta(days=2), next_hours)
    out['dow_sin'] = np.sin(2 * np.pi * out['dow'] / 7.0)
    out['dow_cos'] = np.cos(2 * np.pi * out['dow'] / 7.0)
    out['is_saturday'] = (out['dow'] == 5).astype(float)
    out['is_sunday'] = (out['dow'] == 6).astype(float)
    special = out.get('special', pd.Series(index=out.index, dtype=object)).astype(str)
    out['is_special'] = (~special.isin(['None', 'nan', '', 'Değil', 'Degil'])).astype(float)
    return out, grid


def _target_features(history, grid, target, hour, fc_wx, model_signal):
    def lag(day, target_hour=hour):
        if target_hour < 0 or target_hour > 23:
            return np.nan
        return grid.get((pd.Timestamp(target).normalize() - pd.Timedelta(days=day), int(target_hour)), np.nan)

    values = {}
    for key in ('temp', 'ghi', 'cloud', 'humidity', 'wind', 'precip'):
        seq = (fc_wx or {}).get(key) or []
        values[key] = seq[hour] if len(seq) > hour else np.nan
    def available_mean(items):
        valid = [float(value) for value in items if _f(value) is not None]
        return float(np.mean(valid)) if valid else np.nan

    values.update({
        'lag1': lag(1), 'lag2': lag(2), 'lag7': lag(7), 'lag14': lag(14),
        'asof_d': lag(2), 'asof_d1': lag(3), 'asof_d7': lag(9),
        'roll7': available_mean([lag(i) for i in range(2, 9)]),
        'roll14': available_mean([lag(i) for i in range(2, 16)]),
        'weekly_profile': available_mean([lag(i) for i in (7, 14, 21, 28)]),
        'profile_prev': lag(2, hour - 1), 'profile_next': lag(2, hour + 1),
    })
    dow = target.weekday()
    values.update({
        'dow_sin': np.sin(2 * np.pi * dow / 7.0),
        'dow_cos': np.cos(2 * np.pi * dow / 7.0),
        'is_saturday': float(dow == 5), 'is_sunday': float(dow == 6),
        'is_special': float(any(bool((model_signal or {}).get(k)) for k in
                                ('flag_holiday', 'flag_bridge', 'flag_ramadan'))),
        'trend': 0.0,
    })
    return values


def _consolidated_hour(sh, query, target, fc_value, model_signal, performance):
    """Ridge + analog + model uzlasisi + bias duzeltmeli konsolide beklenti."""
    candidates = [
        'temp', 'ghi', 'cloud', 'humidity', 'wind', 'precip',
        'lag1', 'lag7', 'lag14', 'asof_d', 'asof_d1', 'asof_d7',
        'roll7', 'roll14', 'weekly_profile',
        'profile_prev', 'profile_next', 'dow_sin', 'dow_cos',
        'is_saturday', 'is_sunday', 'is_special', 'trend',
    ]
    train = sh.copy()
    train['trend'] = (train['dt'] - pd.Timestamp(target)).dt.days.astype(float) / 365.0
    features = []
    for name in candidates:
        qv = _f(query.get(name))
        if qv is None or name not in train:
            continue
        vals = pd.to_numeric(train[name], errors='coerce')
        if vals.notna().sum() >= 25 and float(vals.std(skipna=True) or 0.0) > 1e-8:
            features.append(name)
    if len(train) < 25 or len(features) < 2:
        return None

    X = train[features].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    Y = pd.to_numeric(train['load'], errors='coerce').to_numpy(float)
    valid_y = np.isfinite(Y)
    X, Y, train = X[valid_y], Y[valid_y], train.loc[valid_y]
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    q = np.array([float(query[name]) for name in features], dtype=float)
    q = np.where(np.isnan(q), med, q)
    mean = np.mean(X, axis=0); std = np.std(X, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    Z = (X - mean) / std; zq = (q - mean) / std

    age_days = (pd.Timestamp(target) - pd.to_datetime(train['dt'])).dt.days.to_numpy(float)
    weights = np.power(0.5, np.maximum(age_days, 0) / 540.0)
    design = np.column_stack([np.ones(len(Z)), Z])
    root_w = np.sqrt(weights)
    wd = design * root_w[:, None]; wy = Y * root_w
    penalty = np.eye(design.shape[1]) * 3.0; penalty[0, 0] = 0.0
    try:
        beta = np.linalg.solve(wd.T @ wd + penalty, wd.T @ wy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(wd.T @ wd + penalty) @ (wd.T @ wy)
    fitted = design @ beta
    ridge_pred = float(np.dot(np.r_[1.0, zq], beta))
    resid = Y - fitted
    denom = float(np.sum(weights * (Y - np.average(Y, weights=weights)) ** 2))
    r2 = 1.0 - float(np.sum(weights * resid ** 2)) / denom if denom > 1e-9 else 0.0
    r2 = float(np.clip(r2, -1.0, 1.0))

    analog_features = [i for i, name in enumerate(features) if name != 'trend']
    analog_pred = None; analog_distance = None
    if analog_features:
        dist = np.sqrt(np.mean((Z[:, analog_features] - zq[analog_features]) ** 2, axis=1))
        k = min(30, len(dist))
        nearest = np.argsort(dist)[:k]
        dw = np.exp(-np.clip(dist[nearest], 0, 20))
        analog_pred = float(np.average(Y[nearest], weights=dw)) if dw.sum() > 0 else float(np.mean(Y[nearest]))
        analog_distance = float(np.mean(dist[nearest]))

    model_values = [float(model_signal.get(k)) for k in ('xgb', 'lgbm', 'cat', 'chronos')
                    if _f((model_signal or {}).get(k)) is not None]
    model_center = float(np.median(model_values)) if len(model_values) >= 2 else None
    model_spread = float(np.std(model_values)) if len(model_values) >= 2 else None
    perf = performance or {}
    bias30 = _f(perf.get('bias30')); mape30 = _f(perf.get('mape30')); n30 = int(perf.get('n30') or 0)
    # Kullanici Excel'i sonradan degistirse bile ikinci gorus kendi kendini takip
    # etmesin: bias uzmani pipeline'in ozgun final sinyalinden baslar.
    bias_base = _f((model_signal or {}).get('final'))
    if bias_base is None:
        bias_base = _f(fc_value)
    bias_corrected = float(bias_base - bias30) if bias_base is not None and bias30 is not None and n30 >= 3 else None

    analog_score = 1.0 / (1.0 + max(analog_distance or 2.0, 0.0))
    consensus_score = 0.5
    if model_center and model_spread is not None:
        consensus_score = float(np.clip(1.0 - (model_spread / abs(model_center)) / 0.10, 0.0, 1.0))
    perf_reliability = min(n30 / 20.0, 1.0) * (1.0 - min((mape30 or 15.0) / 15.0, 1.0))
    experts = [("ridge", ridge_pred, 0.25 + 0.25 * max(r2, 0.0))]
    if analog_pred is not None:
        experts.append(("analog", analog_pred, 0.08 + 0.22 * analog_score))
    if bias_corrected is not None:
        experts.append(("bias", bias_corrected, 0.05 + 0.15 * perf_reliability))
    if model_center is not None:
        experts.append(("models", model_center, 0.05 + 0.10 * consensus_score))
    expert_weights = np.array([item[2] for item in experts], float)
    expert_values = np.array([item[1] for item in experts], float)
    expected = float(np.average(expert_values, weights=expert_weights))

    residual_half = float(np.percentile(np.abs(resid), 95)) if len(resid) else 0.0
    expert_half = float(np.std(expert_values) * 1.96) if len(expert_values) > 1 else 0.0
    model_half = float((model_spread or 0.0) * 1.96)
    half_width = max(residual_half, expert_half, model_half, abs(expected) * 0.01)
    lo, hi = expected - half_width, expected + half_width
    band_pct = (2.0 * half_width / abs(expected) * 100.0) if abs(expected) > 1e-9 else 100.0

    sample_score = min(len(train) / 120.0, 1.0)
    fit_score = float(np.clip((r2 + 0.10) / 0.80, 0.0, 1.0))
    band_score = float(np.clip(1.0 - band_pct / 25.0, 0.0, 1.0))
    performance_score = perf_reliability if n30 else 0.35
    confidence_score = 100.0 * (
        0.20 * sample_score + 0.22 * fit_score + 0.18 * analog_score +
        0.15 * consensus_score + 0.15 * performance_score + 0.10 * band_score
    )
    confidence = "Yüksek" if confidence_score >= 72 else "Orta" if confidence_score >= 50 else "Düşük"
    importance = sorted(zip(features, np.abs(beta[1:])), key=lambda x: x[1], reverse=True)
    drivers = []
    for name, _ in importance:
        label = FEATURE_LABELS.get(name, name)
        if label not in drivers:
            drivers.append(label)
        if len(drivers) == 4:
            break
    return {
        "exp": _f(expected), "lo": _f(lo), "hi": _f(hi),
        "ridge_exp": _f(ridge_pred), "analog_exp": _f(analog_pred),
        "analog_distance": _f(analog_distance), "r2": _f(r2),
        "model_center": _f(model_center), "model_spread": _f(model_spread),
        "bias_corrected": _f(bias_corrected), "bias7": _f(perf.get('bias7')),
        "mape7": _f(perf.get('mape7')), "bias30": bias30, "mape30": mape30,
        "perf_n7": int(perf.get('n7') or 0), "perf_n30": n30,
        "band_pct": _f(band_pct), "confidence": confidence,
        "confidence_score": _f(confidence_score), "drivers": drivers,
        "features_used": features, "expert_weights": {name: _f(weight / expert_weights.sum())
                                                         for (name, _, weight) in experts},
    }


# ══════════════════════════════════════════════════════════════════════
#  COMPUTE
# ══════════════════════════════════════════════════════════════════════
def compute(merged, fc, fc_wx, fc_date, edas, model_signals=None, hourly_performance=None):
    merged = merged.copy()
    merged['dow'] = merged['dt'].dt.dayofweek
    merged['month'] = merged['dt'].dt.month
    merged['doy'] = merged['dt'].dt.dayofyear
    TODAY = date.fromisoformat(fc_date)
    has_ghi = bool('ghi' in merged and merged['ghi'].notna().sum() > 100)
    has_cloud = bool('cloud' in merged and merged['cloud'].notna().sum() > 100)
    has_wind = bool('wind' in merged and merged['wind'].notna().sum() > 100)
    has_special = 'special' in merged

    # özel gün adı → tarih haritası
    special_map = {}
    if has_special:
        sp = merged.dropna(subset=['special'])
        sp = sp[~sp['special'].astype(str).isin(['Değil', 'Degil', 'nan', ''])]
        for d, name in sp.groupby(sp['dt'].dt.date)['special'].first().items():
            special_map[str(d)] = str(name)

    # Konsolide beklenti icin lag/seviye/profil feature'lari yalnizca gecmise
    # bakacak sekilde bir kez olusturulur.
    merged, load_grid = _add_consolidated_features(merged)

    def gs(ds):
        d = pd.Timestamp(ds).date()
        day = merged[merged['dt'].dt.date == d]
        if len(day) < 20:
            return None
        day = day.set_index('h').sort_index().reindex(range(24))
        r = {"load": [_f(v) for v in day['load'].values]}
        if day['load'].notna().sum() < 18:
            return None
        r["temp"] = [_f(v) for v in day['temp'].values] if 'temp' in day else None
        if has_ghi:   r["ghi"] = [_f(v) for v in day['ghi'].values]
        if has_cloud: r["cloud"] = [_f(v) for v in day['cloud'].values]
        if has_wind:  r["wind"] = [_f(v) for v in day['wind'].values]
        if 'humidity' in day: r["humidity"] = [_f(v) for v in day['humidity'].values]
        if 'precip' in day: r["precip"] = [_f(v) for v in day['precip'].values]
        r["special"] = special_map.get(str(d))
        return r

    # ── CP: karşılaştırma günleri ─────────────────────────────────────
    cp = {}
    if fc:
        for off, lbl in CMP_PAIRS:
            s = gs(str(TODAY - timedelta(days=off)))
            if s:
                cp[lbl] = s

    # ── P95: saatlik hafta-üstü hata bandı ────────────────────────────
    p95 = {}
    for h in range(24):
        seg = merged[merged['h'] == h].dropna(subset=['load'])
        if len(seg) < 40:
            continue
        # seg yalnizca tek bir saati icerir; 7 satir = bir hafta onceki ayni saat.
        err = seg['load'].values - seg['load'].shift(7).values
        err = err[~np.isnan(err)]
        if len(err) > 50:
            ml = float(np.mean(seg['load'].values))
            p95[h] = {"p5_ape": _f(np.percentile(np.abs(err), 5) / ml * 100),
                      "p95_ape": _f(np.percentile(np.abs(err), 95) / ml * 100),
                      "p50_ape": _f(np.median(np.abs(err)) / ml * 100),
                      "p5_err": _f(np.percentile(err, 5)),
                      "p95_err": _f(np.percentile(err, 95))}

    # ── SN: sezon / saat-grubu / gün-tipi global eğim (MW/°C) ─────────
    sn = {}
    seg = merged.dropna(subset=['temp', 'load'])
    if len(seg) > 100:
        for nm, ms in SEASONS:
            s = _slope(seg[seg['month'].isin(ms)]['temp'], seg[seg['month'].isin(ms)]['load'])
            if s is not None: sn[nm] = s
        for hg, hr in HOUR_GROUPS:
            s = _slope(seg[seg['h'].isin(hr)]['temp'], seg[seg['h'].isin(hr)]['load'])
            if s is not None: sn[hg] = s
        for lbl, msk in [("Haftaici", seg['dow'] < 5), ("Cumartesi", seg['dow'] == 5),
                         ("Pazar", seg['dow'] == 6)]:
            s = _slope(seg[msk]['temp'], seg[msk]['load'])
            if s is not None: sn[lbl] = s

    # ── SNB: SICAKLIK ARALIĞI (bin) bazında yerel duyarlılık ──────────
    # "20-25°C aralığında her +1°C → +X MW" — sezon kırılımlı.
    snb = {"bins": [], "overall": [], "season": {}}
    if len(seg) > 200:
        for lo, hi in TEMP_BINS:
            snb["bins"].append(f"{lo}-{hi}")
            bs = seg[(seg['temp'] >= lo) & (seg['temp'] < hi)]
            snb["overall"].append({
                "slope": _slope(bs['temp'], bs['load']),
                "n": int(len(bs)),
                "mean_load": _f(bs['load'].mean()) if len(bs) else None,
                "mean_temp": _f(bs['temp'].mean()) if len(bs) else None,
            })
        for nm, ms in SEASONS:
            ss = seg[seg['month'].isin(ms)]
            row = []
            for lo, hi in TEMP_BINS:
                bs = ss[(ss['temp'] >= lo) & (ss['temp'] < hi)]
                row.append({"slope": _slope(bs['temp'], bs['load']), "n": int(len(bs))})
            snb["season"][nm] = row

    # ── HTE: saatlik sıcaklık etkisi (son 7 gün) ──────────────────────
    hte = {}
    for h in range(24):
        sh = merged[merged['h'] == h].dropna(subset=['temp', 'load']).tail(21)
        if len(sh) >= 10 and np.std(sh['temp'].values) > 0.5:
            sl = np.polyfit(sh['temp'].values, sh['load'].values, 1)[0]
            r = np.corrcoef(sh['temp'].values, sh['load'].values)[0, 1]
            hte[h] = {"slope": _f(sl), "r2": _f(r * r), "n": int(len(sh))}

    # ── REC: KONSOLIDE saatlik ikinci-gorus motoru + belirsizlik ──────────
    # Ridge (takvim+lag+coklu hava), benzer gun, model uzlasisi ve son
    # performans bias'i birlestirilir. Final forecast yalnizca uzmanlardan biri;
    # istatistiksel omurga ayrica korunur.
    rec = []
    ref = cp.get("1 hafta once")
    doy0 = TODAY.timetuple().tm_yday
    tgt_weekend = TODAY.weekday() >= 5
    # aynı mevsim (±40 gün) + aynı gün-tipi (haftaici/haftasonu) + son 3 yıl:
    # yıl-üstü yük büyümesi ve haftaici/haftasonu karışımı P95'i şişirmesin.
    dd = np.minimum(np.abs(merged['doy'] - doy0), 365 - np.abs(merged['doy'] - doy0))
    recent = merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=1100))
    daytype = (merged['dow'] >= 5) if tgt_weekend else (merged['dow'] < 5)
    base_mask = (dd <= 40) & recent & daytype
    for h in range(24):
        temp_fc = fc_wx.get("temp", [None] * 24)[h] if fc_wx else None
        signal = (model_signals or {}).get(h, {})
        perf = (hourly_performance or {}).get(h, {})
        sh = merged[base_mask & (merged['h'] == h) &
                    (merged['dt'] < pd.Timestamp(TODAY))].dropna(subset=['load'])
        if len(sh) < 25:  # gün-tipi filtresi çok daralttıysa mevsim+son 3 yıla düş
            sh = merged[(dd <= 40) & recent & (merged['h'] == h) &
                        (merged['dt'] < pd.Timestamp(TODAY))].dropna(subset=['load'])
        entry = {"h": h, "fc": _f(fc[h]) if fc else None, "temp_fc": _f(temp_fc),
                 "n": int(len(sh)), "exp": None, "lo": None, "hi": None,
                 "lastweek": _f(ref["load"][h]) if ref and ref.get("load") else None,
                 "models": {k: _f(signal.get(k)) for k in
                            ('xgb', 'lgbm', 'cat', 'chronos', 'ensemble')},
                 "weather": {
                     k: _f(((fc_wx or {}).get(k) or [None] * 24)[h])
                     for k in ('temp', 'ghi', 'cloud', 'humidity', 'wind', 'precip')
                 }}
        if len(sh) >= 25:
            query = _target_features(merged, load_grid, TODAY, h, fc_wx, signal)
            result = _consolidated_hour(
                sh, query, TODAY, fc[h] if fc else None, signal, perf,
            )
            if result:
                entry.update(result)
        rec.append(entry)

    # ── DRIFT: D→D+2 sıcaklık & tüketim kayması ───────────────────────
    drift = {"temp_fc": (fc_wx.get("temp") if fc_wx else None)}
    if ref and ref.get("temp"):
        drift["temp_lastweek"] = ref["temp"]
    if cp.get("2 hafta once", {}).get("temp"):
        drift["temp_2week"] = cp["2 hafta once"]["temp"]
    # son 14 gün günlük ortalama sıcaklık trendi
    tail = merged[merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=16))]
    dmt = tail.dropna(subset=['temp']).groupby(tail['dt'].dt.date)['temp'].mean()
    drift["daily_temp"] = [[str(d), _f(v)] for d, v in dmt.items()]
    # tüketim benzerliği: FC vs her CP günü (MAPE + korelasyon)
    sim = []
    if fc:
        fca = np.array(fc, float)
        for lbl, s in cp.items():
            if not s.get("load"):
                continue
            la = np.array([x if x is not None else np.nan for x in s["load"]], float)
            mk = ~np.isnan(la) & (la > 0)
            if mk.sum() > 12:
                mape = float(np.mean(np.abs((fca[mk] - la[mk]) / la[mk])) * 100)
                corr = float(np.corrcoef(fca[mk], la[mk])[0, 1])
                sim.append({"lbl": lbl, "mape": _f(mape), "corr": _f(corr)})
    drift["sim"] = sim
    # son 7 gün: saatler-arası Δsıcaklık → Δyük (MW/°C)
    d7 = merged[merged['dt'] >= (pd.Timestamp(TODAY) - pd.Timedelta(days=8))].sort_values(['dt', 'h'])
    dl = d7['load'].diff().values; dtp = d7['temp'].diff().values
    mk = ~(np.isnan(dl) | np.isnan(dtp))
    if mk.sum() > 20 and np.std(dtp[mk]) > 0.3:
        drift["hourly_dtemp_slope"] = _f(np.polyfit(dtp[mk], dl[mk], 1)[0])

    # ── SPECIAL: özel gün etkileri ────────────────────────────────────
    special = {}
    fri = merged[(merged['dow'] == 4) & (merged['h'].isin([12, 13]))]
    base = merged[(merged['dow'].isin([1, 2, 3])) & (merged['h'].isin([12, 13]))]
    if len(fri) > 50 and len(base) > 100:
        fm, wm = fri['load'].mean(), base['load'].mean()
        special["friday"] = {"mw": _f(fm - wm), "pct": _f((fm - wm) / wm * 100)}
        # sıcaklık aralığına göre cuma etkisi
        by_temp = []
        for lo, hi in [(-10, 15), (15, 22), (22, 28), (28, 50)]:
            fb = fri[(fri['temp'] >= lo) & (fri['temp'] < hi)]
            bb = base[(base['temp'] >= lo) & (base['temp'] < hi)]
            if len(fb) > 10 and len(bb) > 20:
                by_temp.append({"range": f"{lo}-{hi}C", "mw": _f(fb['load'].mean() - bb['load'].mean()),
                                "n": int(len(fb))})
        special["friday_by_temp"] = by_temp
    # hafta sonu etkisi (saatlik, haftaiçi ortalamaya göre %)
    we, wd = {}, {}
    for h in range(24):
        a = merged[(merged['dow'] == 6) & (merged['h'] == h)]['load'].mean()
        b = merged[(merged['dow'] < 5) & (merged['h'] == h)]['load'].mean()
        if not np.isnan(a) and not np.isnan(b) and b > 0:
            we[h] = _f((a - b) / b * 100)
    special["sunday_pct"] = we
    # D+2 özel mi?
    special["target_is_special"] = special_map.get(fc_date)

    # ── L7: son 14 günün MAPE'si (fc dosyaları vs actual) — wrapper doldurur ──
    return {
        "D": fc_date, "EDAS": edas, "FC": [_f(v) for v in fc] if fc else None,
        "WX": fc_wx if fc_wx else None, "CP": cp, "SN": sn, "SNB": snb,
        "P95": p95, "P95K": [int(k) for k in p95.keys()], "HTE": hte,
        "REC": rec, "DRIFT": drift, "SPECIAL": special,
        "MODEL_SIGNALS": model_signals or {},
        "HOURLY_PERFORMANCE": hourly_performance or {},
        "HAS": {"ghi": has_ghi, "cloud": has_cloud, "wind": has_wind,
                "humidity": bool('humidity' in merged and merged['humidity'].notna().sum() > 100),
                "precip": bool('precip' in merged and merged['precip'].notna().sum() > 100)},
    }


# ══════════════════════════════════════════════════════════════════════
#  RENDER
# ══════════════════════════════════════════════════════════════════════
def render(data, title, last7=None):
    data = dict(data)
    data["L7"] = last7 or []
    data_json = json.dumps(data, ensure_ascii=False)
    html = _TEMPLATE.replace("__TITLE__", title).replace(
        "__EDAS__", data.get("EDAS", "")).replace(
        "__FCD__", data.get("D", "")).replace(
        "/*__DATA__*/", "const DATA=" + data_json + ";")
    return html


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box}body{margin:0;background:#0b0e14;color:#dfe6f3;font-family:'Segoe UI',sans-serif;font-size:13.5px}
.app{display:flex;min-height:100vh}
nav{width:190px;background:linear-gradient(180deg,#0d111a,#0b0e14);border-right:1px solid #26304a;padding:12px 0;position:sticky;top:0;height:100vh;overflow:auto;flex-shrink:0}
nav h1{font-size:13px;color:#5ad1a0;margin:4px 12px 2px;font-weight:700}
nav .sub{font-size:10px;color:#8b97b3;margin:0 12px 10px}
nav button{display:block;width:100%;text-align:left;background:none;border:0;color:#8b97b3;padding:8px 12px;cursor:pointer;font-size:12px;border-left:3px solid transparent}
nav button:hover{color:#dfe6f3;background:#11151f}
nav button.on{color:#dfe6f3;background:#11151f;border-left-color:#5ad1a0;font-weight:600}
main{flex:1;padding:18px 22px 60px;max-width:1180px}
.tab{display:none;animation:fade .2s}.tab.on{display:block}
@keyframes fade{from{opacity:0}to{opacity:1}}
h2{font-size:18px;margin:0 0 3px;font-weight:700}
.lead{color:#8b97b3;margin:0 0 14px;font-size:12px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:14px}
.card{background:#131823;border:1px solid #26304a;border-radius:9px;padding:10px 12px}
.card .lab{font-size:10px;color:#8b97b3;text-transform:uppercase}
.card .val{font-size:20px;font-weight:700;margin-top:2px}
.chartbox{position:relative;height:260px;margin-bottom:8px}
.chartbox.tall{height:340px}
.panel{background:#131823;border:1px solid #26304a;border-radius:10px;padding:12px 14px;margin-bottom:12px}
.panel h3{margin:0 0 2px;font-size:13px;color:#c7d0e3}
.panel .note{color:#8b97b3;font-size:11px;margin:0 0 8px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
table.dt{width:100%;font-size:11px;color:#c7d0e3;border-collapse:collapse}
table.dt th{color:#8b97b3;font-weight:600;padding:3px 5px;text-align:center;border-bottom:1px solid #26304a}
table.dt td{padding:3px 5px;text-align:center;border-bottom:1px solid #1c2334}
.rm{padding:11px 14px;margin-bottom:9px;background:#131823;border:1px solid #26304a;border-left:4px solid #5ad1a0;border-radius:8px}
.rm.warn{border-left-color:#ffce6a}.rm.danger{border-left-color:#ff6b6b}.rm.ok{border-left-color:#5ad1a0}
.rm h4{margin:0 0 3px;font-size:13px}.rm p{margin:3px 0;font-size:12px;color:#c7d0e3;line-height:1.55}
.rm .meta{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:#8b97b3;margin:4px 0}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600}
input[type=range]{width:100%}
.slider-val{display:inline-block;background:#26304a;padding:2px 9px;border-radius:4px;font-weight:700}
.badge{font-size:10px;padding:1px 6px;border-radius:4px;background:#2a2140;color:#c9a7ff;margin-left:5px}
@media(max-width:820px){.grid2,.grid3{grid-template-columns:1fr}}
</style></head><body><div class="app">
<nav><h1>__EDAS__ STLF</h1><div class="sub">D+2 &middot; __FCD__</div><div id="nv"></div></nav>
<main id="mn"></main></div>
<script>
/*__DATA__*/
const D=DATA.D, FC=DATA.FC, WX=DATA.WX, CP=DATA.CP, SN=DATA.SN, SNB=DATA.SNB,
      L7=DATA.L7, P95=DATA.P95, P95K=DATA.P95K, HTE=DATA.HTE, REC=DATA.REC,
      DRIFT=DATA.DRIFT, SPECIAL=DATA.SPECIAL, HAS=DATA.HAS;
const hrs=Array.from({length:24},(_,i)=>i);
const pal=['#E53935','#1E88E5','#43A047','#FB8C00','#8E24AA','#00ACC1'];
const GRID={color:'#1c2334'}, TICK={color:'#8b97b3'};
const AX=(t)=>({title:{display:!!t,text:t,color:'#8b97b3'},ticks:TICK,grid:GRID});
const LEG={labels:{color:'#8b97b3',boxWidth:10,font:{size:9}}};
const charts={};
function ch(id,cfg){const x=document.getElementById(id);if(!x)return;if(charts[id])charts[id].destroy();
  cfg.options=Object.assign({responsive:true,maintainAspectRatio:false},cfg.options||{});charts[id]=new Chart(x,cfg);}
function fx(v,d){return v==null?'&mdash;':(+v).toFixed(d==null?0:d);}
function g(id){document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.t===id));
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t.id==='t_'+id));
  if(g._i[id])g._i[id]();location.hash=id;}
g._i={};const T={};
const avg=a=>a.reduce((x,y)=>x+(y||0),0)/a.length;

// ═══ TAB 1: OZET ═══
T.ozet=()=>{let h='<h2>Ozet</h2><p class="lead">'+D+' icin D+2 tahmini &mdash; '+DATA.EDAS+'</p><div class="cards">';
 if(FC){h+='<div class="card"><div class="lab">Ortalama</div><div class="val" style="color:#5ad1a0">'+fx(avg(FC))+'</div><div class="lab">MWh</div></div>';
 const mx=Math.max(...FC);h+='<div class="card"><div class="lab">Pik</div><div class="val" style="color:#ffce6a">'+fx(mx)+'</div><div class="lab">s.'+FC.indexOf(mx)+':00</div></div>';
 h+='<div class="card"><div class="lab">Min</div><div class="val" style="color:#7fb2ff">'+fx(Math.min(...FC))+'</div><div class="lab">MWh</div></div>';}
 if(P95K.length){const m=avg(P95K.map(k=>P95[k].p95_ape));h+='<div class="card"><div class="lab">P95 bant</div><div class="val" style="color:#ff8a8a;font-size:17px">&plusmn;'+fx(m,1)+'%</div></div>';}
 if(SPECIAL.target_is_special)h+='<div class="card" style="border-color:#8E24AA"><div class="lab">Ozel Gun</div><div class="val" style="color:#c9a7ff;font-size:14px">'+SPECIAL.target_is_special+'</div></div>';
 h+='</div>';
 if(L7.length)h+='<div class="panel"><h3>Son gunler MAPE (%)</h3><p class="note">Teslim edilen tahmin vs gerceklesme</p><div class="chartbox"><canvas id="c7"></canvas></div></div>';
 if(WX&&WX.temp)h+='<div class="panel"><h3>D+2 tahmin havasi</h3><div class="chartbox tall"><canvas id="cw"></canvas></div></div>';
 return h;};
g._i.ozet=()=>{
 if(L7.length)ch('c7',{type:'bar',data:{labels:L7.map(x=>x[0]),datasets:[{label:'MAPE%',data:L7.map(x=>x[1]),backgroundColor:L7.map(x=>x[1]<3?'rgba(90,209,160,.75)':x[1]<6?'rgba(255,206,106,.75)':'rgba(255,107,107,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true,...AX('%')},x:AX()}}});
 if(WX&&WX.temp)ch('cw',{type:'line',data:{labels:hrs,datasets:[{label:'Sicaklik C',data:WX.temp,borderColor:'#ff9f5a',borderWidth:2,pointRadius:2,yAxisID:'y'},WX.ghi?{label:'GHI W/m2',data:WX.ghi,borderColor:'#ffce6a',borderWidth:1.5,pointRadius:1,yAxisID:'y2'}:null].filter(Boolean)},options:{plugins:{legend:LEG},scales:{x:AX(),y:{position:'left',...AX('C')},y2:{position:'right',title:{display:true,text:'GHI',color:'#8b97b3'},ticks:TICK,grid:{drawOnChartArea:false}}}}});};

// ═══ TAB 2: KARSILASTIRMA ═══
T.karsi=()=>{const k=Object.keys(CP);if(!k.length||!FC)return '<h2>Karsilastirma</h2><p class="lead">Veri yok</p>';
 let badges=k.map(l=>CP[l].special?l+' <span class="badge">'+CP[l].special+'</span>':l).join(' &middot; ');
 let h='<h2>D+2 Karsilastirma</h2><p class="lead">Referans gunler: '+badges+'</p>';
 h+='<div class="panel"><h3>Yuk profili + P95 bandi</h3><div class="chartbox tall"><canvas id="c1"></canvas></div></div>';
 h+='<div class="grid2"><div class="panel"><h3>Normalize profil (yuk/ortalama)</h3><p class="note">Sekil karsilastirmasi</p><div class="chartbox"><canvas id="c2"></canvas></div></div>';
 h+='<div class="panel"><h3>Sicaklik karsilastirmasi</h3><div class="chartbox"><canvas id="c3"></canvas></div></div></div>';return h;};
g._i.karsi=()=>{const k=Object.keys(CP);if(!k.length||!FC)return;
 let d1=[{label:'TAHMIN '+D,data:FC,borderColor:'#E53935',borderWidth:3,pointRadius:3}];
 if(P95K.length){d1.push({label:'P95 ust',data:hrs.map(h=>P95[h]?FC[h]+P95[h].p95_err:null),borderColor:'transparent',backgroundColor:'rgba(229,57,53,.10)',pointRadius:0,fill:'+1'},
  {label:'P95 alt',data:hrs.map(h=>P95[h]?FC[h]+P95[h].p5_err:null),borderColor:'transparent',pointRadius:0,fill:false});}
 k.forEach((l,i)=>d1.push({label:l,data:CP[l].load,borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]}));
 ch('c1',{type:'line',data:{labels:hrs,datasets:d1},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('MWh')}}});
 let fm=avg(FC),d2=[{label:'TAHMIN',data:FC.map(v=>v/fm),borderColor:'#E53935',borderWidth:3,pointRadius:2}];
 k.forEach((l,i)=>{let m=avg(CP[l].load);d2.push({label:l,data:CP[l].load.map(v=>v/m),borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]});});
 ch('c2',{type:'line',data:{labels:hrs,datasets:d2},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('yuk/ort')}}});
 let d3=[];if(WX&&WX.temp)d3.push({label:'TAHMIN',data:WX.temp,borderColor:'#E53935',borderWidth:2,pointRadius:2});
 k.forEach((l,i)=>{if(CP[l].temp)d3.push({label:l,data:CP[l].temp,borderColor:pal[(i+1)%pal.length],borderWidth:1.5,pointRadius:0,borderDash:[6,3]});});
 if(d3.length)ch('c3',{type:'line',data:{labels:hrs,datasets:d3},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('C')}}});};

// ═══ TAB 3: SICAKLIK & DUYARLILIK ═══
T.sens=()=>{let h='<h2>Sicaklik &amp; Duyarlilik</h2><p class="lead">Sicaklik aralik/sezon/saat kirilimlarinda +1C -> kac MW yuk</p>';
 if(SNB&&SNB.bins&&SNB.bins.length){h+='<div class="panel"><h3>Sicaklik araligi bazinda duyarlilik (MW/C)</h3><p class="note">Yerel egim; her bar o sicaklik kovasindaki +1C etkisi</p><div class="chartbox"><canvas id="cb1"></canvas></div>';
  h+='<table class="dt"><tr><th>Sezon \\ Aralik</th>'+SNB.bins.map(b=>'<th>'+b+'C</th>').join('')+'</tr>';
  Object.keys(SNB.season).forEach(sz=>{h+='<tr><td style="color:#c7d0e3;font-weight:600">'+sz+'</td>'+SNB.season[sz].map(c=>{let v=c.slope;return '<td style="color:'+(v==null?'#4a5570':v>0?'#ff9f5a':'#5ad1a0')+'">'+(v==null?'&mdash;':(v>0?'+':'')+v)+'</td>';}).join('')+'</tr>';});
  h+='</table></div>';}
 if(SN&&Object.keys(SN).length)h+='<div class="panel"><h3>Sezon / saat-grubu / gun-tipi duyarlilik (MW/C)</h3><div class="chartbox"><canvas id="cs2"></canvas></div></div>';
 if(Object.keys(HTE).length)h+='<div class="panel"><h3>Saatlik sicaklik etkisi (son 7 gun, MW/C)</h3><div class="chartbox"><canvas id="cs1"></canvas></div></div>';
 h+='<div class="panel"><h3>Senaryo motoru</h3><p class="note">Sicaklik sapmasina gore beklenen yuk degisimi (aktif sezon egimi ile)</p>';
 h+='<div style="margin:10px 0"><label style="color:#8b97b3;font-size:12px">Sicaklik sapmasi: <span id="sval" class="slider-val">0 C</span></label>';
 h+='<input type="range" id="srange" min="-6" max="6" value="0" step="1"></div><div id="scenOut" style="font-size:12px;color:#c7d0e3"></div></div>';return h;};
g._i.sens=()=>{
 if(SNB&&SNB.bins&&SNB.bins.length)ch('cb1',{type:'bar',data:{labels:SNB.bins.map(b=>b+'C'),datasets:[{label:'MW/C',data:SNB.overall.map(o=>o.slope),backgroundColor:SNB.overall.map(o=>o.slope==null?'#33405e':o.slope>0?'rgba(255,159,90,.8)':'rgba(90,209,160,.8)')}]},options:{plugins:{legend:{display:false},tooltip:{callbacks:{afterLabel:(x)=>'n='+(SNB.overall[x.dataIndex].n||0)+', ort yuk '+fx(SNB.overall[x.dataIndex].mean_load)}}},scales:{y:AX('MW/C'),x:AX()}}});
 if(SN&&Object.keys(SN).length){const e=Object.entries(SN);ch('cs2',{type:'bar',data:{labels:e.map(x=>x[0]),datasets:[{label:'MW/C',data:e.map(x=>x[1]),backgroundColor:e.map(x=>x[1]>0?'rgba(255,159,90,.8)':'rgba(90,209,160,.8)')}]},options:{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:AX('MW/C'),y:AX()}}});}
 if(Object.keys(HTE).length)ch('cs1',{type:'bar',data:{labels:hrs.map(h=>h+':00'),datasets:[{label:'MW/C',data:hrs.map(h=>HTE[h]?HTE[h].slope:0),backgroundColor:hrs.map(h=>HTE[h]&&HTE[h].slope>0?'rgba(255,159,90,.75)':'rgba(90,209,160,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:AX('MW/C'),x:AX()}}});
 const sr=document.getElementById('srange');
 // aktif sezon egimi
 const mo=+D.slice(5,7);const szName=(mo>=12||mo<=2)?'Kis':(mo<=5)?'Ilkbahar':(mo<=8)?'Yaz':'Sonbahar';
 const sl=SN[szName]||SN['Yaz']||0;
 function upd(dv){const el=document.getElementById('sval');if(el)el.textContent=(dv>0?'+':'')+dv+' C';
  const mw=Math.round(sl*dv);let o='<b>Aktif sezon ('+szName+'):</b> +1C -> '+sl+' MW<br>';
  o+='<b>'+ (dv>0?'+':'')+dv+'C sapma:</b> yuk ~<b style="color:'+(mw>0?'#ff9f5a':'#5ad1a0')+'">'+(mw>0?'+':'')+mw+' MW</b>';
  if(FC)o+=' &rarr; yeni pik ~<b>'+Math.round(Math.max(...FC)+mw)+' MWh</b>';
  document.getElementById('scenOut').innerHTML=o;}
 if(sr){sr.oninput=function(){upd(+this.value);};}upd(0);};

// ═══ TAB 4: CROSS CHECK ═══
T.cross=()=>{const k=Object.keys(CP);if(!k.length)return '<h2>Cross Check</h2><p class="lead">Veri yok</p>';
 let cells='<div class="panel"><h3>Sicaklik &times; Yuk</h3><div class="chartbox"><canvas id="cx1"></canvas></div></div>';
 if(HAS.ghi)cells+='<div class="panel"><h3>GHI &times; Yuk</h3><div class="chartbox"><canvas id="cx2"></canvas></div></div>';
 if(HAS.cloud)cells+='<div class="panel"><h3>Bulut &times; Yuk</h3><div class="chartbox"><canvas id="cx3"></canvas></div></div>';
 if(HAS.wind)cells+='<div class="panel"><h3>Ruzgar &times; Yuk</h3><div class="chartbox"><canvas id="cx4"></canvas></div></div>';
 if(HAS.humidity)cells+='<div class="panel"><h3>Nem &times; Yuk</h3><div class="chartbox"><canvas id="cx5"></canvas></div></div>';
 if(HAS.precip)cells+='<div class="panel"><h3>Yagis &times; Yuk</h3><div class="chartbox"><canvas id="cx6"></canvas></div></div>';
 return '<h2>Cross Check</h2><p class="lead">Referans gunlerde hava &times; yuk sacilimi</p><div class="grid2">'+cells+'</div>';};
g._i.cross=()=>{const k=Object.keys(CP);if(!k.length)return;
 function sc(id,key,ax){let ds=[];if(WX&&WX[key]&&FC)ds.push({label:'TAHMIN',data:WX[key].map((v,i)=>({x:v,y:FC[i]})),backgroundColor:'#E53935',pointRadius:5});
  k.forEach((l,i)=>{if(CP[l][key]&&CP[l].load)ds.push({label:l,data:CP[l][key].map((v,j)=>({x:v,y:CP[l].load[j]})),backgroundColor:pal[(i+1)%pal.length],pointRadius:3});});
  if(ds.length)ch(id,{type:'scatter',data:{datasets:ds},options:{plugins:{legend:LEG},scales:{x:AX(ax),y:AX('MWh')}}});}
 sc('cx1','temp','C');if(HAS.ghi)sc('cx2','ghi','GHI W/m2');if(HAS.cloud)sc('cx3','cloud','Bulut %');if(HAS.wind)sc('cx4','wind','Ruzgar');if(HAS.humidity)sc('cx5','humidity','Nem %');if(HAS.precip)sc('cx6','precip','Yagis');};

// ═══ TAB 5: ONERILER (konsolide ikinci gorus) ═══
T.rec=()=>{let h='<h2>Oneriler</h2><p class="lead">Takvim + gecmis yuk + cok-istasyon hava + benzer gun + model uzlasisi + saatlik performans</p>';
 if(REC&&REC.some(r=>r.exp!=null))h+='<div class="panel"><h3>Model tahmini vs konsolide beklenti + %95 belirsizlik bandi</h3><div class="chartbox tall"><canvas id="cr1"></canvas></div></div>';
 h+='<div id="recList"></div>';return h;};
g._i.rec=()=>{
 if(!REC)return;
 const hasExp=REC.some(r=>r.exp!=null);
 if(hasExp)ch('cr1',{type:'line',data:{labels:hrs,datasets:[
   {label:'MODEL',data:REC.map(r=>r.fc),borderColor:'#E53935',borderWidth:3,pointRadius:3},
   {label:'Konsolide Beklenen',data:REC.map(r=>r.exp),borderColor:'#5ad1a0',borderWidth:2,pointRadius:2,borderDash:[5,3]},
   {label:'P95 ust',data:REC.map(r=>r.hi),borderColor:'transparent',backgroundColor:'rgba(90,209,160,.10)',pointRadius:0,fill:'+1'},
   {label:'P95 alt',data:REC.map(r=>r.lo),borderColor:'transparent',pointRadius:0,fill:false}
  ]},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('MWh')}}});
 // oneri kartlari: en cok sapan saatler
 let flagged=REC.filter(r=>r.exp!=null&&r.fc!=null).map(r=>{
   const md=r.lastweek!=null?r.fc-r.lastweek:null, ed=r.lastweek!=null?r.exp-r.lastweek:null;
   const out=(r.hi!=null&&r.lo!=null)&&(r.fc>r.hi||r.fc<r.lo);
   const gap=Math.abs(r.fc-r.exp);
   return {...r,md,ed,out,gap};
 }).sort((a,b)=>b.gap-a.gap);
 let top=flagged.filter(r=>r.out||r.gap>15).slice(0,8);
 let html='';
 if(!top.length)html='<div class="rm ok"><h4>Model konsolide beklentiyle uyumlu</h4><p>Hicbir saatte model tahmini belirsizlik bandi disinda degil; belirgin sapma yok.</p></div>';
 top.forEach(r=>{const cls=r.out?'danger':'warn';
   html+='<div class="rm '+cls+'"><h4>Saat '+r.h+':00 '+(r.out?'<span class="pill" style="background:#3a1f1f;color:#ff9b9b">P95 DISI</span>':'<span class="pill" style="background:#3a331f;color:#ffce6a">SAPMA</span>')+'</h4>';
   html+='<div class="meta"><span>Model: <b>'+fx(r.fc)+' MWh</b></span><span>Beklenen: <b>'+fx(r.exp)+' MWh</b></span>';
   if(r.lo!=null)html+='<span>%95 bant: '+fx(r.lo)+' &ndash; '+fx(r.hi)+' MWh</span>';
   html+='<span>Guven: <b>'+(r.confidence||'--')+' '+(r.confidence_score!=null?'%'+fx(r.confidence_score,0):'')+'</b></span><span>n='+r.n+'</span></div>';
   let s='Saat '+r.h+':00 icin model <b>'+fx(r.fc)+' MWh</b> ongoruyor';
   if(r.md!=null)s+=' (gecen haftaya gore '+(r.md>0?'+':'')+fx(r.md)+' MWh)';
   s+='. Konsolide ikinci gorus <b>'+fx(r.exp)+' MWh</b>';
   if(r.ed!=null)s+=' (yani ~'+(r.ed>0?'+':'')+fx(r.ed)+' MWh degisim)';
   s+='. ';
   if(r.r2!=null)s+=' R2='+fx(r.r2,2)+', benzer-gun uzakligi='+fx(r.analog_distance,2)+'.';
   if(r.model_spread!=null)s+=' Model yayilimi '+fx(r.model_spread)+' MWh.';
   if(r.mape30!=null)s+=' Saatlik MAPE30 %'+fx(r.mape30,1)+'.';
   if(r.drivers&&r.drivers.length)s+=' Ana etkenler: <b>'+r.drivers.join(', ')+'</b>. ';
   if(r.out)s+='Model tahmini <b>%95 tahmin araligi ('+fx(r.lo)+'-'+fx(r.hi)+' MWh) DISINDA</b> &mdash; '+(r.fc>r.hi?'asiri yuksek, dusurulmesi':'asiri dusuk, yukseltilmesi')+' degerlendirilebilir.';
   else s+='Fark belirsizlik bandi icinde; model ile konsolide beklenti arasinda ~'+fx(r.gap)+' MWh acik var.';
   html+='<p>'+s+'</p></div>';});
 document.getElementById('recList').innerHTML=html;};

// ═══ TAB 6: DRIFT (D -> D+2) ═══
T.drift=()=>{let h='<h2>D &rarr; D+2 Kayma</h2><p class="lead">Sicaklik ve tuketim benzerliginin son donemdeki degisimi</p>';
 h+='<div class="grid2"><div class="panel"><h3>Saatlik sicaklik: tahmin vs gecmis</h3><div class="chartbox"><canvas id="cd1"></canvas></div></div>';
 h+='<div class="panel"><h3>Gunluk ort. sicaklik trendi (son 2 hafta)</h3><div class="chartbox"><canvas id="cd2"></canvas></div></div></div>';
 if(DRIFT.sim&&DRIFT.sim.length){h+='<div class="panel"><h3>Tuketim benzerligi: D+2 tahmini vs referans gunler</h3><table class="dt"><tr><th>Referans</th><th>MAPE %</th><th>Sekil korelasyon</th><th>Yorum</th></tr>';
  DRIFT.sim.forEach(s=>{const yor=s.mape<3?'cok benzer':s.mape<7?'benzer':'farkli';h+='<tr><td>'+s.lbl+'</td><td style="color:'+(s.mape<3?'#5ad1a0':s.mape<7?'#ffce6a':'#ff6b6b')+'">'+fx(s.mape,1)+'</td><td>'+fx(s.corr,2)+'</td><td style="color:#8b97b3">'+yor+'</td></tr>';});
  h+='</table></div>';}
 if(DRIFT.hourly_dtemp_slope!=null)h+='<div class="rm"><h4>Saatler-arasi degisim (son 7 gun)</h4><p>Bir saatten digerine sicaklik +1C degistiginde tuketim ~<b>'+fx(DRIFT.hourly_dtemp_slope)+' MW</b> yonunde hareket etti.</p></div>';
 return h;};
g._i.drift=()=>{
 let ds=[];if(DRIFT.temp_fc)ds.push({label:'D+2 tahmin',data:DRIFT.temp_fc,borderColor:'#E53935',borderWidth:3,pointRadius:2});
 if(DRIFT.temp_lastweek)ds.push({label:'1 hafta once',data:DRIFT.temp_lastweek,borderColor:'#1E88E5',borderWidth:1.5,pointRadius:0,borderDash:[6,3]});
 if(DRIFT.temp_2week)ds.push({label:'2 hafta once',data:DRIFT.temp_2week,borderColor:'#43A047',borderWidth:1.5,pointRadius:0,borderDash:[6,3]});
 if(ds.length)ch('cd1',{type:'line',data:{labels:hrs,datasets:ds},options:{plugins:{legend:LEG},scales:{x:AX(),y:AX('C')}}});
 if(DRIFT.daily_temp&&DRIFT.daily_temp.length)ch('cd2',{type:'line',data:{labels:DRIFT.daily_temp.map(x=>x[0].slice(5)),datasets:[{label:'Ort C',data:DRIFT.daily_temp.map(x=>x[1]),borderColor:'#ff9f5a',borderWidth:2,pointRadius:3,tension:.3}]},options:{plugins:{legend:{display:false}},scales:{x:AX(),y:AX('C')}}});};

// ═══ TAB 7: OZEL GUNLER ═══
T.ozel=()=>{let h='<h2>Ozel Gun Etkileri</h2><p class="lead">Cuma namazi, hafta sonu ve tatil kirilimlari</p>';
 if(SPECIAL.target_is_special)h+='<div class="rm" style="border-left-color:#8E24AA"><h4>D+2 ozel gun: '+SPECIAL.target_is_special+'</h4><p>Tahmin gunu ozel gune denk geliyor &mdash; tatil/arife yuk dususu goz onunde bulundurulmali.</p></div>';
 if(SPECIAL.friday){h+='<div class="panel"><h3>Cuma namazi etkisi (12:00-13:00)</h3><p class="note">Hafta ici (Sal-Per) ayni saatlere gore fark</p>';
  h+='<p style="font-size:13px">Ortalama: <b style="color:'+(SPECIAL.friday.mw<0?'#5ad1a0':'#ff9f5a')+'">'+fx(SPECIAL.friday.mw)+' MW ('+fx(SPECIAL.friday.pct,1)+'%)</b></p>';
  if(SPECIAL.friday_by_temp&&SPECIAL.friday_by_temp.length){h+='<table class="dt"><tr><th>Sicaklik araligi</th><th>Etki (MW)</th><th>n</th></tr>';
   SPECIAL.friday_by_temp.forEach(r=>h+='<tr><td>'+r.range+'</td><td style="color:'+(r.mw<0?'#5ad1a0':'#ff9f5a')+'">'+fx(r.mw)+'</td><td>'+r.n+'</td></tr>');h+='</table>';}
  h+='</div>';}
 h+='<div class="panel"><h3>Pazar gunu saatlik etki (haftaici %)</h3><div class="chartbox"><canvas id="co1"></canvas></div></div>';
 return h;};
g._i.ozel=()=>{
 if(SPECIAL.sunday_pct){const v=hrs.map(h=>SPECIAL.sunday_pct[h]!=null?SPECIAL.sunday_pct[h]:null);
  ch('co1',{type:'bar',data:{labels:hrs.map(h=>h+':00'),datasets:[{label:'Pazar vs haftaici %',data:v,backgroundColor:v.map(x=>x==null?'#33405e':x<0?'rgba(90,209,160,.75)':'rgba(255,159,90,.75)')}]},options:{plugins:{legend:{display:false}},scales:{y:AX('%'),x:AX()}}});}};

// ═══ BOOT ═══
const NAV=[['ozet','Ozet'],['karsi','Karsilastirma'],['sens','Sicaklik & Duyarlilik'],['cross','Cross Check'],['rec','Oneriler'],['drift','D -> D+2 Kayma'],['ozel','Ozel Gunler']];
document.getElementById('nv').innerHTML=NAV.map(x=>'<button data-t="'+x[0]+'">'+x[1]+'</button>').join('');
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>g(b.dataset.t));
NAV.forEach(x=>{const d=document.createElement('div');d.className='tab';d.id='t_'+x[0];d.innerHTML=T[x[0]]();document.getElementById('mn').appendChild(d);});
g(location.hash&&document.getElementById('t_'+location.hash.slice(1))?location.hash.slice(1):'ozet');
</script></body></html>"""
