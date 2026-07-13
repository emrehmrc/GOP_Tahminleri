"""experiments/adm_pv_bias_correction_ab.py — Faz 2 2b-1 devami (2026-07-13) walkforward A/B.

Hipotez (12 Temmuz post-mortem'i, MASTER_PLAN.md Faz 2b-1): ADM'nin Pazar sorunu
gece/aksam degil OGLEN/PV BLOGUNA ozgu -- Final_Pred pv bloğunda (10-16) hem ham
Ensemble'dan hem naive lag168'den kotu, en kotu 3 saat hepsi solar-hours penceresinde
(PV_BIAS_SOLAR_HOURS=7-18) over-forecast. postprocess'teki `pv_bias_delta`
(src/pv_bias_correction.py — month x hour x GHI-quartile lookup) supheli: yardim
etmek yerine zarar mi veriyor?

Yontem: retrain YOK, sandbox YOK -- forecast_log_v zaten pv_bias_delta'yi RUTIN
loglar (FORECAST_LOG_SCHEMA alani), actuals_log_v'den gercek actual okunur.
Ablation: `pred_no_pv = y_pred_final - pv_bias_delta` (PV duzeltmesi hic
uygulanmamis hali) ile gercek Final_Pred karsilastirilir. Bu monitoring.duckdb'yi
SADECE read-only okur, hicbir canli dosyaya yazmaz.

Governance (MASTER_PLAN.md): sonuc olumlu olsa bile canliya otomatik girmez --
config flag (ENABLE_PV_BIAS_CORRECTION) degisikligi AYRI bir kullanici onayi
gerektirir.
"""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
import config_live as C
from monitoring.scorecard import HOUR_BLOCKS, DAY_TYPE_GROUPS

WINDOW_DAYS = 30


def mape(pred: pd.Series, actual: pd.Series) -> float:
    v = actual.notna() & pred.notna() & (actual != 0)
    if v.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[v] - pred[v]) / actual[v])) * 100)


def _load() -> pd.DataFrame:
    """forecast_log_v x actuals_log_v, T+2, pv_bias_delta dahil (scorecard._joined_hourly
    bunu secmiyor -- burada ayrica cekiliyor)."""
    con = duckdb.connect(str(C.TENANT.monitoring_db), read_only=True)
    try:
        cutoff = (pd.Timestamp.today() - pd.Timedelta(days=WINDOW_DAYS)).date().isoformat()
        df = con.execute(
            """
            SELECT f.target_date, f.target_ts, f.day_type, f.horizon_day,
                   f.y_pred_final, f.y_pred_ens_raw, f.pv_bias_delta,
                   a.y_actual
            FROM forecast_log_v f
            INNER JOIN actuals_log_v a ON f.edas_id = a.edas_id AND f.target_ts = a.target_ts
            WHERE f.horizon_day = 'T+2' AND f.target_date >= ? AND a.y_actual IS NOT NULL
            """,
            [cutoff],
        ).df()
    finally:
        con.close()
    return df


def run() -> None:
    df = _load()
    if df.empty:
        print("ADM icin veri yok.")
        return

    df["target_ts"] = pd.to_datetime(df["target_ts"])
    df["hour"] = df["target_ts"].dt.hour
    df["hour_block"] = None
    for label, hrs in HOUR_BLOCKS.items():
        df.loc[df["hour"].isin(hrs), "hour_block"] = label
    df["day_type_group"] = df["day_type"].map(DAY_TYPE_GROUPS).fillna("ozel_gun")
    df["pv_bias_delta"] = df["pv_bias_delta"].fillna(0.0)
    df["pred_no_pv"] = df["y_pred_final"] - df["pv_bias_delta"]

    n_days = df["target_date"].nunique()
    print(f"Toplam saat: {len(df)}  ({n_days} gun, {df['target_date'].min()}..{df['target_date'].max()})")
    print()

    def _row(g: pd.DataFrame) -> dict:
        return {
            "n_hours": len(g),
            "mape_final_with_pv": mape(g["y_pred_final"], g["y_actual"]),
            "mape_no_pv_correction": mape(g["pred_no_pv"], g["y_actual"]),
            "mape_ens_raw": mape(g["y_pred_ens_raw"], g["y_actual"]),
        }

    print("## Genel (tum saatler, T+2)")
    overall = _row(df)
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print()

    print("## Saat-bloguna gore (PV duzeltmesinin aktif oldugu 'pv' blogu asil test)")
    block_rows = []
    for label in HOUR_BLOCKS:
        g = df[df["hour_block"] == label]
        if g.empty:
            continue
        row = {"hour_block": label, **_row(g)}
        block_rows.append(row)
        print(f"  {label:10s} n={row['n_hours']:3d}  final(pv-duzeltmeli)={row['mape_final_with_pv']:.2f}%  "
              f"pv-duzeltmesiz={row['mape_no_pv_correction']:.2f}%  ens_raw={row['mape_ens_raw']:.2f}%")
    block_df = pd.DataFrame(block_rows)
    print()

    print("## Sadece Pazar gunleri, 'pv' blogu (12 Temmuz post-mortem'inin odagi)")
    pazar_pv = df[(df["day_type_group"] == "pazar") & (df["hour_block"] == "pv")]
    pazar_pv_row = _row(pazar_pv) if not pazar_pv.empty else None
    if pazar_pv_row:
        for k, v in pazar_pv_row.items():
            print(f"  {k}: {v}")
    else:
        print("  Pazar+pv kesisiminde veri yok.")
    print()

    print("## Gun bazinda ('pv' blogu)")
    per_day = df[df["hour_block"] == "pv"].groupby("target_date").apply(
        lambda g: pd.Series(_row(g)), include_groups=False,
    )
    print(per_day.round(2).to_string())

    delta_overall = overall["mape_final_with_pv"] - overall["mape_no_pv_correction"]
    delta_pv_block = block_df.loc[block_df["hour_block"] == "pv", "mape_final_with_pv"].iloc[0] - \
        block_df.loc[block_df["hour_block"] == "pv", "mape_no_pv_correction"].iloc[0] if not block_df.empty else float("nan")
    print()
    print(f"PV duzeltmesinin katkisi (final - no_pv, negatif=PV duzeltmesi YARDIMCI oluyor):")
    print(f"  genel: {delta_overall:+.2f}pp   pv-blogu: {delta_pv_block:+.2f}pp")

    out_dir = C.OUTPUT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "ensemble_ab_adm_pv_bias_2026-07-13.md"
    lines = [
        "# Walkforward A/B — ADM PV-bias postprocess correction (Faz 2 2b-1 devami, 2026-07-13)",
        "",
        f"Gerçek üretim verisi (forecast_log_v x actuals_log_v, T+2, son {WINDOW_DAYS} gün, "
        f"{n_days} gün, {df['target_date'].min()}..{df['target_date'].max()}). Retrain/sandbox YOK — "
        "`pv_bias_delta` zaten her canlı run'da forecast_log'a loglanıyor, ablation SADECE bu deltayı "
        "geri çıkarıp karşılaştırıyor.",
        "",
        "## Genel (tüm saatler)",
        "",
        pd.DataFrame([overall]).round(2).to_markdown(index=False),
        "",
        "## Saat-bloğuna göre",
        "",
        block_df.round(2).to_markdown(index=False) if not block_df.empty else "(veri yok)",
        "",
        "## Sadece Pazar + pv bloğu (12 Temmuz post-mortem'inin odağı)",
        "",
        pd.DataFrame([pazar_pv_row]).round(2).to_markdown(index=False) if pazar_pv_row else "(veri yok)",
        "",
        "## Gün bazında (pv bloğu)",
        "",
        per_day.round(2).to_markdown(),
        "",
        f"**PV düzeltmesinin katkısı** (final - no_pv, negatif=PV düzeltmesi YARDIMCI oluyor): "
        f"genel {delta_overall:+.2f}pp, pv-bloğu {delta_pv_block:+.2f}pp",
        "",
        "## Sonuç",
        "",
        "**KARIŞIK/YETERSİZ KANIT — canlıya değişiklik girmedi.** Genel etkisi ihmal edilebilir "
        "(pv-bloğunda +0.10pp, düzeltme ortalamada hafif zararlı ama pratik olarak nötr). Gün "
        "bazında tutarsız: 07-06/07-07'de PV düzeltmesi belirgin YARDIMCI (7.6%→6.1%), 07-08/07-09'da "
        "belirgin ZARARLI, 07-12 Pazar'da (post-mortem'in odağı) ÇOK ZARARLI (final %7.02 vs "
        "ens_raw %5.07 — düzeltmeSİZ +1.95pp daha iyi olurdu), ama 07-05 Pazar'da neredeyse nötr. "
        "Sadece 2 Pazar örneği var — istatistiksel güç çok düşük. **Bu, `pv_bias_lookup.json`'ın "
        "donmuş/statik month×hour×GHI-quartile lookup olmasının aynı hastalığı taşıdığını gösteriyor** "
        "(bkz. reddedilen GDZ sabit saatlik bias-correction denemesi — statik düzeltmeler "
        "genelleşmiyor). Kör bir `ENABLE_PV_BIAS_CORRECTION=False` değişikliği bazı günleri "
        "iyileştirirken bazılarını kötüleştirir — net kazanç kanıtlanamadı. **Öneri (uygulanmadı, "
        "ayrı onay gerekir):** statik lookup yerine 2c'deki rolling/adaptif yaklaşımın bir "
        "PV-bloğu varyantı düşünülebilir, ya da lookup'ın kaç güncel örnekle fit edildiği / hangi "
        "OOF penceresinden geldiği denetlenmeli (muhtemelen bayat/az örnekli).",
        "",
        "## Governance notu",
        "",
        "Bu rapor MASTER_PLAN.md §Faz 2c governance kuralı gereği üretildi: hiçbir postprocess "
        "değişikliği bu tür bir walkforward A/B raporu olmadan canlıya girmez. Sonuç ne olursa olsun "
        "canlıya alma (ENABLE_PV_BIAS_CORRECTION=False veya lookup yeniden-fit) AYRI bir kullanıcı "
        "onayı gerektirir.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    run()
