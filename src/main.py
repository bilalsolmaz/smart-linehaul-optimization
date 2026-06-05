"""
Ana çalıştırma scripti (v6 - Competition).
Tüm pipeline'ı yönetir: veri yükleme → bias hesaplama → tahmin → düzeltme → optimizasyon → çıktı.

v6 İyileştirmeler:
- [A] Bias düzeltmesi (rolling CV tabanlı DOW çarpanları)
- [B] Rolling cross-validation (multi-window metrikler)
- [C] Konsolidasyon maliyet entegrasyonu
- [D] Post-holiday rebound tespiti
- [E] LightGBM hiperparametre tuning
"""
import argparse
import logging
import os
import sys
import time
import pandas as pd
import numpy as np

# Encoding sorunlarını çöz
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Proje kökünü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load_all_data
from src.distance_calculator import build_distance_matrix
from src.demand_forecaster import (
    forecast_ensemble, backtest, compute_quantile_forecasts,
    rolling_cv, compute_bias_factors, apply_bias_correction,
    detect_rebound_period,
)
from src.vehicle_optimizer import (
    optimize_all_routes, calculate_rental_daily_cost,
    find_consolidation_opportunities, estimate_consolidation_savings,
)


# ──────────────────────────────────────────────
# Logging Konfigürasyonu
# ──────────────────────────────────────────────

def setup_logging():
    """Logging altyapısını yapılandırır — konsol + dosya."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(config.OUTPUT_DIR, "pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger(__name__)


# ──────────────────────────────────────────────
# CLI Argümanları
# ──────────────────────────────────────────────

def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="Hepsiburada Lojistik Anahat Optimizasyonu v6"
    )
    parser.add_argument(
        "--skip-backtest", action="store_true",
        help="Backtesting adımını atla (hızlı çalıştırma için)"
    )
    parser.add_argument(
        "--skip-rolling-cv", action="store_true",
        help="Rolling cross-validation adımını atla"
    )
    parser.add_argument(
        "--no-bias", action="store_true",
        help="Bias düzeltmesini devre dışı bırak"
    )
    parser.add_argument(
        "--forecast-start", type=str, default=config.FORECAST_START,
        help="Tahmin başlangıç tarihi (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--forecast-end", type=str, default=config.FORECAST_END,
        help="Tahmin bitiş tarihi (YYYY-MM-DD)"
    )
    return parser.parse_args()


def main():
    logger = setup_logging()
    args = parse_args()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  HEPSİBURADA LOJİSTİK ANAHAT OPTİMİZASYONU v6")
    logger.info("  Talep Tahminleme & Spot Araç Optimizasyonu (Competition)")
    logger.info("=" * 70)

    # ══════════════════════════════════════════
    # Faz 0: Veri Yükleme
    # ══════════════════════════════════════════
    logger.info(">>> FAZ 0: Veri Yükleme")
    data = load_all_data()
    talep_df = data["talep"]
    koordinatlar = data["koordinatlar"]
    kiralik_df = data["kiralik"]
    arac_maliyet_df = data["arac_maliyet"]

    distances = build_distance_matrix(koordinatlar)
    logger.info(f"Mesafe matrisi: {len(distances)} rota çifti")

    routes = list(talep_df.groupby(["cikis", "varis"]).groups.keys())
    logger.info(f"Toplam rota: {len(routes)}")

    # ══════════════════════════════════════════
    # Faz 1A: [B] Rolling Cross-Validation
    # ══════════════════════════════════════════
    rolling_cv_results = None
    if not args.skip_rolling_cv:
        logger.info("=" * 70)
        logger.info(">>> FAZ 1A: Rolling Cross-Validation")
        logger.info("=" * 70)
        rolling_cv_results = rolling_cv(talep_df, routes)

    # ══════════════════════════════════════════
    # Faz 1B: [A] Bias Faktörleri Hesapla
    # ══════════════════════════════════════════
    bias_factors = None
    if config.BIAS_CORRECTION_ENABLED and not args.no_bias:
        logger.info("=" * 70)
        logger.info(">>> FAZ 1B: Bias Faktörleri (Rolling CV tabanlı)")
        logger.info("=" * 70)
        bias_factors = compute_bias_factors(talep_df, routes)

    # ══════════════════════════════════════════
    # Faz 1C: Single Holdout Backtest
    # ══════════════════════════════════════════
    if not args.skip_backtest:
        logger.info("=" * 70)
        logger.info(">>> FAZ 1C: Backtesting (4-10 Mayıs holdout)")
        logger.info("=" * 70)

        backtest_results = backtest(
            talep_df,
            config.HOLDOUT_START,
            config.HOLDOUT_END,
            routes,
        )
    else:
        logger.info(">>> FAZ 1C: Backtesting ATLANDI (--skip-backtest)")
        backtest_results = {
            "wmape": None, "mape": None, "daily_wmape": None,
            "rmse": None, "mae": None, "over_forecast_desi": None,
        }

    # ══════════════════════════════════════════
    # Faz 2: Refit & Nihai Tahmin (Bias Düzeltmeli)
    # ══════════════════════════════════════════
    logger.info("=" * 70)
    logger.info(f">>> FAZ 2: Nihai Tahmin ({args.forecast_start} - {args.forecast_end})")
    logger.info("=" * 70)

    forecast_dates = pd.date_range(args.forecast_start, args.forecast_end, freq="D")

    # [D] Post-holiday rebound tespiti
    rebound_factor = detect_rebound_period(forecast_dates)

    # Ensemble tahmin (LightGBM tuning aktif)
    tahmin_df = forecast_ensemble(
        talep_df,
        forecast_dates,
        routes,
        tune_lgb=True,
    )

    logger.info(f"Ham tahmin toplam: {tahmin_df['tahmin_desi'].sum():,.0f} desi")

    # [A] Bias düzeltmesi uygula
    if bias_factors:
        tahmin_df = apply_bias_correction(tahmin_df, bias_factors)

    # [D] Rebound düzeltmesi uygula
    if rebound_factor > 1.0:
        logger.info(f"Post-holiday rebound düzeltmesi: ×{rebound_factor:.3f}")
        # Sadece hafta içi günlere uygula (hafta sonu zaten düşük)
        weekday_mask = tahmin_df["tarih"].dt.dayofweek < 5
        tahmin_df.loc[weekday_mask, "tahmin_desi"] = (
            tahmin_df.loc[weekday_mask, "tahmin_desi"] * rebound_factor
        ).round(2)

    logger.info(f"Düzeltilmiş tahmin toplam: {tahmin_df['tahmin_desi'].sum():,.0f} desi")

    # Günlük dağılım
    logger.info(f"Günlük toplam tahmin:")
    daily_totals = tahmin_df.groupby("tarih")["tahmin_desi"].sum()
    for tarih, total in daily_totals.items():
        day_name = pd.to_datetime(tarih).strftime("%A")
        logger.info(f"  {pd.to_datetime(tarih).strftime('%Y-%m-%d')} ({day_name}): {total:,.0f} desi")

    # Quantile tahminleri
    logger.info("P10-P50-P90 güven aralıkları hesaplanıyor...")
    quantile_df = compute_quantile_forecasts(talep_df, forecast_dates, routes, tahmin_df)
    logger.info(f"Quantile tahminleri: {len(quantile_df)} satır")

    # ══════════════════════════════════════════
    # Faz 3: Spot Araç Optimizasyonu
    # ══════════════════════════════════════════
    logger.info("=" * 70)
    logger.info(">>> FAZ 3: Spot Araç Optimizasyonu")
    logger.info("=" * 70)

    planning_df, cost_summary = optimize_all_routes(
        tahmin_df, kiralik_df, arac_maliyet_df, distances
    )

    # [C] Konsolidasyon fırsatları — maliyet entegrasyonlu
    logger.info("Konsolidasyon fırsatları tespit ediliyor...")
    consol_df = find_consolidation_opportunities(
        tahmin_df, arac_maliyet_df, distances
    )

    consol_savings = 0
    if len(consol_df) > 0:
        consol_savings = estimate_consolidation_savings(
            consol_df, arac_maliyet_df
        )
        logger.info(f"{len(consol_df)} konsolidasyon fırsatı bulundu")
        logger.info(f"Toplam mesafe tasarrufu: {consol_df['tasarruf_km'].sum():.0f} km")
        logger.info(f"Tahmini maliyet tasarrufu: {consol_savings:,.0f} TL")
    else:
        logger.info("Bu veri setinde konsolidasyon fırsatı bulunamadı.")

    # Konsolidasyon bilgisi (yalnızca bilgilendirme — yarışma kuralı gereği
    # bu aşamada konsolidasyon sürece dahil değildir)
    cost_summary["konsolidasyon_tasarrufu_bilgi"] = consol_savings

    # ══════════════════════════════════════════
    # Faz 4: Çıktıları Kaydet
    # ══════════════════════════════════════════
    logger.info("=" * 70)
    logger.info(">>> FAZ 4: Çıktıları Kaydet")
    logger.info("=" * 70)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. Tahmin Excel
    tahmin_output = tahmin_df[["cikis", "varis", "tarih", "tahmin_desi"]].copy()
    tahmin_output.columns = [
        "Çıkış Transfer Merkezi",
        "Varış Transfer Merkezi",
        "Tarih",
        "Tahmin Desi",
    ]
    tahmin_output["Tarih"] = tahmin_output["Tarih"].dt.strftime("%Y-%m-%d")
    tahmin_output.to_excel(config.TAHMIN_OUTPUT_FILE, index=False, engine="openpyxl")
    logger.info(f"Tahmin kaydedildi: {config.TAHMIN_OUTPUT_FILE}")

    # 2. Araç Planlama Excel
    planning_output = planning_df.copy()
    planning_output.columns = [
        "Tarih", "Gün", "Çıkış Transfer Merkezi", "Varış Transfer Merkezi",
        "Tahmin Desi", "Kiralık Kapasite (desi)", "Kalan Talep (desi)",
        "Mesafe (km)", "Spot Tır", "Spot Kamyon", "Spot Hafif Kamyon",
        "Spot Kamyonet", "Spot Toplam Kapasite (desi)", "Spot Maliyet (TL)",
    ]
    planning_output["Tarih"] = pd.to_datetime(planning_output["Tarih"]).dt.strftime("%Y-%m-%d")
    planning_output.to_excel(config.ARAC_PLANLAMA_OUTPUT_FILE, index=False, engine="openpyxl")
    logger.info(f"Araç planlaması kaydedildi: {config.ARAC_PLANLAMA_OUTPUT_FILE}")

    # 3. Quantile tahminleri Excel
    quantile_output = quantile_df.copy()
    quantile_output["tarih"] = quantile_output["tarih"].dt.strftime("%Y-%m-%d")
    quantile_path = os.path.join(config.OUTPUT_DIR, "tahmin_quantile.xlsx")
    quantile_output.to_excel(quantile_path, index=False, engine="openpyxl")
    logger.info(f"Quantile tahminleri kaydedildi: {quantile_path}")

    # 4. Konsolidasyon önerileri Excel
    if len(consol_df) > 0:
        consol_path = os.path.join(config.OUTPUT_DIR, "konsolidasyon_onerileri.xlsx")
        consol_output = consol_df.copy()
        consol_output["tarih"] = pd.to_datetime(consol_output["tarih"]).dt.strftime("%Y-%m-%d")
        consol_output.to_excel(consol_path, index=False, engine="openpyxl")
        logger.info(f"Konsolidasyon önerileri kaydedildi: {consol_path}")

    # ══════════════════════════════════════════
    # Final Özet
    # ══════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("=" * 70)
    logger.info("  SONUÇ ÖZETİ")
    logger.info("=" * 70)

    if rolling_cv_results and rolling_cv_results.get("avg_wmape") is not None:
        logger.info(f"📊 Rolling CV Sonuçları ({rolling_cv_results['n_windows']} pencere):")
        logger.info(f"     WMAPE (ortalama): {rolling_cv_results['avg_wmape']:.2f}%")
        logger.info(f"     Günlük WMAPE (ortalama): {rolling_cv_results['avg_daily_wmape']:.2f}%")
        logger.info(f"     MAPE (ortalama): {rolling_cv_results['avg_mape']:.2f}%")

    if backtest_results.get("wmape") is not None:
        logger.info(f"📊 Single Holdout (4-10 Mayıs):")
        logger.info(f"     WMAPE (ağırlıklı): {backtest_results['wmape']:.2f}%")
        logger.info(f"     MAPE (talep>100): {backtest_results['mape']:.2f}%")
        logger.info(f"     Günlük WMAPE: {backtest_results.get('daily_wmape', 0):.2f}%")
        logger.info(f"     RMSE: {backtest_results['rmse']:,.0f}")
        logger.info(f"     MAE: {backtest_results['mae']:,.0f}")

    if bias_factors:
        logger.info(f"🔧 Bias Düzeltmesi:")
        day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
        for dow in range(7):
            logger.info(f"     {day_names[dow]}: ×{bias_factors[dow]:.3f}")

    logger.info(f"📦 Talep Tahminleri ({args.forecast_start} - {args.forecast_end}):")
    logger.info(f"     Toplam tahmin: {tahmin_df['tahmin_desi'].sum():,.0f} desi")
    logger.info(f"     Günlük ortalama: {tahmin_df['tahmin_desi'].sum() / len(forecast_dates):,.0f} desi")

    logger.info(f"🚛 Maliyet Özeti:")
    logger.info(f"     Kiralık araç (haftalık): {cost_summary['kiralik_haftalik_maliyet']:,.0f} TL")
    logger.info(f"     Spot araç (haftalık): {cost_summary['spot_haftalik_maliyet']:,.0f} TL")
    logger.info(f"     ═══════════════════════════════════════")
    logger.info(f"     TOPLAM MALİYET: {cost_summary['toplam_haftalik_maliyet']:,.0f} TL")
    if consol_savings > 0:
        logger.info(f"     (Bilgi: Konsolidasyon potansiyeli: −{consol_savings:,.0f} TL — bu aşamada dahil değil)")

    logger.info(f"🚚 Spot Araç Dağılımı (haftalık toplam):")
    for arac, adet in cost_summary["spot_arac_toplam"].items():
        logger.info(f"     {arac}: {int(adet)} araç")

    if len(consol_df) > 0:
        logger.info(f"🔄 Konsolidasyon: {len(consol_df)} fırsat, "
                     f"{consol_df['tasarruf_km'].sum():.0f} km, "
                     f"{consol_savings:,.0f} TL tasarruf")

    logger.info(f"⏱️  Çalışma süresi: {elapsed:.1f} saniye")
    logger.info(f"📄 Log dosyası: {os.path.join(config.OUTPUT_DIR, 'pipeline.log')}")

    return cost_summary, backtest_results, rolling_cv_results


if __name__ == "__main__":
    cost_summary, backtest_results, rolling_cv_results = main()
