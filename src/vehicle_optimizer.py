"""
Spot araç optimizasyonu modülü.
Mixed-Integer Linear Programming (MILP) ile maliyet minimizasyonu.

Düzeltmeler/İyileştirmeler:
- [Düzeltme 3] Greedy fallback mantık hatası düzeltildi
- [İyileştirme 6] Konsolidasyon fırsat tespiti eklendi
- [İyileştirme 7] TM kapasite kısıtı desteği eklendi
- [İyileştirme 8] print → logging
"""
import logging
import numpy as np
import pandas as pd
from . import config
from .distance_calculator import get_route_distance

logger = logging.getLogger(__name__)


def calculate_rental_capacity(kiralik_df, arac_maliyet_df):
    """
    Kiralık araçların rota bazında günlük kapasitesini hesaplar.

    Returns:
        dict {(cikis, varis): toplam_kapasite_desi}
    """
    kapasite_map = dict(zip(arac_maliyet_df["arac_adi"], arac_maliyet_df["kapasite"]))
    rental_cap = {}

    for _, row in kiralik_df.iterrows():
        key = (row["cikis"], row["varis"])
        kap = kapasite_map.get(row["arac_turu"], 0) * row["arac_sayisi"]
        rental_cap[key] = rental_cap.get(key, 0) + kap

    return rental_cap


def calculate_rental_daily_cost(kiralik_df, arac_maliyet_df, distances):
    """
    Kiralık araçların günlük toplam sabit maliyetini hesaplar.

    Returns:
        float: günlük toplam kiralık maliyet (TL)
    """
    maliyet_map = {}
    for _, row in arac_maliyet_df.iterrows():
        maliyet_map[row["arac_adi"]] = {
            "gunluk": row["kiralik_gunluk"],
            "km": row["kiralik_km"],
        }

    total_cost = 0
    for _, row in kiralik_df.iterrows():
        arac = row["arac_turu"]
        n = row["arac_sayisi"]
        dist = get_route_distance(distances, row["cikis"], row["varis"])
        cost = n * (maliyet_map[arac]["gunluk"] + maliyet_map[arac]["km"] * dist)
        total_cost += cost

    return total_cost


def optimize_spot_vehicles_for_route(remaining_demand, arac_maliyet_df, route_distance):
    """
    Tek bir rota için spot araç optimizasyonu.
    PuLP ile MILP veya greedy fallback.

    Args:
        remaining_demand: kiralık kapasiteden sonra kalan desi talebi
        arac_maliyet_df: araç maliyet tablosu
        route_distance: rota mesafesi (km)
    Returns:
        dict: {arac_tipi: adet, ...}, toplam_maliyet
    """
    if remaining_demand <= 0:
        return {}, 0

    try:
        return _optimize_with_pulp(remaining_demand, arac_maliyet_df, route_distance)
    except ImportError:
        return _optimize_greedy(remaining_demand, arac_maliyet_df, route_distance)


def _optimize_with_pulp(remaining_demand, arac_maliyet_df, route_distance):
    """PuLP ile MILP optimizasyonu."""
    import pulp

    prob = pulp.LpProblem("SpotVehicleOptimization", pulp.LpMinimize)

    # Karar değişkenleri: her araç tipi için adet
    vehicles = {}
    costs = {}
    capacities = {}

    for _, row in arac_maliyet_df.iterrows():
        name = row["arac_adi"].replace(" ", "_")
        vehicles[row["arac_adi"]] = pulp.LpVariable(
            f"spot_{name}", lowBound=0, cat="Integer"
        )
        costs[row["arac_adi"]] = row["spot_gunluk"] + row["spot_km"] * route_distance
        capacities[row["arac_adi"]] = row["kapasite"]

    # Amaç fonksiyonu: toplam spot maliyeti minimize et
    prob += pulp.lpSum([
        vehicles[arac] * costs[arac] for arac in vehicles
    ])

    # Kısıt: toplam kapasite >= kalan talep
    prob += pulp.lpSum([
        vehicles[arac] * capacities[arac] for arac in vehicles
    ]) >= remaining_demand

    # Çöz
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        # Çözüm bulunamazsa greedy'ye düş
        logger.warning("PuLP çözüm bulunamadı, greedy'e düşülüyor")
        return _optimize_greedy(remaining_demand, arac_maliyet_df, route_distance)

    result = {}
    total_cost = 0
    for arac, var in vehicles.items():
        count = int(var.varValue)
        if count > 0:
            result[arac] = count
            total_cost += count * costs[arac]

    return result, total_cost


def _optimize_greedy(remaining_demand, arac_maliyet_df, route_distance):
    """
    Greedy optimizasyon (PuLP yoksa fallback).
    [Düzeltme 3] — Nesne eşitliği hatası ve mantık hatası düzeltildi.
    Desi başına en düşük maliyetli araç tipinden tam adet doldur, kalanı en ucuzla kapat.
    """
    vehicle_options = []
    for _, row in arac_maliyet_df.iterrows():
        total_cost = row["spot_gunluk"] + row["spot_km"] * route_distance
        cost_per_desi = total_cost / row["kapasite"] if row["kapasite"] > 0 else float("inf")
        vehicle_options.append({
            "arac": row["arac_adi"],
            "kapasite": row["kapasite"],
            "maliyet": total_cost,
            "desi_maliyet": cost_per_desi,
        })

    # Desi başına maliyete göre sırala (en ucuz önce)
    vehicle_options.sort(key=lambda x: x["desi_maliyet"])

    result = {}
    total_cost = 0.0
    remaining = remaining_demand

    for opt in vehicle_options:
        if remaining <= 0:
            break
        count = int(remaining // opt["kapasite"])  # tam adet
        if count > 0:
            result[opt["arac"]] = count
            total_cost += count * opt["maliyet"]
            remaining -= count * opt["kapasite"]

    # Kalan kesirli talebi en ucuz araçla karşıla
    if remaining > 0:
        best = vehicle_options[0]
        result[best["arac"]] = result.get(best["arac"], 0) + 1
        total_cost += best["maliyet"]

    return result, total_cost


def optimize_all_routes(tahmin_df, kiralik_df, arac_maliyet_df, distances,
                        tm_kapasite=None):
    """
    Tüm rotalar ve günler için spot araç optimizasyonu.
    [İyileştirme 7] — TM kapasite kısıtı desteği eklendi.

    Args:
        tahmin_df: DataFrame [cikis, varis, tarih, tahmin_desi]
        kiralik_df: kiralık araç listesi
        arac_maliyet_df: araç kapasite/maliyet tablosu
        distances: mesafe sözlüğü
        tm_kapasite: dict {tm_adi: {"maks_tir": int, "maks_desi": int}} veya None
    Returns:
        planning_df: DataFrame - araç planlama detayları
        cost_summary: dict - maliyet özeti
    """
    rental_cap = calculate_rental_capacity(kiralik_df, arac_maliyet_df)
    rental_daily_cost = calculate_rental_daily_cost(kiralik_df, arac_maliyet_df, distances)

    logger.info(f"Kiralık araç günlük kapasitesi: {sum(rental_cap.values()):,.0f} desi")
    logger.info(f"Kiralık araç günlük maliyeti: {rental_daily_cost:,.0f} TL")

    # TM günlük kullanım sayacı [İyileştirme 7]
    tm_daily_usage = {}  # {(tarih, tm): {"tir": 0, "desi": 0}}

    planning_rows = []
    total_spot_cost = 0
    total_dates = tahmin_df["tarih"].nunique()
    kisit_ihlal_sayisi = 0

    for tarih in sorted(tahmin_df["tarih"].unique()):
        day_data = tahmin_df[tahmin_df["tarih"] == tarih]
        day_name = pd.to_datetime(tarih).strftime("%A")

        for _, row in day_data.iterrows():
            cikis = row["cikis"]
            varis = row["varis"]
            talep = row["tahmin_desi"]
            route_key = (cikis, varis)

            # Kiralık kapasite düş
            kiralik_kap = rental_cap.get(route_key, 0)
            remaining = max(0, talep - kiralik_kap)

            # Rota mesafesi
            dist = get_route_distance(distances, cikis, varis)

            # Spot araç optimizasyonu
            if remaining > 0:
                spot_plan, spot_cost = optimize_spot_vehicles_for_route(
                    remaining, arac_maliyet_df, dist
                )
            else:
                spot_plan = {}
                spot_cost = 0

            # TM kapasite kısıtı kontrolü [İyileştirme 7]
            if tm_kapasite:
                tm_key = (tarih, cikis)
                if tm_key not in tm_daily_usage:
                    tm_daily_usage[tm_key] = {"tir": 0, "desi": 0}

                spot_tir_sayisi = spot_plan.get("Tır", 0)
                tm_daily_usage[tm_key]["tir"] += spot_tir_sayisi
                tm_daily_usage[tm_key]["desi"] += talep

                if cikis in tm_kapasite:
                    maks_tir = tm_kapasite[cikis].get("maks_tir", float("inf"))
                    maks_desi = tm_kapasite[cikis].get("maks_desi", float("inf"))
                    if (tm_daily_usage[tm_key]["tir"] > maks_tir or
                            tm_daily_usage[tm_key]["desi"] > maks_desi):
                        kisit_ihlal_sayisi += 1
                        logger.warning(f"TM kapasite ihlali: {cikis}, {tarih}")

            total_spot_cost += spot_cost

            # Toplam kapasite hesapla
            total_spot_kap = sum(
                arac_maliyet_df[arac_maliyet_df["arac_adi"] == a]["kapasite"].values[0] * n
                for a, n in spot_plan.items()
            ) if spot_plan else 0

            planning_rows.append({
                "tarih": tarih,
                "gun": day_name,
                "cikis": cikis,
                "varis": varis,
                "tahmin_desi": round(talep, 2),
                "kiralik_kapasite": kiralik_kap,
                "kalan_talep": round(remaining, 2),
                "mesafe_km": round(dist, 1),
                "spot_tir": spot_plan.get("Tır", 0),
                "spot_kamyon": spot_plan.get("Kamyon", 0),
                "spot_hafif_kamyon": spot_plan.get("Hafif Kamyon", 0),
                "spot_kamyonet": spot_plan.get("Kamyonet", 0),
                "spot_toplam_kapasite": total_spot_kap,
                "spot_maliyet_tl": round(spot_cost, 2),
            })

    planning_df = pd.DataFrame(planning_rows)

    # TM kapasite sonucu [İyileştirme 7]
    if tm_kapasite:
        if kisit_ihlal_sayisi > 0:
            logger.warning(f"Toplam {kisit_ihlal_sayisi} TM kapasite ihlali tespit edildi!")
        else:
            logger.info("TM kapasite kısıtları ihlal edilmedi.")

    # Maliyet özeti
    total_rental_cost = rental_daily_cost * total_dates
    total_cost = total_rental_cost + total_spot_cost

    cost_summary = {
        "toplam_gun": total_dates,
        "kiralik_gunluk_maliyet": rental_daily_cost,
        "kiralik_haftalik_maliyet": total_rental_cost,
        "spot_haftalik_maliyet": total_spot_cost,
        "toplam_haftalik_maliyet": total_cost,
        "gunluk_ortalama_maliyet": total_cost / total_dates,
        "spot_arac_toplam": planning_df[["spot_tir", "spot_kamyon", "spot_hafif_kamyon", "spot_kamyonet"]].sum().to_dict(),
    }

    logger.info(f"=== MALİYET ÖZETİ ===")
    logger.info(f"  Kiralık haftalık maliyet: {total_rental_cost:,.0f} TL")
    logger.info(f"  Spot haftalık maliyet: {total_spot_cost:,.0f} TL")
    logger.info(f"  TOPLAM HAFTALIK MALİYET: {total_cost:,.0f} TL")
    logger.info(f"  Günlük ortalama maliyet: {total_cost / total_dates:,.0f} TL")

    return planning_df, cost_summary


# ──────────────────────────────────────────────
# Konsolidasyon Modülü [İyileştirme 6]
# ──────────────────────────────────────────────

def find_consolidation_opportunities(tahmin_df, arac_maliyet_df,
                                     distances, threshold_desi=5000):
    """
    Düşük hacimli rotaları ara TM üzerinden birleştirme fırsatı tespit eder.
    Kural: Aynı çıkış TM'sinden aynı güne ait talep < threshold_desi ise
           yakın bir ara TM üzerinden konsolidasyon önerilir.

    Args:
        threshold_desi: Bu değerin altındaki günlük rota talebi konsolidasyon adayı
    Returns:
        DataFrame — konsolidasyon önerileri
    """
    results = []

    # Çıkış TM bazında grupla
    grouped = tahmin_df.groupby(["tarih", "cikis"])

    for (tarih, cikis), group in grouped:
        low_demand = group[group["tahmin_desi"] < threshold_desi]

        if len(low_demand) < 2:
            continue

        # Aynı çıkıştan düşük hacimli rotaları eşleştir
        low_list = low_demand.to_dict("records")
        for i in range(len(low_list)):
            for j in range(i + 1, len(low_list)):
                row1 = low_list[i]
                row2 = low_list[j]

                if row1["varis"] == row2["varis"]:
                    continue

                combined = row1["tahmin_desi"] + row2["tahmin_desi"]

                # En büyük araç kapasitesini bul
                max_kap = arac_maliyet_df["kapasite"].max()

                if combined <= max_kap * 1.1:  # %10 tolerans
                    direct_dist1 = distances.get((cikis, row1["varis"]), 0)
                    direct_dist2 = distances.get((cikis, row2["varis"]), 0)
                    via_dist = (distances.get((cikis, row1["varis"]), 0) +
                                distances.get((row1["varis"], row2["varis"]), 0))

                    if via_dist < (direct_dist1 + direct_dist2) * 0.95:
                        results.append({
                            "tarih":             tarih,
                            "cikis":             cikis,
                            "varis_1":           row1["varis"],
                            "talep_1":           round(row1["tahmin_desi"], 2),
                            "varis_2":           row2["varis"],
                            "talep_2":           round(row2["tahmin_desi"], 2),
                            "toplam_talep":      round(combined, 2),
                            "direkt_mesafe_km":  round(direct_dist1 + direct_dist2, 1),
                            "konsolidasyon_km":  round(via_dist, 1),
                            "tasarruf_km":       round((direct_dist1 + direct_dist2) - via_dist, 1),
                        })

    if results:
        return pd.DataFrame(results).drop_duplicates(
            subset=["tarih", "cikis", "varis_1", "varis_2"]
        )
    return pd.DataFrame()


def estimate_consolidation_savings(consol_df, arac_maliyet_df):
    """
    [C] Konsolidasyon fırsatlarının tahmini maliyet tasarrufunu hesaplar.

    Her fırsat için:
    - Direkt: 2 ayrı araç × ayrı mesafeler
    - Konsolide: 1 birleşik araç × konsolidasyon mesafesi
    Aradaki fark toplam tasarruf.

    Args:
        consol_df: find_consolidation_opportunities() çıktısı
        arac_maliyet_df: araç maliyet tablosu
    Returns:
        float: toplam tahmini tasarruf (TL)
    """
    if len(consol_df) == 0:
        return 0

    # En ucuz araç bilgisi (desi başına spot maliyet)
    # Konsolidasyon genelde küçük hacimler — Kamyonet veya Hafif Kamyon kullanılır
    spot_costs = {}
    for _, row in arac_maliyet_df.iterrows():
        spot_costs[row["arac_adi"]] = {
            "gunluk": row["spot_gunluk"],
            "km": row["spot_km"],
            "kapasite": row["kapasite"],
        }

    total_savings = 0

    for _, row in consol_df.iterrows():
        talep1 = row["talep_1"]
        talep2 = row["talep_2"]
        combined = row["toplam_talep"]
        direct_km = row["direkt_mesafe_km"]
        consol_km = row["konsolidasyon_km"]
        saved_km = row["tasarruf_km"]

        # Her düşük hacimli rota için uygun araç bul
        best_separate_cost = 0
        for t in [talep1, talep2]:
            best_cost = float("inf")
            for arac, info in spot_costs.items():
                if info["kapasite"] >= t:
                    km_share = direct_km / 2  # her rota ortalama yarı mesafe
                    cost = info["gunluk"] + info["km"] * km_share
                    if cost < best_cost:
                        best_cost = cost
            if best_cost < float("inf"):
                best_separate_cost += best_cost

        # Birleşik araç maliyeti
        best_combined_cost = float("inf")
        for arac, info in spot_costs.items():
            if info["kapasite"] >= combined:
                cost = info["gunluk"] + info["km"] * consol_km
                if cost < best_combined_cost:
                    best_combined_cost = cost

        if best_combined_cost < float("inf") and best_separate_cost > 0:
            saving = max(0, best_separate_cost - best_combined_cost)
            total_savings += saving

    return round(total_savings, 2)

