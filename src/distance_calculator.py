"""
Mesafe hesaplama modülü.
Haversine formülü ile kuş uçuşu mesafesi × 1.3 karayolu faktörü.
"""
import logging
import math
from . import config

logger = logging.getLogger(__name__)


def haversine_km(lat1, lon1, lat2, lon2):
    """
    İki koordinat arası kuş uçuşu mesafesini km olarak hesaplar.
    """
    R = 6371  # Dünya yarıçapı (km)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def road_distance_km(lat1, lon1, lat2, lon2):
    """
    Tahmini karayolu mesafesi = kuş uçuşu × ROAD_FACTOR.
    """
    return haversine_km(lat1, lon1, lat2, lon2) * config.ROAD_FACTOR


def build_distance_matrix(koordinatlar):
    """
    Tüm merkez çiftleri arası karayolu mesafe matrisini oluşturur.

    Args:
        koordinatlar: dict {merkez_adi: (enlem, boylam)}
    Returns:
        dict {(cikis, varis): mesafe_km}
    """
    distances = {}
    merkezler = list(koordinatlar.keys())

    for i, m1 in enumerate(merkezler):
        for j, m2 in enumerate(merkezler):
            if i != j:
                lat1, lon1 = koordinatlar[m1]
                lat2, lon2 = koordinatlar[m2]
                distances[(m1, m2)] = road_distance_km(lat1, lon1, lat2, lon2)

    return distances


def get_route_distance(distances, cikis, varis):
    """
    Belirli bir rota için mesafeyi döner.
    Bulunamazsa 0 döner ve uyarı verir.
    """
    key = (cikis, varis)
    if key in distances:
        return distances[key]
    else:
        logger.warning(f"Mesafe bulunamadı: {cikis} -> {varis}")
        return 0

