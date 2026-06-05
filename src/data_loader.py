"""
Veri yükleme ve temizleme modülü.
Excel dosyalarını okur, sütun isimlerini standartlaştırır.

İyileştirmeler:
- [İyileştirme 8] print → logging
- [İyileştirme 9] Pozisyon bağımlılığı yerine sütun ismi ile okuma
"""
import logging
import os
import pandas as pd
from . import config

logger = logging.getLogger(__name__)

# [İyileştirme 9] Sütun ismi eşleme haritaları
TALEP_COLUMN_MAP = {
    "çıkış transfer merkezi": "cikis",
    "varış transfer merkezi":  "varis",
    "tarih":                   "tarih",
    "desi":                    "desi",
}

KOORDINAT_COLUMN_MAP = {
    "transfer merkezi": "merkez",
    "enlem":            "enlem",
    "boylam":           "boylam",
}

KIRALIK_COLUMN_MAP = {
    "çıkış transfer merkezi": "cikis",
    "varış transfer merkezi":  "varis",
    "araç sayısı":             "arac_sayisi",
    "araç türü":               "arac_turu",
}

ARAC_COLUMN_MAP = {
    "araç":     "arac_adi",
    "kapasite":  "kapasite",
}


def _find_file(keyword):
    """Türkçe karakter sorununu aşarak dosya bulur."""
    for f in os.listdir(config.DATA_DIR):
        if keyword in f and f.endswith(".xlsx"):
            return os.path.join(config.DATA_DIR, f)
    raise FileNotFoundError(f"{keyword} içeren dosya bulunamadı!")


def _smart_rename(df, column_map):
    """
    [İyileştirme 9] Sütun isimlerini akıllıca standartlaştırır.
    Önce sütun adlarını pattern ile eşlemeyi dener, bulamazsa pozisyon bazlı fallback.
    """
    actual_cols = df.columns.tolist()
    rename_dict = {}
    for actual in actual_cols:
        for pattern, standard in column_map.items():
            if pattern.lower() in actual.lower():
                rename_dict[actual] = standard
                break

    expected_count = len(set(column_map.values()))
    if len(rename_dict) >= expected_count:
        df = df.rename(columns=rename_dict)
        logger.debug(f"Sütunlar pattern ile eşleştirildi: {rename_dict}")
    else:
        # Fallback: pozisyon bazlı
        standard_names = list(dict.fromkeys(column_map.values()))  # sıra koruyan unique
        if len(actual_cols) >= len(standard_names):
            df.columns = standard_names[:len(actual_cols)] if len(actual_cols) == len(standard_names) else df.columns
            logger.debug(f"Sütunlar pozisyon bazlı atandı: {standard_names}")

    return df


def load_desi_talep():
    """
    Desi talep verisini yükler.
    Returns: DataFrame [cikis, varis, tarih, desi]
    """
    df = pd.read_excel(_find_file("Desi"))
    # [İyileştirme 9] Önce isim bazlı, sonra fallback
    actual_cols = df.columns.tolist()
    rename_dict = {}
    for actual in actual_cols:
        for pattern, standard in TALEP_COLUMN_MAP.items():
            if pattern in actual.lower():
                rename_dict[actual] = standard
                break
    if len(rename_dict) == 4:
        df = df.rename(columns=rename_dict)
    else:
        # Fallback: pozisyon bazlı (önceki davranış)
        df.columns = ["cikis", "varis", "tarih", "desi"]

    df["tarih"] = pd.to_datetime(df["tarih"])
    df = df.sort_values(["tarih", "cikis", "varis"]).reset_index(drop=True)
    return df


def load_koordinatlar():
    """
    Transfer merkezi koordinatlarını yükler.
    Kocaeli eksik - config'den eklenir.
    Returns: dict {merkez_adi: (enlem, boylam)}
    """
    df = pd.read_excel(_find_file("Koordinat"))
    # [İyileştirme 9]
    actual_cols = df.columns.tolist()
    rename_dict = {}
    for actual in actual_cols:
        for pattern, standard in KOORDINAT_COLUMN_MAP.items():
            if pattern in actual.lower():
                rename_dict[actual] = standard
                break
    if len(rename_dict) == 3:
        df = df.rename(columns=rename_dict)
    else:
        df.columns = ["merkez", "enlem", "boylam"]

    koord = {}
    for _, row in df.iterrows():
        koord[row["merkez"]] = (row["enlem"], row["boylam"])

    # Kocaeli: v2 koordinat dosyasında mevcut, güvenlik için fallback
    if "Kocaeli" not in koord:
        koord["Kocaeli"] = config.KOCAELI_KOORDINAT
        logger.warning("Kocaeli koordinatı dosyada bulunamadı, config'den eklendi.")

    return koord


def load_kiralik_araclar():
    """
    Günlük kiralık araç listesini yükler.
    Returns: DataFrame [cikis, varis, arac_sayisi, arac_turu]
    """
    filepath = _find_file("Kiral")
    df = pd.read_excel(filepath)
    # [İyileştirme 9]
    actual_cols = df.columns.tolist()
    rename_dict = {}
    for actual in actual_cols:
        for pattern, standard in KIRALIK_COLUMN_MAP.items():
            if pattern in actual.lower():
                rename_dict[actual] = standard
                break
    if len(rename_dict) == 4:
        df = df.rename(columns=rename_dict)
    else:
        df.columns = ["cikis", "varis", "arac_sayisi", "arac_turu"]
    return df


def load_arac_kapasite_maliyet():
    """
    Araç tipi kapasite ve maliyet bilgilerini yükler.
    Returns: DataFrame [arac_adi, kapasite, kiralik_gunluk, kiralik_km,
                         spot_gunluk, spot_km]
    """
    df = pd.read_excel(_find_file("Kapasite"))
    df.columns = [
        "arac_adi",
        "kapasite",
        "kiralik_gunluk",
        "kiralik_km",
        "spot_gunluk",
        "spot_km",
    ]
    return df


def load_all_data():
    """
    Tüm verileri yükler ve bir dict olarak döner.
    Returns: dict with keys: talep, koordinatlar, kiralik, arac_maliyet
    """
    data = {
        "talep": load_desi_talep(),
        "koordinatlar": load_koordinatlar(),
        "kiralik": load_kiralik_araclar(),
        "arac_maliyet": load_arac_kapasite_maliyet(),
    }
    logger.info(f"Talep: {len(data['talep'])} kayıt, "
                f"{data['talep']['tarih'].nunique()} gün, "
                f"{len(data['talep'].groupby(['cikis', 'varis']))} rota")
    logger.info(f"Koordinatlar: {len(data['koordinatlar'])} merkez")
    logger.info(f"Kiralık araçlar: {len(data['kiralik'])} rota")
    logger.info(f"Araç tipleri: {len(data['arac_maliyet'])} tip")
    return data
