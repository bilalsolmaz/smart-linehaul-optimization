"""
Proje yapılandırma dosyası.
Sabit parametreler, dosya yolları, koordinatlar ve maliyet bilgileri.
"""
import os

# ──────────────────────────────────────────────
# Dosya Yolları
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Girdi dosyaları
DESI_TALEP_FILE = os.path.join(DATA_DIR, "Desi_talep.xlsx")
KOORDINATLAR_FILE = os.path.join(DATA_DIR, "Koordinatlar.xlsx")
ARAC_KAPASITE_FILE = os.path.join(DATA_DIR, "Araç_Kapasite_Maliyet.xlsx")
# Kiralık araçlar dosya adı Türkçe karakter içerdiğinden dinamik bulunacak
KIRALIK_ARACLAR_FILE = None  # data_loader tarafından set edilir

# Çıktı dosyaları
TAHMIN_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tahmin_talep.xlsx")
ARAC_PLANLAMA_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arac_planlama.xlsx")

# ──────────────────────────────────────────────
# Koordinatlar (Kocaeli eksik - hardcode)
# ──────────────────────────────────────────────
KOCAELI_KOORDINAT = (40.7654, 29.9408)

# Karayolu düzeltme faktörü
ROAD_FACTOR = 1.3

# ──────────────────────────────────────────────
# Tarih Parametreleri
# ──────────────────────────────────────────────
FORECAST_START = "2026-05-11"
FORECAST_END = "2026-05-17"
HOLDOUT_START = "2026-05-04"
HOLDOUT_END = "2026-05-10"

# ──────────────────────────────────────────────
# Rolling CV Parametreleri [İyileştirme B]
# ──────────────────────────────────────────────
ROLLING_CV_WINDOWS = [
    # (holdout_start, holdout_end) — 4 haftalık kayan pencere
    ("2026-04-13", "2026-04-19"),  # Hafta 16
    ("2026-04-20", "2026-04-26"),  # Hafta 17
    ("2026-04-27", "2026-05-03"),  # Hafta 18 (1 Mayıs tatil haftası)
    ("2026-05-04", "2026-05-10"),  # Hafta 19 (post-holiday rebound)
]

# ──────────────────────────────────────────────
# Türkiye Resmi Tatilleri (2026)
# ──────────────────────────────────────────────
PUBLIC_HOLIDAYS_2026 = [
    "2026-01-01",  # Yılbaşı
    "2026-04-23",  # Ulusal Egemenlik ve Çocuk Bayramı
    "2026-05-01",  # İşçi Bayramı
    "2026-05-19",  # Atatürk'ü Anma, Gençlik ve Spor Bayramı
    # Ramazan Bayramı 2026 - yaklaşık tarihler
    "2026-03-20",  # Ramazan Bayramı 1. gün (tahmini)
    "2026-03-21",  # Ramazan Bayramı 2. gün
    "2026-03-22",  # Ramazan Bayramı 3. gün
]

# ──────────────────────────────────────────────
# Tahminleme Parametreleri
# ──────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "stl": 0.30,
    "wma": 0.25,
    "ml": 0.30,
    "dow_avg": 0.15,  # Stabilizatör - son 6 hafta gün ortalaması
}

WMA_WEEKS = 5  # Ağırlıklı hareketli ortalama kaç hafta geriye bakacak
WMA_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]  # En son hafta en yüksek ağırlık

# ──────────────────────────────────────────────
# Bias Düzeltmesi [İyileştirme A]
# ──────────────────────────────────────────────
# Otomatik olarak rolling CV'den hesaplanır, bu sadece varsayılan değerler
BIAS_CORRECTION_ENABLED = True
BIAS_DAMPING_FACTOR = 0.70  # Overfitting önlemek için düzeltme oranını sınırla

# ──────────────────────────────────────────────
# Post-Holiday Rebound [İyileştirme D]
# ──────────────────────────────────────────────
REBOUND_DETECTION_ENABLED = True
REBOUND_LOOKBACK_DAYS = 14  # Son 14 günde tatil var mı?

# ──────────────────────────────────────────────
# LightGBM Hiperparametre Tuning [İyileştirme E]
# ──────────────────────────────────────────────
LGBM_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.02, 0.05],
    "num_leaves": [15, 25],
}

# ──────────────────────────────────────────────
# Araç Tipleri (referans - veriden de okunur)
# ──────────────────────────────────────────────
VEHICLE_TYPES = ["Tır", "Kamyon", "Hafif Kamyon", "Kamyonet"]
