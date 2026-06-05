# Hepsiburada Yapay Zeka Destekli Lojistik Anahat Optimizasyonu

## 📋 Proje Özeti

Bu proje, Hepsiburada transfer merkezleri arasındaki yük taşıma taleplerini tahminleyerek, minimum maliyetle spot araç optimizasyonu gerçekleştirmektedir.

**Hedef**: 11-17 Mayıs 2026 haftası için rota bazında günlük desi talebini tahminlemek ve maliyeti minimize eden spot araç ataması yapmak.

---

## 💰 Maliyet Bilgisi (11-17 Mayıs 2026)

| Maliyet Kalemi | Tutar (TL) |
|---------------|-----------|
| Kiralık Araç (haftalık) | 855,213 |
| Spot Araç (haftalık) | 11,862,985 |
| **TOPLAM MALİYET** | **12,718,198** |

### Günlük Tahmin Profili

| Gün | Tahmin Desi | Bias | Rebound |
|-----|------------|------|---------|
| Pazartesi (11 Mayıs) | 1,793,061 | ×0.977 | ×1.077 |
| Salı (12 Mayıs) | 1,324,256 | ×0.968 | ×1.077 |
| Çarşamba (13 Mayıs) | 1,264,783 | ×0.956 | ×1.077 |
| Perşembe (14 Mayıs) | 1,085,574 | ×0.908 | ×1.077 |
| Cuma (15 Mayıs) | 1,402,090 | ×1.108 | ×1.077 |
| Cumartesi (16 Mayıs) | 855,087 | ×1.058 | — |
| Pazar (17 Mayıs) | 251,090 | ×0.923 | — |
| **Haftalık Toplam** | **7,975,941** | | |

### Spot Araç Dağılımı (Haftalık Toplam)

| Araç Tipi | Kapasite (desi) | Adet |
|-----------|----------------|------|
| Tır | 22,400 | 216 |
| Kamyon | 12,000 | 228 |
| Hafif Kamyon | 7,200 | 7 |
| Kamyonet | 5,600 | 191 |

---

## 🔬 Yöntem

### 1. Talep Tahminleme (Hibrit Ensemble Model v6)

4 aylık (1 Ocak – 10 Mayıs 2026) tarihsel veri üzerinden 4 farklı yöntemin ağırlıklı ortalaması + kalibrasyon katmanı:

| Yöntem | Ağırlık | Açıklama |
|--------|---------|----------|
| STL Decomposition | %30 | Trend + haftalık mevsimsellik çarpanları |
| Weighted Moving Average | %25 | Son 5 haftanın ağırlıklı ortalaması (tatil filtrelemeli) |
| LightGBM | %30 | 17 feature, grid search tuning (rota bazında ayrı modeller) |
| DOW Average | %15 | Son 8 haftanın gün ortalaması (stabilizatör) |

#### Feature Engineering
- **Temporal**: Haftanın günü, ay, hafta numarası, ay içi gün
- **Tatil**: Resmi tatil bayrağı, tatil sonrası birikmiş talep etkisi, rebound haftası
- **Lag**: 7, 14, 21, 28 günlük gecikmeli değerler (anomali düzeltmeli, vectorized)
- **Rolling**: 7, 14, 28 günlük hareketli ortalama ve standart sapma
- **Mevsimsel**: Aynı haftanın gününün son 4 haftadaki ortalaması

#### Anomali Tespiti ve Düzeltme
- **30 Nisan 2026**: 1 Mayıs öncesi — 23K desi (normal ~960K)
- **1 Mayıs 2026**: İşçi Bayramı — 102K desi (normal ~960K)
- **10 Mayıs 2026**: Veri kesilmesi — 25K desi (normal ~250K)

#### Kalibrasyon Katmanı

**DOW Bias Düzeltmesi**: 4 haftalık rolling cross-validation'dan elde edilen günlük bias faktörleri ile sistematik sapma düzeltilir. Damping=%70 ile overfitting önlenir.

**Post-Holiday Rebound**: Tahmin dönemi öncesindeki 14 günde tatil/anomali varsa, birikmiş talep rebound'u için çarpan uygulanır (hafta içi: ×1.077).

#### P10-P50-P90 Güven Aralığı
Her rota × gün için tarihsel varyasyon katsayısı üzerinden ±1.28σ güven aralığı.

#### LightGBM Hiperparametre Tuning
Grid search: n_estimators×max_depth×learning_rate×num_leaves, rota bazında internal 80/20 validation split ile en iyi kombinasyon seçilir.

### 2. Araç Optimizasyonu (MILP)

PuLP MILP ile her rota × gün için minimum maliyetli spot araç kombinasyonu:

```
Minimize: Σ [araç_sayısı(r,t) × (spot_sabit_maliyet(t) + spot_km_maliyet(t) × mesafe(r))]
Subject to: Σ [araç_sayısı(r,t) × kapasite(t)] + kiralık_kapasite(r) ≥ tahmin_talep(r)
```

### 3. Konsolidasyon Analizi (Ek Analiz — bu aşamada sürece dahil değil)

Düşük hacimli rotaların birleştirme fırsatları otomatik tespit ve potansiyel maliyet tasarrufu hesaplama:
- **194 fırsat** → **60,237 km mesafe** → **1,761,906 TL potansiyel tasarruf**
- *Not: Yarışma kuralları gereği bu aşamada konsolidasyon değerlendirmeye dahil değildir.*

---

## 📊 Doğrulama Sonuçları

### Rolling Cross-Validation (4 Pencere Ortalaması)

| Pencere | Tarih | WMAPE | Günlük WMAPE |
|---------|-------|-------|-------------|
| 1 | 13-19 Nisan | Normal hafta | ✅ |
| 2 | 20-26 Nisan | 23 Nisan haftası | ✅ |
| 3 | 27 Nis-3 May | 1 Mayıs haftası | ✅ |
| 4 | 4-10 Mayıs | Post-holiday | ✅ |
| **Ortalama** | | **%22.53** | **%17.10** |

### Single Holdout (4-9 Mayıs)

| Metrik | Değer |
|--------|-------|
| WMAPE (Ağırlıklı) | %23.66 |
| Günlük WMAPE | %20.29 |
| MAPE (talep > 100 desi) | %34.63 |
| RMSE | 5,308 |
| MAE | 3,675 |

---

## 📁 Dosya Yapısı

```
kod/
├── data/                           # Girdi verileri
├── output/                         # Çıktı dosyaları
│   ├── tahmin_talep.xlsx           # Tahminlenen talep (bias + rebound düzeltmeli)
│   ├── arac_planlama.xlsx          # Araç planlaması
│   ├── tahmin_quantile.xlsx        # P10-P50-P90 güven aralıkları
│   ├── konsolidasyon_onerileri.xlsx # Konsolidasyon fırsatları + maliyet
│   └── pipeline.log               # Detaylı çalışma logu
├── src/
│   ├── config.py                   # Yapılandırma + CV parametreleri
│   ├── data_loader.py              # Veri yükleme (akıllı sütun eşleme)
│   ├── distance_calculator.py      # Haversine × 1.3
│   ├── demand_forecaster.py        # Ensemble + bias + rebound + rolling CV
│   ├── vehicle_optimizer.py        # MILP + konsolidasyon + maliyet tasarrufu
│   └── main.py                     # 4 fazlı pipeline (CLI destekli)
├── requirements.txt
└── README.md
```

## 🚀 Çalıştırma

```bash
pip install -r requirements.txt

# Tam çalıştırma (Rolling CV + Bias + Backtest + Tahmin + Optimizasyon)
python -m src.main

# Hızlı (backtest + rolling CV atla)
python -m src.main --skip-backtest --skip-rolling-cv

# Bias düzeltmesiz
python -m src.main --no-bias
```
