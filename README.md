# Ev Kira Tahmin Modeli

Bu proje, konut özelliklerine göre kira fiyatı tahmin edebilen bir makine öğrenimi modelini içermektedir. Projede çeşitli algoritmalar karşılaştırılmış, veri ön işleme adımları uygulanmış ve performans analizleri gerçekleştirilmiştir.

## 🔗 GitHub Proje Linki



## 📊 Kullanılan Veri Seti

- **Veri Adı:** House_Rent_Prediction
- **Boyut:** 4746 satır × 12 sütun
- **Hedef Değişken:** `Rent`
- **Öznitelikler:**
  - `BHK` (Oda Sayısı)
  - `Size` (m²)
  - `Bathroom` (Banyo Sayısı)
  - `Area Type` (Alan Türü)
  - `City` (Şehir)
  - `Furnishing Status` (Mobilya Durumu)
  - `Tenant Preferred` (Kiracı Tercihi)
  - `Point of Contact` (İletişim Noktası)

## ⚙️ Veri Ön İşleme Adımları

1. **Gereksiz Sütunların Çıkarılması:** `Posted On`, `Area Locality`, `Floor` sütunları kaldırıldı.
2. **Aykırı Değer Temizliği:** `Rent` sütununda 500.000 üzeri değerler çıkarıldı.
3. **Kategorik Değişkenler:** One-Hot Encoding ile dönüştürüldü.
4. **Veri Bölme:** %70 eğitim, %30 test.
5. **Ölçekleme:** `StandardScaler` kullanılarak standartlaştırıldı.

## 📈 Korelasyon Analizi

Veri setindeki sayısal sütunlar arasındaki ilişkiler analiz edilmiştir (detaylar teknik raporda).

## 🤖 Kullanılan Makine Öğrenimi Modelleri

| Model                | MAE   | RMSE  | R²    |
|---------------------|-------|-------|-------|
| Linear Regression    | 0.352 | 0.540 | 0.530 |
| Decision Tree        | 0.229 | 0.528 | 0.550 |
| Random Forest        | 0.197 | 0.433 | 0.698 |
| XGBoost              | 0.190 | 0.429 | 0.703 |

## 🧪 Ölçeklenmemiş Verilerle Performans

| Model         | MAE     | RMSE    | R²    |
|---------------|---------|---------|-------|
| Decision Tree | 11519.99| 26454.64| 0.577 |
| Random Forest | 10123.03| 22253.36| 0.701 |
| XGBoost       | 9835.36 | 22152.75| 0.703 |

## ✅ Sonuç

- En başarılı model **XGBoost** olmuştur (`R² = 0.703`).
- Veri ölçekleme işlemi, özellikle **Decision Tree** ve **Random Forest** algoritmalarında performans artışı sağlamıştır.

---

