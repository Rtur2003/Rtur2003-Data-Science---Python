# NOEL PERAKENDE SATIŞ ANALİZİ VE MAKİNE ÖĞRENMESİ
## Python ile Veri Bilimine Giriş - Final Projesi

---

# SLAYT 1: PROJE TANITIMI

## Proje Adı
**Noel Perakende Satış Analizi ve Teslimat Tahmini**

## Amaç
- E-ticaret lojistik verilerini analiz etmek
- Teslimat performansını etkileyen faktörleri bulmak
- Makine öğrenmesi ile teslimat durumunu tahmin etmek

## Veri Seti
- **Kaynak:** Christmas Retail Sales - Shipping & Delivery Dataset
- **İçerik:** 7 farklı tablo (OrderHeader, OrderLine, Product, Promotion, Fulfillment, Returns, Calendar)
- **Format:** Excel dosyası (.xlsx)

---

# SLAYT 2: KULLANILAN KÜTÜPHANELER

```python
import pandas as pd          # Veri işleme ve analiz
import numpy as np           # Sayısal hesaplamalar
import matplotlib.pyplot as plt  # Görselleştirme
import seaborn as sns        # İstatistiksel görselleştirme
from scipy import stats      # İstatistiksel testler
from sklearn.model_selection import train_test_split  # Veri bölme
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Ön işleme
from sklearn.linear_model import LogisticRegression  # Lojistik Regresyon
from sklearn.tree import DecisionTreeClassifier      # Karar Ağacı
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.neighbors import KNeighborsClassifier   # KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Neden Bu Kütüphaneler?**
- Pandas: Veri çerçevesi işlemleri (DataFrame)
- NumPy: Hızlı matematiksel işlemler
- Matplotlib/Seaborn: Profesyonel grafikler
- Scipy: Bilimsel hesaplamalar ve testler
- Sklearn: Makine öğrenmesi algoritmaları

---

# SLAYT 3: VERİ YÜKLEME

## Excel'den Veri Okuma
```python
excel_dosyasi = pd.ExcelFile('Christmas_Retail_Sales_and_Marketing_Analysis_Dataset.xlsx')
print("Tablolar:", excel_dosyasi.sheet_names)
```

## 7 Tablo Yüklendi:
| Tablo | Açıklama | Satır Sayısı |
|-------|----------|--------------|
| OrderHeader | Sipariş başlıkları | ~10,000+ |
| OrderLine | Sipariş detayları | ~50,000+ |
| Product | Ürün bilgileri | ~500+ |
| Promotion | Promosyon bilgileri | ~100+ |
| Fulfillment | Teslimat bilgileri | ~10,000+ |
| Returns | İade bilgileri | ~2,000+ |
| Calendar | Tarih bilgileri | 365 |

---

# SLAYT 4: TABLO YAPILARI

## Her Tablonun İncelenmesi
```python
for tablo_adi in excel_dosyasi.sheet_names:
    df_temp = pd.read_excel(excel_dosyasi, sheet_name=tablo_adi)
    print(f"{tablo_adi}: {df_temp.shape[0]} satır, {df_temp.shape[1]} sütun")
    print(f"Sütunlar: {df_temp.columns.tolist()}")
```

## Önemli Sütunlar:
- **OrderID:** Sipariş numarası (birincil anahtar)
- **ProductID:** Ürün numarası
- **CustomerID:** Müşteri numarası
- **DeliveryStatus:** Teslimat durumu (OnTime/Late)
- **ShipCost$:** Kargo maliyeti

---

# SLAYT 5: VERİ BİRLEŞTİRME (MERGE)

## Neden Birleştirme Gerekli?
Veriler 7 farklı tabloda dağılmış durumda. Analiz için tek bir tablo oluşturmalıyız.

## Merge İşlemleri
```python
# Sipariş detayı + Sipariş başlığı
df = pd.merge(order_line, order_header, on='OrderID', how='left')

# + Ürün bilgileri
df = pd.merge(df, product, on='ProductID', how='left')

# + Teslimat bilgileri
df = pd.merge(df, fulfillment, on='ShipmentID', how='left')
```

## Merge Tipleri:
- **left:** Sol tablodaki tüm satırları koru
- **inner:** Sadece eşleşen satırları al
- **outer:** Tüm satırları al

**Sonuc:** Tüm bilgiler tek DataFrame'de toplandı!

---

# SLAYT 6: EKSİK VERİ ANALİZİ

## Eksik Verileri Tespit Etme
```python
eksik_veriler = df.isnull().sum()
eksik_yuzde = (df.isnull().sum() / len(df)) * 100
```

## Neden Önemli?
- Eksik veriler model performansını düşürür
- İstatistiksel hesaplamaları etkiler
- Veri kalitesini gösterir

## Çözüm Yöntemleri:
1. **Silme:** Çok az eksik varsa satırı sil
2. **Doldurma:** Ortalama, medyan veya mod ile doldur
3. **Tahmin:** ML ile eksik değerleri tahmin et

---

# SLAYT 7: TEMEL İSTATİSTİKLER

## Merkezi Eğilim Ölçüleri
```python
print("Ortalama:", df['ShipCost$'].mean())
print("Medyan:", df['ShipCost$'].median())
print("Mod:", df['ShipCost$'].mode()[0])
```

## Dağılım Ölçüleri
```python
print("Standart Sapma:", df['ShipCost$'].std())
print("Varyans:", df['ShipCost$'].var())
print("Minimum:", df['ShipCost$'].min())
print("Maximum:", df['ShipCost$'].max())
```

## describe() Fonksiyonu
```python
df.describe()  # Tüm istatistikleri tek seferde gösterir
```

---

# SLAYT 8: KATEGORİK DEĞİŞKEN GRAFİKLERİ

## Countplot - Kategori Sayıları
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category')
plt.title('Kategori Dağılımı')
plt.xticks(rotation=45)
plt.show()
```

## Ne Gördük?
- En çok satılan kategoriler
- Kategori dengesizlikleri
- İş kararları için içerik

## Diğer Kategorik Grafikler:
- Teslimat durumu dağılımı
- Ödeme tipi dağılımı
- Kargo firması dağılımı

---

# SLAYT 9: SAYISAL DEĞİŞKEN GRAFİKLERİ

## Histogram - Dağılım
```python
plt.figure(figsize=(10, 6))
plt.hist(df['ShipCost$'], bins=30, edgecolor='black')
plt.title('Kargo Maliyeti Dağılımı')
plt.xlabel('Maliyet ($)')
plt.ylabel('Frekans')
plt.show()
```

## Ne Anlatıyor?
- Verilerin nasıl dağıldığını gösterir
- Normal dağılım mı, çarpık mı?
- Aykırı değerler var mı?

---

# SLAYT 10: BOXPLOT (KUTU GRAFİĞİ)

## Boxplot Oluşturma
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='LineRevenue$')
plt.title('Kategorilere Göre Gelir Dağılımı')
plt.xticks(rotation=45)
plt.show()
```

## Boxplot Ne Gösterir?
- **Kutu:** Q1-Q3 arası (verilerin %50'si)
- **Cizgi:** Medyan
- **Bıyıklar:** Min-Max (aykırı hariç)
- **Noktalar:** Aykırı değerler

## Aykırı Değer Tespiti
IQR yöntemi ile aykırı değerler belirlenir.

---

# SLAYT 11: KORELASYON ANALİZİ

## Korelasyon Matrisi
```python
korelasyon = df[sayisal_sutunlar].corr()
```

## Heatmap Görselleştirme
```python
plt.figure(figsize=(12, 8))
sns.heatmap(korelasyon, annot=True, cmap='coolwarm', center=0)
plt.title('Korelasyon Matrisi')
plt.show()
```

## Korelasyon Yorumu:
- **+1:** Mükemmel pozitif ilişki
- **0:** İlişki yok
- **-1:** Mükemmel negatif ilişki

## Bulduğumuz İlişkiler:
- Miktar ile Gelir arasında pozitif ilişki
- Fiyat ile Maliyet arasında pozitif ilişki

---

# SLAYT 12: GROUPBY VE AGGREGATION

## Gruplandırma İşlemleri
```python
kategori_analiz = df.groupby('Category').agg({
    'LineRevenue$': ['sum', 'mean', 'count'],
    'ShipCost$': 'mean'
})
```

## Pivot Table
```python
pivot = pd.pivot_table(df,
                       values='LineRevenue$',
                       index='Category',
                       columns='DeliveryStatus',
                       aggfunc='sum')
```

## Neden Kullanılır?
- Kategorilere göre özet istatistikler
- Karşılaştırmalı analizler
- Raporlama için özet tablolar

---

# SLAYT 13: NORMALLİK TESTİ (SHAPIRO-WILK)

## Hipotez
- **H0:** Veri normal dağılmıştır
- **H1:** Veri normal dağılmamıştır

## Test Uygulama
```python
from scipy.stats import shapiro

stat, p_value = shapiro(df['ShipCost$'].sample(5000))
print(f"P-deger: {p_value}")

if p_value > 0.05:
    print("Normal dağılım kabul edilir")
else:
    print("Normal dağılım reddedilir")
```

## Neden Önemli?
- Parametrik testler normal dağılım gerektirir
- Model seçimini etkiler

---

# SLAYT 14: T-TESTİ

## Amaç
İki grubun ortalamaları arasında anlamlı fark var mı?

## Örnek: Zamanında vs Gecikmeli Teslimat
```python
from scipy.stats import ttest_ind

grup1 = df[df['DeliveryStatus'] == 'OnTime']['ShipCost$']
grup2 = df[df['DeliveryStatus'] == 'Late']['ShipCost$']

t_stat, p_value = ttest_ind(grup1, grup2)
```

## Yorum
- **p < 0.05:** Gruplar arasında anlamlı fark var
- **p >= 0.05:** Anlamlı fark yok

---

# SLAYT 15: ANOVA TESTİ

## Amaç
İkiden fazla grubun ortalamaları arasında fark var mı?

## Örnek: Kategorilere Göre Gelir
```python
from scipy.stats import f_oneway

gruplar = [grup['LineRevenue$'] for name, grup in df.groupby('Category')]
f_stat, p_value = f_oneway(*gruplar)
```

## Yorum
- Kategoriler arasında gelir farkı var mı?
- Hangi kategori daha karlı?

---

# SLAYT 16: HEDEF DEĞİŞKEN OLUŞTURMA

## Sınıflandırma Problemi
Teslimat zamanında mı yoksa gecikmeli mi olacak?

## Hedef Değişken
```python
df['Basarili_Teslimat'] = (df['DeliveryStatus'] == 'OnTime').astype(int)
# OnTime -> 1 (Başarılı)
# Late -> 0 (Başarısız/Gecikmeli)
```

## Dağılım Kontrolü
```python
print(df['Basarili_Teslimat'].value_counts())
print(f"Başarı Oranı: %{df['Basarili_Teslimat'].mean() * 100:.2f}")
```

---

# SLAYT 17: ÖZELLİK SEÇİMİ

## Kullanılacak Özellikler
```python
ozellikler = ['Qty', 'UnitPrice', 'LineRevenue$', 'ShipCost$',
              'Category', 'PaymentType', 'Carrier', 'ServiceLevel']
```

## Neden Bu Özellikler?
- **Qty:** Sipariş miktarı teslimat süresini etkileyebilir
- **ShipCost$:** Kargo maliyeti servis kalitesini gösterir
- **Carrier:** Farklı kargo firmaları farklı performans
- **ServiceLevel:** Express vs Standard teslimat

---

# SLAYT 18: LABEL ENCODING

## Kategorik -> Sayısal Dönüşüm
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])
df['PaymentType_encoded'] = le.fit_transform(df['PaymentType'])
df['Carrier_encoded'] = le.fit_transform(df['Carrier'])
df['ServiceLevel_encoded'] = le.fit_transform(df['ServiceLevel'])
```

## Neden Gerekli?
- ML algoritmaları sayısal değer ister
- Kategorik değerleri sayıya çeviriyoruz

## Ornek:
- Electronics -> 0
- Clothing -> 1
- Books -> 2

---

# SLAYT 19: EĞİTİM-TEST AYIRMA

## Veriyi Bölme
```python
from sklearn.model_selection import train_test_split

X = df[ozellik_sutunlari]  # Özellikler
y = df['Basarili_Teslimat']  # Hedef

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Parametreler:
- **test_size=0.2:** %20 test, %80 eğitim
- **random_state=42:** Tekrarlanabilirlik

## Neden Ayırıyoruz?
- Model görünmeyen veri üzerinde test edilmeli
- Overfitting'i önlemek için

---

# SLAYT 20: ÖLÇEKLENDİRME (SCALING)

## StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Formül
```
z = (x - ortalama) / standart_sapma
```

## Neden Gerekli?
- Farklı ölçeklerdeki değişkenleri eşitler
- Örnek: Fiyat (0-1000) vs Miktar (1-10)
- KNN ve Lojistik Regresyon için kritik

---

# SLAYT 21: LOJİSTİK REGRESYON

## Model Oluşturma
```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train_scaled, y_train)
```

## Tahmin
```python
y_pred = log_reg.predict(X_test_scaled)
```

## Değerlendirme
```python
from sklearn.metrics import accuracy_score
print(f"Dogruluk: %{accuracy_score(y_test, y_pred) * 100:.2f}")
```

## Ne Zaman Kullanılır?
- İkili sınıflandırma problemleri
- Basit ve yorumlanabilir model istendiğinde

---

# SLAYT 22: KARAR AĞACI

## Model Oluşturma
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
```

## Avantajları:
- Yorumlanabilir (if-else kuralları)
- Ölçeklendirme gerektirmez
- Kategorik ve sayısal veri ile çalışır

## Dezavantajları:
- Overfitting eğilimi
- Küçük değişikliklere hassas

---

# SLAYT 23: RANDOM FOREST

## Model Oluşturma
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## Nasıl Çalışır?
- Birden fazla karar ağacı oluşturur
- Her ağaç farklı veri örneği kullanır
- Sonuçlar oylama ile birleştirilir

## Avantajları:
- Overfitting'e dayanıklı
- Özellik önemliliği gösterir
- Genellikle yüksek performans

---

# SLAYT 24: KNN (K-EN YAKIN KOMŞU)

## Model Oluşturma
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

## Nasıl Çalışır?
- Yeni veri noktası için en yakın k komşuyu bul
- Komşuların çoğunluğunun sınıfını ata

## K Değeri Seçimi:
- Küçük k: Overfitting riski
- Büyük k: Underfitting riski
- Genellikle tek sayı seçilir (3, 5, 7...)

---

# SLAYT 25: CONFUSION MATRIX

## Karışıklık Matrisi
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```

## Matris Yorumu:
|  | Tahmin: 0 | Tahmin: 1 |
|--|-----------|-----------|
| Gercek: 0 | TN | FP |
| Gercek: 1 | FN | TP |

- **TP:** Doğru Pozitif
- **TN:** Doğru Negatif
- **FP:** Yanlış Pozitif (Tip I Hata)
- **FN:** Yanlış Negatif (Tip II Hata)

---

# SLAYT 26: MODEL METRİKLERİ

## Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Metrikler:
- **Accuracy:** (TP+TN) / Toplam
- **Precision:** TP / (TP+FP) - Ne kadar kesin?
- **Recall:** TP / (TP+FN) - Ne kadar kapsamlı?
- **F1-Score:** Precision ve Recall'un harmonik ortalaması

---

# SLAYT 27: MODEL KARŞILAŞTIRMASI

## Tüm Modellerin Sonuçları

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Lojistik Regresyon | %XX | %XX | %XX | %XX |
| Karar Ağacı | %XX | %XX | %XX | %XX |
| Random Forest | %XX | %XX | %XX | %XX |
| KNN | %XX | %XX | %XX | %XX |

## En İyi Model Seçimi:
- En yüksek F1-Score'a sahip model
- İş problemine göre Precision/Recall önceliği

---

# SLAYT 28: ÇAPRAZ DOĞRULAMA

## Cross Validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Ortalama: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")
```

## Neden Önemli?
- Tek train-test bölümüne bağımlı olmaz
- Modelin genel performansını gösterir
- Overfitting kontrolü sağlar

---

# SLAYT 29: ÖZELLİK ÖNEMLİLİĞİ

## Feature Importance (Random Forest)
```python
importance = rf.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title('Özellik Önemliliği')
plt.show()
```

## Yorum:
- Hangi değişkenler tahmini en çok etkiliyor?
- Önemli özelliklere odaklanma
- Gereksiz özellikleri çıkarma

---

# SLAYT 30: SONUÇ VE ÖNERİLER

## Proje Özeti
1. 7 tablodan oluşan e-ticaret verisi analiz edildi
2. Veri birleştirme ve temizleme yapıldı
3. Kapsamlı keşifsel veri analizi gerçekleştirildi
4. İstatistiksel testler uygulandı
5. 4 farklı ML modeli karşılaştırıldı

## Bulgular
- Teslimat performansını etkileyen faktörler belirlendi
- En başarılı model: [Model Adı]
- Önemli özellikler: [Liste]

## İş Önerileri
- Teslimat sürelerini iyileştirmek için [öneri]
- Müşteri memnuniyetini artırmak için [öneri]

---

# SLAYT 31: KAYNAKLAR

## Kullanılan Teknolojiler
- Python 3.x
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

## Veri Seti
- Christmas Retail Sales - Shipping & Delivery Dataset

## Referanslar
- Ders Notları: Python ile Veri Bilimine Giriş
- Scikit-learn Documentation
- Pandas Documentation

---

# TEŞEKKÜRLER!

## Sorular?

**Proje Sahibi:** [İsim]
**Ders:** Python ile Veri Bilimine Giriş ve Makine Öğrenmesi
**Tarih:** Ocak 2026
