# NOEL PERAKENDE SATIS ANALIZI VE MAKINE OGRENMESI
## Python ile Veri Bilimine Giris - Final Projesi

---

# SLAYT 1: PROJE TANITIMI

## Proje Adi
**Noel Perakende Satis Analizi ve Teslimat Tahmini**

## Amac
- E-ticaret lojistik verilerini analiz etmek
- Teslimat performansini etkileyen faktorleri bulmak
- Makine ogrenmesi ile teslimat durumunu tahmin etmek

## Veri Seti
- **Kaynak:** Christmas Retail Sales - Shipping & Delivery Dataset
- **Icerik:** 7 farkli tablo (OrderHeader, OrderLine, Product, Promotion, Fulfillment, Returns, Calendar)
- **Format:** Excel dosyasi (.xlsx)

---

# SLAYT 2: KULLANILAN KUTUPHANELER

```python
import pandas as pd          # Veri isleme ve analiz
import numpy as np           # Sayisal hesaplamalar
import matplotlib.pyplot as plt  # Gorsellestirme
import seaborn as sns        # Istatistiksel gorsellestirme
from scipy import stats      # Istatistiksel testler
from sklearn.model_selection import train_test_split  # Veri bolme
from sklearn.preprocessing import LabelEncoder, StandardScaler  # On isleme
from sklearn.linear_model import LogisticRegression  # Lojistik Regresyon
from sklearn.tree import DecisionTreeClassifier      # Karar Agaci
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.neighbors import KNeighborsClassifier   # KNN
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Neden Bu Kutuphaneler?**
- Pandas: Veri cercevesi islemleri (DataFrame)
- NumPy: Hizli matematiksel islemler
- Matplotlib/Seaborn: Profesyonel grafikler
- Scipy: Bilimsel hesaplamalar ve testler
- Sklearn: Makine ogrenmesi algoritmalari

---

# SLAYT 3: VERI YUKLEME

## Excel'den Veri Okuma
```python
excel_dosyasi = pd.ExcelFile('Christmas_Retail_Sales_and_Marketing_Analysis_Dataset.xlsx')
print("Tablolar:", excel_dosyasi.sheet_names)
```

## 7 Tablo Yuklendi:
| Tablo | Aciklama | Satir Sayisi |
|-------|----------|--------------|
| OrderHeader | Siparis basliklari | ~10,000+ |
| OrderLine | Siparis detaylari | ~50,000+ |
| Product | Urun bilgileri | ~500+ |
| Promotion | Promosyon bilgileri | ~100+ |
| Fulfillment | Teslimat bilgileri | ~10,000+ |
| Returns | Iade bilgileri | ~2,000+ |
| Calendar | Tarih bilgileri | 365 |

---

# SLAYT 4: TABLO YAPILARI

## Her Tablonun Incelenmesi
```python
for tablo_adi in excel_dosyasi.sheet_names:
    df_temp = pd.read_excel(excel_dosyasi, sheet_name=tablo_adi)
    print(f"{tablo_adi}: {df_temp.shape[0]} satir, {df_temp.shape[1]} sutun")
    print(f"Sutunlar: {df_temp.columns.tolist()}")
```

## Onemli Sutunlar:
- **OrderID:** Siparis numarasi (birincil anahtar)
- **ProductID:** Urun numarasi
- **CustomerID:** Musteri numarasi
- **DeliveryStatus:** Teslimat durumu (OnTime/Late)
- **ShipCost$:** Kargo maliyeti

---

# SLAYT 5: VERI BIRLESTIRME (MERGE)

## Neden Birlestirme Gerekli?
Veriler 7 farkli tabloda dagilmis durumda. Analiz icin tek bir tablo olusturmaliyiz.

## Merge Islemleri
```python
# Siparis detayi + Siparis basligi
df = pd.merge(order_line, order_header, on='OrderID', how='left')

# + Urun bilgileri
df = pd.merge(df, product, on='ProductID', how='left')

# + Teslimat bilgileri
df = pd.merge(df, fulfillment, on='ShipmentID', how='left')
```

## Merge Tipleri:
- **left:** Sol tablodaki tum satirlari koru
- **inner:** Sadece eslesen satirlari al
- **outer:** Tum satirlari al

**Sonuc:** Tum bilgiler tek DataFrame'de toplandi!

---

# SLAYT 6: EKSIK VERI ANALIZI

## Eksik Verileri Tespit Etme
```python
eksik_veriler = df.isnull().sum()
eksik_yuzde = (df.isnull().sum() / len(df)) * 100
```

## Neden Onemli?
- Eksik veriler model performansini dusurur
- Istatistiksel hesaplamalari etkiler
- Veri kalitesini gosterir

## Cozum Yontemleri:
1. **Silme:** Cok az eksik varsa satiri sil
2. **Doldurma:** Ortalama, medyan veya mod ile doldur
3. **Tahmin:** ML ile eksik degerleri tahmin et

---

# SLAYT 7: TEMEL ISTATISTIKLER

## Merkezi Egilim Olculeri
```python
print("Ortalama:", df['ShipCost$'].mean())
print("Medyan:", df['ShipCost$'].median())
print("Mod:", df['ShipCost$'].mode()[0])
```

## Dagilim Olculeri
```python
print("Standart Sapma:", df['ShipCost$'].std())
print("Varyans:", df['ShipCost$'].var())
print("Minimum:", df['ShipCost$'].min())
print("Maximum:", df['ShipCost$'].max())
```

## describe() Fonksiyonu
```python
df.describe()  # Tum istatistikleri tek seferde gosterir
```

---

# SLAYT 8: KATEGORIK DEGISKEN GRAFIKLERI

## Countplot - Kategori Sayilari
```python
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Category')
plt.title('Kategori Dagilimi')
plt.xticks(rotation=45)
plt.show()
```

## Ne Gorduk?
- En cok satilan kategoriler
- Kategori dengesizlikleri
- Is kararlari icin icerik

## Diger Kategorik Grafikler:
- Teslimat durumu dagilimi
- Odeme tipi dagilimi
- Kargo firmasi dagilimi

---

# SLAYT 9: SAYISAL DEGISKEN GRAFIKLERI

## Histogram - Dagilim
```python
plt.figure(figsize=(10, 6))
plt.hist(df['ShipCost$'], bins=30, edgecolor='black')
plt.title('Kargo Maliyeti Dagilimi')
plt.xlabel('Maliyet ($)')
plt.ylabel('Frekans')
plt.show()
```

## Ne Anlatiyor?
- Verilerin nasil dagildigini gosterir
- Normal dagilim mi, carpik mi?
- Aykiri degerler var mi?

---

# SLAYT 10: BOXPLOT (KUTU GRAFIGI)

## Boxplot Olusturma
```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Category', y='LineRevenue$')
plt.title('Kategorilere Gore Gelir Dagilimi')
plt.xticks(rotation=45)
plt.show()
```

## Boxplot Ne Gosterir?
- **Kutu:** Q1-Q3 arasi (verilerin %50'si)
- **Cizgi:** Medyan
- **Biyiklar:** Min-Max (aykiri haric)
- **Noktalar:** Aykiri degerler

## Aykiri Deger Tespiti
IQR yontemi ile aykiri degerler belirlenir.

---

# SLAYT 11: KORELASYON ANALIZI

## Korelasyon Matrisi
```python
korelasyon = df[sayisal_sutunlar].corr()
```

## Heatmap Gorsellestirme
```python
plt.figure(figsize=(12, 8))
sns.heatmap(korelasyon, annot=True, cmap='coolwarm', center=0)
plt.title('Korelasyon Matrisi')
plt.show()
```

## Korelasyon Yorumu:
- **+1:** Mukemmel pozitif iliski
- **0:** Iliski yok
- **-1:** Mukemmel negatif iliski

## Buldugumuz Iliskiler:
- Miktar ile Gelir arasinda pozitif iliski
- Fiyat ile Maliyet arasinda pozitif iliski

---

# SLAYT 12: GROUPBY VE AGGREGATION

## Gruplandirma Islemleri
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

## Neden Kullanilir?
- Kategorilere gore ozet istatistikler
- Karsilastirmali analizler
- Raporlama icin ozet tablolar

---

# SLAYT 13: NORMALLIK TESTI (SHAPIRO-WILK)

## Hipotez
- **H0:** Veri normal dagilmistir
- **H1:** Veri normal dagilmamistir

## Test Uygulama
```python
from scipy.stats import shapiro

stat, p_value = shapiro(df['ShipCost$'].sample(5000))
print(f"P-deger: {p_value}")

if p_value > 0.05:
    print("Normal dagilim kabul edilir")
else:
    print("Normal dagilim reddedilir")
```

## Neden Onemli?
- Parametrik testler normal dagilim gerektirir
- Model secimini etkiler

---

# SLAYT 14: T-TESTI

## Amac
Iki grubun ortalamalari arasinda anlamli fark var mi?

## Ornek: Zamaninda vs Gecikmeli Teslimat
```python
from scipy.stats import ttest_ind

grup1 = df[df['DeliveryStatus'] == 'OnTime']['ShipCost$']
grup2 = df[df['DeliveryStatus'] == 'Late']['ShipCost$']

t_stat, p_value = ttest_ind(grup1, grup2)
```

## Yorum
- **p < 0.05:** Gruplar arasinda anlamli fark var
- **p >= 0.05:** Anlamli fark yok

---

# SLAYT 15: ANOVA TESTI

## Amac
Ikiden fazla grubun ortalamalari arasinda fark var mi?

## Ornek: Kategorilere Gore Gelir
```python
from scipy.stats import f_oneway

gruplar = [grup['LineRevenue$'] for name, grup in df.groupby('Category')]
f_stat, p_value = f_oneway(*gruplar)
```

## Yorum
- Kategoriler arasinda gelir farki var mi?
- Hangi kategori daha karli?

---

# SLAYT 16: HEDEF DEGISKEN OLUSTURMA

## Siniflandirma Problemi
Teslimat zamaninda mi yoksa gecikmeli mi olacak?

## Hedef Degisken
```python
df['Basarili_Teslimat'] = (df['DeliveryStatus'] == 'OnTime').astype(int)
# OnTime -> 1 (Basarili)
# Late -> 0 (Basarisiz/Gecikmeli)
```

## Dagilim Kontrolu
```python
print(df['Basarili_Teslimat'].value_counts())
print(f"Basari Orani: %{df['Basarili_Teslimat'].mean() * 100:.2f}")
```

---

# SLAYT 17: OZELLIK SECIMI

## Kullanilacak Ozellikler
```python
ozellikler = ['Qty', 'UnitPrice', 'LineRevenue$', 'ShipCost$',
              'Category', 'PaymentType', 'Carrier', 'ServiceLevel']
```

## Neden Bu Ozellikler?
- **Qty:** Siparis miktari teslimat suresini etkileyebilir
- **ShipCost$:** Kargo maliyeti servis kalitesini gosterir
- **Carrier:** Farkli kargo firmalari farkli performans
- **ServiceLevel:** Express vs Standard teslimat

---

# SLAYT 18: LABEL ENCODING

## Kategorik -> Sayisal Donusum
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])
df['PaymentType_encoded'] = le.fit_transform(df['PaymentType'])
df['Carrier_encoded'] = le.fit_transform(df['Carrier'])
df['ServiceLevel_encoded'] = le.fit_transform(df['ServiceLevel'])
```

## Neden Gerekli?
- ML algoritmalari sayisal deger ister
- Kategorik degerleri sayiya ceviriyoruz

## Ornek:
- Electronics -> 0
- Clothing -> 1
- Books -> 2

---

# SLAYT 19: EGITIM-TEST AYIRMA

## Veriyi Bolme
```python
from sklearn.model_selection import train_test_split

X = df[ozellik_sutunlari]  # Ozellikler
y = df['Basarili_Teslimat']  # Hedef

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Parametreler:
- **test_size=0.2:** %20 test, %80 egitim
- **random_state=42:** Tekrarlanabilirlik

## Neden Ayiriyoruz?
- Model gorunmeyen veri uzerinde test edilmeli
- Overfitting'i onlemek icin

---

# SLAYT 20: OLCEKLENDIRME (SCALING)

## StandardScaler
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Formul
```
z = (x - ortalama) / standart_sapma
```

## Neden Gerekli?
- Farkli olceklerdeki degiskenleri esitler
- Ornek: Fiyat (0-1000) vs Miktar (1-10)
- KNN ve Lojistik Regresyon icin kritik

---

# SLAYT 21: LOJISTIK REGRESYON

## Model Olusturma
```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train_scaled, y_train)
```

## Tahmin
```python
y_pred = log_reg.predict(X_test_scaled)
```

## Degerlendirme
```python
from sklearn.metrics import accuracy_score
print(f"Dogruluk: %{accuracy_score(y_test, y_pred) * 100:.2f}")
```

## Ne Zaman Kullanilir?
- Ikili siniflandirma problemleri
- Basit ve yorumlanabilir model istendiginde

---

# SLAYT 22: KARAR AGACI

## Model Olusturma
```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
```

## Avantajlari:
- Yorumlanabilir (if-else kurallari)
- Olceklendirme gerektirmez
- Kategorik ve sayisal veri ile calisir

## Dezavantajlari:
- Overfitting egilimi
- Kucuk degisikliklere hassas

---

# SLAYT 23: RANDOM FOREST

## Model Olusturma
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## Nasil Calisir?
- Birden fazla karar agaci olusturur
- Her agac farkli veri ornegi kullanir
- Sonuclar oylama ile birlestirilir

## Avantajlari:
- Overfitting'e dayanikli
- Ozellik onemliligi gosterir
- Genellikle yuksek performans

---

# SLAYT 24: KNN (K-EN YAKIN KOMSU)

## Model Olusturma
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

## Nasil Calisir?
- Yeni veri noktasi icin en yakin k komsuyu bul
- Komularin cogunlugunun sinifini ata

## K Degeri Secimi:
- Kucuk k: Overfitting riski
- Buyuk k: Underfitting riski
- Genellikle tek sayi secilir (3, 5, 7...)

---

# SLAYT 25: CONFUSION MATRIX

## Karisiklik Matrisi
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

- **TP:** Dogru Pozitif
- **TN:** Dogru Negatif
- **FP:** Yanlis Pozitif (Tip I Hata)
- **FN:** Yanlis Negatif (Tip II Hata)

---

# SLAYT 26: MODEL METRIKLERI

## Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Metrikler:
- **Accuracy:** (TP+TN) / Toplam
- **Precision:** TP / (TP+FP) - Ne kadar kesin?
- **Recall:** TP / (TP+FN) - Ne kadar kapsamli?
- **F1-Score:** Precision ve Recall'un harmonik ortalamasi

---

# SLAYT 27: MODEL KARSILASTIRMASI

## Tum Modellerin Sonuclari

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Lojistik Regresyon | %XX | %XX | %XX | %XX |
| Karar Agaci | %XX | %XX | %XX | %XX |
| Random Forest | %XX | %XX | %XX | %XX |
| KNN | %XX | %XX | %XX | %XX |

## En Iyi Model Secimi:
- En yuksek F1-Score'a sahip model
- Is problemine gore Precision/Recall onceligi

---

# SLAYT 28: CAPRAZ DOGRULAMA

## Cross Validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Ortalama: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")
```

## Neden Onemli?
- Tek train-test bolumune bagimli olmaz
- Modelin genel performansini gosterir
- Overfitting kontrolu saglar

---

# SLAYT 29: OZELLIK ONEMLILIGI

## Feature Importance (Random Forest)
```python
importance = rf.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title('Ozellik Onemliligi')
plt.show()
```

## Yorum:
- Hangi degiskenler tahmini en cok etkiliyor?
- Onemli ozelliklere odaklanma
- Gereksiz ozellikleri cikarma

---

# SLAYT 30: SONUC VE ONERILER

## Proje Ozeti
1. 7 tablodan olusan e-ticaret verisi analiz edildi
2. Veri birlestirme ve temizleme yapildi
3. Kapsamli kesifsel veri analizi gerceklestirildi
4. Istatistiksel testler uygulandi
5. 4 farkli ML modeli karsilastirildi

## Bulgular
- Teslimat performansini etkileyen faktorler belirlendi
- En basarili model: [Model Adi]
- Onemli ozellikler: [Liste]

## Is Onerileri
- Teslimat surelerini iyilestirmek icin [oneri]
- Musteri memnuniyetini artirmak icin [oneri]

---

# SLAYT 31: KAYNAKLAR

## Kullanilan Teknolojiler
- Python 3.x
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn

## Veri Seti
- Christmas Retail Sales - Shipping & Delivery Dataset

## Referanslar
- Ders Notlari: Python ile Veri Bilimine Giris
- Scikit-learn Documentation
- Pandas Documentation

---

# TESEKKURLER!

## Sorular?

**Proje Sahibi:** [Isim]
**Ders:** Python ile Veri Bilimine Giris ve Makine Ogrenmesi
**Tarih:** Ocak 2026
