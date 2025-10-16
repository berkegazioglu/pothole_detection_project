# YOLOv8 ile Gerçek Zamanlı Çukur Tespiti Projesi

Bu proje, yol yüzeylerindeki çukurları (pothole) resimler ve videolar üzerinden gerçek zamanlı olarak tespit etmek için geliştirilmiş bir bilgisayarla görü (Computer Vision) uygulamasıdır. Proje, son teknoloji nesne tespiti algoritmalarından biri olan **YOLOv8** modelini temel almaktadır.


## 📝 Proje Açıklaması

Yol altyapısının bakımı ve güvenliği, şehir yönetimi ve otonom sürüş teknolojileri için kritik bir öneme sahiptir. Bu proje, araç kameraları veya drone'lar tarafından çekilen görüntülerdeki yol kusurlarını otomatik olarak tespit ederek bakım süreçlerini hızlandırmayı ve sürüş güvenliğini artırmayı hedefler. Kaggle ve Roboflow gibi kaynaklardan toplanan çeşitli veri setleriyle özel olarak eğitilmiş bir YOLOv8 modeli kullanarak, çukurların konumları yüksek doğruluk ve hızla belirlenir.

## ✨ Temel Özellikler

- **Gerçek Zamanlı Tespit:** Videolar üzerinde saniyede yüksek kare hızı (FPS) ile akıcı bir şekilde çukur tespiti yapabilir.
- **Yüksek Doğruluk:** Özel bir veri seti ile eğitilmiş model, farklı ışık ve yol koşullarında bile başarılı sonuçlar vermektedir.
- **Modüler ve Yönetilebilir Kod:** `main.py` script'i üzerinden tüm eğitim ve tahmin süreçleri komut satırı argümanları ile kontrol edilebilir.
- **Esnek Kullanım:** Hem statik resimler (.jpg, .png) hem de video dosyaları (.mp4, .avi) üzerinde çalışabilir.
- **Düzenli Proje Yapısı:** Veri setleri, çıktılar ve kodlar, en iyi pratiklere uygun olarak ayrı klasörlerde organize edilmiştir.

## 🛠️ Kullanılan Teknolojiler

- **Python 3.11 (64-bit)**: Projenin çalışması için gerekli olan ve en güncel kütüphanelerle uyumluluğu test edilmiş Python sürümü.
- **YOLOv8 (Ultralytics)**: Nesne tespiti için kullanılan ana model.
- **PyTorch (CUDA Destekli)**: Modelin GPU üzerinde eğitimi ve çalıştırılması için kullanılan derin öğrenme kütüphanesi.
- **OpenCV-Python**: Görüntü ve video işleme, sonuçların ekranda gösterilmesi için kullanılır.
- **Kaggle & Roboflow**: Veri setlerinin temin edildiği ve etiketlendiği platformlar.

## 🚀 Kurulum ve Başlangıç

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla izleyebilirsiniz.

**1. Projeyi Klonlayın:**
```bash
git clone [https://github.com/kullanici-adiniz/pothole-detection-project.git](https://github.com/kullanici-adiniz/pothole-detection-project.git)
cd pothole-detection-project
````

**2. Sanal Ortam Oluşturun (Python 3.11 ile):**
Bu proje, `ultralytics` kütüphanesinin güncel sürümleriyle tam uyumluluk için **Python 3.11 (64-bit)** gerektirir.

```bash
# Python 3.11'i kullanarak 'venv' adında bir sanal ortam oluşturun
py -3.11 -m venv venv
```

**3. Sanal Ortamı Aktif Edin:**

```bash
# Windows için
.\venv\Scripts\activate
```

**4. Gerekli Kütüphaneleri Doğru Sırada Yükleyin:**
Bu sıra, olası versiyon çakışmalarını önlemek için önemlidir.

```bash
# a) Önce pip'i güncelleyin
python.exe -m pip install --upgrade pip

# b) GPU destekli PyTorch'u kurun (En Önemli Adım)
python.exe -m pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# c) Geri kalan tüm kütüphaneleri requirements.txt'den kurun
python.exe -m pip install -r requirements.txt
```

**5. Veri Setini Hazırlayın:**

  - Projenin eğitilebilmesi için `data/processed/` klasörünün içinde `images` ve `labels` klasörleri bulunmalıdır.
  - `data.yaml` dosyası, bu klasörlerin yolunu doğru bir şekilde göstermelidir.

## 📖 Kullanım

Tüm işlemler, projenin ana klasöründeki `main.py` script'i üzerinden komut satırı aracılığıyla yönetilir.

### 1\. Modeli Eğitme (`train`)

Yeni bir model eğitmek veya mevcut bir eğitimi devam ettirmek için `train` komutunu kullanın.

**Örnek 1: YOLOv8s ile Sıfırdan 100 Epoch'luk Eğitim Başlatma**

```bash
python main.py train --model yolov8s.pt --epochs 100 --name yolov8_run_1
```

*Bu komut, sonuçları `outputs/training_runs/yolov8_run_1` klasörüne kaydeder.*

**Örnek 2: Bir Önceki Eğitimden Devam Etme (Fine-Tuning)**

```bash
python main.py train --model outputs/training_runs/yolov8_run_1/weights/best.pt --epochs 50 --name yolov8_run_2
```

*Bu komut, en iyi modelden devam eder ve sonuçları `outputs/training_runs/yolov8_run_2` klasörüne kaydeder.*

### 2\. Tahmin Yapma (`predict`)

Eğittiğiniz bir modelle yeni bir resim veya videoda çukur tespiti yapmak için `predict` komutunu kullanın.

**Örnek:**

```bash
python main.py predict --model outputs/training_runs/yolov8_run_1/weights/best.pt --source input_media/bozuk_yollar.mp4 --name video_test_sonucu
```

*Bu komut, videoyu işler, sonuçları ekranda gösterir ve işlenmiş videoyu `outputs/predictions/video_test_sonucu` klasörüne kaydeder.*

## 📁 Proje Yapısı

```
pothole_detection_project/
│
├── data/
│   ├── processed/          <-- İşlenmiş, eğitime hazır veri seti
│   │   ├── images/
│   │   └── labels/
│   └── raw/                <-- Orijinal, dokunulmamış veri setleri (yedek)
│
├── input_media/            <-- Test edilecek resim ve videolar
│
├── outputs/                <-- Tüm eğitim ve tahmin çıktıları
│   ├── predictions/
│   └── training_runs/
│
├── venv/                   <-- Projeye özel sanal ortam
│
├── .gitignore              <-- Git'in yok sayacağı dosyalar
├── data.yaml               <-- Ana veri seti yapılandırma dosyası
├── main.py                 <-- Projenin ana kontrol script'i
├── predict.py              <-- Tahmin fonksiyonunu içeren modül
├── prepare_dataset.py      <-- Ham XML verilerini işlemek için araç
├── README.md               <-- Bu dosya
├── requirements.txt        <-- Gerekli Python kütüphaneleri
└── train.py                <-- Eğitim fonksiyonunu içeren modül
```

## 🔮 Gelecekteki Geliştirmeler

  - [ ] **Sürekli Öğrenme Döngüsü:** Modelin hiç görmediği videolardaki hatalı veya eksik tahminlerini etiketleyerek veri setini sürekli zenginleştirmek ve modeli periyodik olarak yeniden eğitmek.
  - [ ] **GPS Entegrasyonu:** Tespit edilen çukurların GPS verileriyle eşleştirilerek bir harita üzerinde görselleştirilmesi.
  - [ ] **Web Arayüzü:** Flask veya FastAPI kullanarak, kullanıcıların video yükleyip sonuçları bir web sitesi üzerinden görebileceği bir servis oluşturmak.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

```
```