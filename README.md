
# YOLOv8 ile Gerçek Zamanlı Çukur Tespiti Projesi

Bu proje, yol yüzeylerindeki çukurları (pothole) resimler ve videolar üzerinden gerçek zamanlı olarak tespit etmek için geliştirilmiş bir bilgisayarla görü (Computer Vision) uygulamasıdır. Proje, son teknoloji nesne tespiti algoritmalarından biri olan **YOLOv8** modelini temel almaktadır.

*(Not: Bu alana kendi oluşturduğunuz bir proje demosunu veya ekran görüntüsünü ekleyebilirsiniz.)*

## 📝 Proje Açıklaması

Yol altyapısının bakımı ve güvenliği, şehir yönetimi ve otonom sürüş teknolojileri için kritik bir öneme sahiptir. Bu proje, araç kameraları veya drone'lar tarafından çekilen görüntülerdeki yol kusurlarını otomatik olarak tespit ederek bakım süreçlerini hızlandırmayı ve sürüş güvenliğini artırmayı hedefler. Özel olarak eğitilmiş bir YOLOv8 modeli kullanarak, çukurların konumları yüksek doğruluk ve hızla belirlenir.

## ✨ Temel Özellikler

  - **Gerçek Zamanlı Tespit:** Videolar üzerinde saniyede yüksek kare hızı (FPS) ile akıcı bir şekilde çukur tespiti yapabilir.
  - **Yüksek Doğruluk:** Özel bir veri seti ile eğitilmiş model, farklı ışık ve yol koşullarında bile başarılı sonuçlar vermektedir.
  - **Esnek Kullanım:** Hem statik resimler (.jpg, .png) hem de video dosyaları (.mp4, .avi) üzerinde çalışabilir.
  - **Kolayca Genişletilebilir:** Kod yapısı, yeni veri setleri ile modelin yeniden eğitilmesine veya farklı nesnelerin (çatlak, rögar kapağı vb.) tespitine olanak tanır.

## 🛠️ Kullanılan Teknolojiler

  - **Python 3.12**
  - **YOLOv8 (Ultralytics)**: Nesne tespiti için kullanılan ana model.
  - **PyTorch**: Modelin eğitimi ve çalıştırılması için kullanılan derin öğrenme kütüphanesi.
  - **OpenCV-Python**: Görüntü ve video işleme, sonuçların ekranda gösterilmesi için kullanılır.
  - **Roboflow**: (İsteğe bağlı) Veri setini etiketlemek, hazırlamak ve yönetmek için kullanılmıştır.

## 🚀 Kurulum ve Başlangıç

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

**1. Projeyi Klonlayın:**

```bash
git clone https://github.com/kullanici-adiniz/pothole-detection-project.git
cd pothole-detection-project
```

**2. Sanal Ortam Oluşturun ve Aktif Edin (Önerilir):**

```bash
# Sanal ortam oluşturma
python -m venv venv

# Sanal ortamı aktif etme (Windows)
.\venv\Scripts\activate
```

**3. Gerekli Kütüphaneleri Yükleyin:**
Proje için gerekli tüm kütüphaneler `requirements.txt` dosyasında listelenmiştir.

```bash
pip install -r requirements.txt
```

*Not: Eğer NVIDIA ekran kartınız (GPU) varsa, PyTorch'un CUDA destekli versiyonunu kurarak eğitim ve tahmin süreçlerini onlarca kat hızlandırabilirsiniz. Bu komut bunu sizin için yapacaktır:*

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**4. Veri Setini ve Eğitilmiş Modeli Hazırlayın:**

  - **Veri Seti:** Bu proje [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection) veri seti ile eğitilmiştir. `prepare_dataset.py` script'i, bu ham veri setini YOLO formatına dönüştürmek için kullanılmıştır.
  - **Eğitilmiş Model:** Projeyi test etmek için eğitilmiş bir model (`best.pt`) gereklidir. Kendi modelinizi eğiterek bu dosyayı elde edebilirsiniz. Büyük dosya boyutları nedeniyle modeller genellikle GitHub'a yüklenmez.

## 📖 Kullanım

Projenin iki ana kullanım amacı vardır: Modeli eğitmek ve eğitilmiş bir modelle tahmin yapmak.

### 1\. Modeli Eğitme (`train.py`)

Kendi veri setinizle modeli sıfırdan eğitmek veya mevcut bir modeli daha da geliştirmek (fine-tuning) için:

1.  Veri setinizin `data.yaml` dosyasında belirtilen formatta ve yolda olduğundan emin olun.
2.  `train.py` dosyasını isteğe göre (epoch sayısı, batch boyutu vb.) düzenleyin.
3.  Aşağıdaki komutu çalıştırın:
    ```bash
    python train.py
    ```
    Eğitim tamamlandığında, en iyi modeliniz (`best.pt`) ve sonuç grafikleri `runs/detect/train` gibi bir klasöre kaydedilecektir.

### 2\. Tahmin Yapma (`predict.py`)

Eğittiğiniz modelle yeni bir resim veya videoda çukur tespiti yapmak için:

1.  Test etmek istediğiniz video veya resim dosyasını `input_media` klasörüne atın.
2.  `predict.py` dosyasını açın ve `MODEL_PATH` ile `MEDIA_PATH` değişkenlerini kendi dosya yollarınıza göre güncelleyin.
    ```python
    # Örnek:
    MODEL_PATH = 'training_results_v2/weights/best.pt'
    MEDIA_PATH = 'input_media/test_video.mp4'
    ```
3.  Aşağıdaki komutu çalıştırın:
    ```bash
    python predict.py
    ```
    İşlem başladığında, sonuçları gösteren bir pencere açılacak ve işlenmiş video/resim `output_media` veya `runs/detect/predict` klasörüne kaydedilecektir.

## 📁 Proje Yapısı

```
pothole_detection_project/
│
├── dataset/            # İşlenmiş, eğitime hazır veri seti
│   ├── images/
│   └── labels/
│
├── input_media/        # Test edilecek resim ve videolar
├── output_media/       # Tahmin sonuçlarının kaydedildiği klasör
│
├── training_results_v2/ # Eğitim sonuçlarının (model ağırlıkları, grafikler) kaydedildiği klasör
│   └── weights/
│       └── best.pt     # En iyi eğitilmiş model
│
├── data.yaml           # Veri setinin yapılandırma dosyası
├── predict.py          # Tahmin script'i
├── train.py            # Eğitim script'i
├── prepare_dataset.py  # (Opsiyonel) Ham veriyi işlemek için script
├── requirements.txt    # Gerekli Python kütüphaneleri
└── README.md           # Bu dosya
```

## 🔮 Gelecekteki Geliştirmeler

  - [ ] Daha çeşitli hava koşullarında (yağmur, gece, kar) çekilmiş verilerle modelin doğruluğunu artırmak.
  - [ ] Tespit edilen çukurların GPS verileriyle entegre edilerek bir harita üzerinde işaretlenmesi.
  - [ ] Flask veya FastAPI kullanarak modelin bir web arayüzü üzerinden hizmet vermesi.
  - [ ] Modelin mobil cihazlarda çalışabilmesi için optimize edilmesi (örn: TensorFlow Lite, ONNX).

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

```
```