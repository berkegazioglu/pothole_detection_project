# Gerekli kütüphaneleri projeye dahil ediyoruz.
from ultralytics import YOLO
import cv2  # OpenCV kütüphanesi, resim ve videoları göstermek için kullanılır.

def main():
    # --- KULLANICI AYARLARI ---

    # 1. EĞİTİLMİŞ MODELİN YOLU
    # Bu yolu, en son yaptığın eğitimin sonuç klasöründeki 'best.pt' dosyasıyla güncellemelisin.
    # Genellikle 'runs/detect/pothole_yolov8s_rtx30502/weights/best.pt' gibi bir yoldur.
    MODEL_PATH = 'training_results_v2/weights/best.pt'

    # 2. TEST EDİLECEK MEDYA DOSYASININ YOLU
    # 'input_media' klasörüne koyduğun bir resim veya videonun adını buraya yaz.
    MEDIA_PATH = 'input_media/bozuk_yollar.mp4'  # Örnek: 'input_media/test_image.png'

    # --- AYARLARI DEĞİŞTİRMEYİ BİTİRİN ---

    # Medya dosyasının uzantısını kontrol ederek resim mi video mu olduğunu anlıyoruz.
    is_video = MEDIA_PATH.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    try:
        # Eğittiğimiz en akıllı modelimizi yüklüyoruz.
        model = YOLO(MODEL_PATH)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Model yüklenemedi! '{MODEL_PATH}' yolunun doğru olduğundan emin olun.")
        print(e)
        return

    print(f"'{MEDIA_PATH}' üzerinde tahmin işlemi başlatılıyor...")

    # Modeli kullanarak tahmin işlemini başlatıyoruz.
    # show=True: Sonuçları ekranda anlık olarak bir pencerede gösterir.
    # save=True: Sonuçları (işlenmiş video veya resim) yeni bir dosya olarak kaydeder.
    # stream=is_video: Videolar için bellek kullanımını optimize eder.
    #results = model.predict(source=MEDIA_PATH, show=True, save=True, stream=is_video)
    results = model.predict(
        source=MEDIA_PATH,
        show=True,
        save=True,
        stream=is_video,
        conf=0.25,  # Güven eşiği
        project='./output_media',  # Ana kayıt klasörü olarak 'output_media'yı belirle
        name='prediction_results'  # 'output_media' içinde bu isimde bir alt klasör oluştur
    )

    # `stream=True` kullandığımız için, video karelerini işlemek için bir döngü gerekir.
    # `show=True` zaten gösterimi yaptığı için bu döngünün içi boş kalabilir,
    # ama programın video bitene kadar kapanmamasını sağlar.
    if is_video:
        for result in results:
            pass  # Her kare işlendiğinde bu döngü çalışır.

    print("Tahmin işlemi tamamlandı!")
    print("Sonuçlar, projenizin ana klasöründeki 'runs/detect/predict...' adlı yeni bir klasöre kaydedildi.")


if __name__ == '__main__':
    main()