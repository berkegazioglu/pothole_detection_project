# Gerekli kütüphaneleri projeye dahil ediyoruz.
from ultralytics import YOLO
import cv2  # OpenCV kütüphanesi, resim ve videoları göstermek için kullanılır.


def run_prediction(model_path, source_path, conf_threshold=0.25, project_dir='outputs/predictions',
                   experiment_name='prediction_run'):
    """
    Eğitilmiş bir YOLO modeli ile tahmin yapmak için ana fonksiyon.
    Tüm ayarlar, daha esnek bir yapı için parametre olarak dışarıdan alınır.
    """
    # Medya dosyasının uzantısını kontrol ederek resim mi video mu olduğunu anlıyoruz.
    is_video = source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    try:
        # Eğittiğimiz en akıllı modelimizi yüklüyoruz.
        model = YOLO(model_path)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Model yüklenemedi! '{model_path}' yolunun doğru olduğundan emin olun.")
        print(e)
        return

    print(f"'{source_path}' üzerinde tahmin işlemi başlatılıyor...")

    # Modeli, dışarıdan gelen parametrelerle tahmin için çalıştırıyoruz.
    results = model.predict(
        source=source_path,
        show=True,  # Sonuçları ekranda anlık olarak bir pencerede göster.
        save=True,  # İşlenmiş video/resmi yeni bir dosya olarak kaydet.
        stream=is_video,  # Videolar için bellek kullanımını optimize et.
        conf=conf_threshold,  # Güven eşiği: Sadece bu orandan daha emin olduğu tespitleri göster.
        project=project_dir,  # Tahmin sonuçlarının kaydedileceği ana klasör.
        name=experiment_name  # Bu tahmin işlemine özel alt klasör adı.
    )

    # `stream=True` kullandığımız için, video karelerinin işlenmesini beklemek için bir döngü gerekir.
    if is_video:
        for _ in results:
            pass  # Her kare işlendiğinde bu döngü çalışır.

    print("Tahmin işlemi tamamlandı!")
    print(f"Sonuçlar, '{project_dir}/{experiment_name}' klasörüne kaydedildi.")


# Bu blok, 'python predict.py' komutuyla script'in doğrudan çalıştırılmasına izin verir.
# Hızlı testler için kullanışlıdır.
if __name__ == '__main__':
    print("Bu script doğrudan çalıştırıldı. Varsayılan yollarla bir test tahmini başlatılıyor...")
    # DİKKAT: Doğrudan çalıştırmadan önce aşağıdaki yolların doğru olduğundan emin olun!
    DEFAULT_MODEL_PATH = 'outputs/training_runs/yolov8_ilk_egitim/weights/best.pt'
    DEFAULT_SOURCE_PATH = 'input_media/bozuk_yollar.mp4'

    if not os.path.exists(DEFAULT_MODEL_PATH):
        print(f"UYARI: Varsayılan model yolu '{DEFAULT_MODEL_PATH}' bulunamadı. Lütfen yolu güncelleyin.")
    elif not os.path.exists(DEFAULT_SOURCE_PATH):
        print(f"UYARI: Varsayılan medya yolu '{DEFAULT_SOURCE_PATH}' bulunamadı. Lütfen yolu güncelleyin.")
    else:
        run_prediction(
            model_path=DEFAULT_MODEL_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            experiment_name='direct_prediction_test'
        )