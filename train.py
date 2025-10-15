from ultralytics import YOLO
import torch


def main():
    # GPU'nun kullanılabilirliğini kontrol etmeye devam ediyoruz.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim için kullanılacak cihaz: {device}")

    # --- DEĞİŞİKLİK 1: ÖĞRENİLMİŞ MODELİ YÜKLEME ---
    # Sıfırdan başlamak yerine, bir önceki eğitimin en iyi sonucunu yüklüyoruz.
    # Bu sayede model, kaldığı yerden daha akıllı bir şekilde öğrenmeye devam eder.
    model_path = 'pothole_yolov8s_rtx30502/weights/best.pt'
    model = YOLO(model_path)
    print(f"'{model_path}' modelinden eğitime devam ediliyor...")

    # Modeli kendi veri setimizle eğitmeye devam ediyoruz
    print("Model eğitimi başlıyor...")
    results = model.train(
        data='data.yaml',

        # Pro İpucu: Devam eğitimlerinde epoch sayısını daha düşük tutmak iyi bir fikirdir.
        # Model zaten temel bilgileri öğrendiği için 50 yerine 25-30 tur daha yeterli olabilir.
        epochs=25,

        imgsz=640,
        batch=4,  # Senin RTX 3050'n için güvenli batch boyutu
        device=device,

        # --- DEĞİŞİKLİK 2: KAYIT YERİNİ AYARLAMA ---
        # 'project' parametresi ana kayıt klasörünü belirler. '.' mevcut klasör demektir.
        # 'name' ise bu klasör içinde oluşacak alt klasörün adıdır.
        project='.',  # Sonuçları doğrudan ana proje klasörüne kaydet
        name='training_results_v2'  # Yeni eğitimin sonuçları bu klasöre kaydedilecek
    )
    print("Eğitim başarıyla tamamlandı!")
    print(f"Model ve sonuçlar şu klasöre kaydedildi: {results.save_dir}")


if __name__ == '__main__':
    main()