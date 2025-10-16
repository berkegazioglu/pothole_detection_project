# Gerekli kütüphaneleri dahil ediyoruz.
from ultralytics import YOLO
import torch

def run_training(model_name='yolov8s.pt', data_config='data.yaml', epochs=100, batch_size=4, project_dir='outputs/training_runs', experiment_name='training_run'):
    """
    YOLO modelini eğitmek için ana fonksiyon.
    Tüm ayarlar, daha esnek bir yapı için parametre olarak dışarıdan alınır.
    """
    # GPU'nun (CUDA) kullanılabilirliğini kontrol eder. Varsa GPU, yoksa CPU kullanır.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Eğitim için kullanılacak cihaz: {device}")

    # Belirtilen model adıyla (örn: 'yolov8s.pt' veya devam edilecek bir modelin yolu) YOLO modelini yüklüyoruz.
    model = YOLO(model_name)
    print(f"'{model_name}' modeli ile eğitime başlanıyor...")
    print(f"Kullanılan veri seti yapılandırması: '{data_config}'")

    # Modeli, dışarıdan gelen parametrelerle eğitmeye başlıyoruz.
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        project=project_dir, # Sonuçların kaydedileceği ana klasör
        name=experiment_name  # Bu eğitim denemesine özel alt klasör adı
    )

    print("Eğitim başarıyla tamamlandı!")
    print(f"Model ve sonuçlar şu klasöre kaydedildi: {results.save_dir}")

# Bu blok, 'python train.py' komutuyla script'in doğrudan çalıştırılmasına
# hala izin verir. Hızlı testler için kullanışlıdır.
if __name__ == '__main__':
    print("Bu script doğrudan çalıştırıldı. Varsayılan ayarlarla bir test eğitimi başlatılıyor...")
    run_training(
        model_name='yolov8s.pt',
        data_config='data.yaml',
        epochs=10, # Test için daha düşük bir epoch sayısı
        name='direct_training_test'
    )