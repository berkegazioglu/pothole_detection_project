import argparse
from train import run_training
from predict import run_prediction

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 ile Çukur Tespiti Projesi Ana Kontrol Script'i")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Çalıştırılacak komut: train veya predict')

    # --- 'train' Komutu İçin Ayarlar (YOLOv8'e Güncellendi) ---
    parser_train = subparsers.add_parser('train', help='Yeni bir YOLO modeli eğitir.')
    parser_train.add_argument('--model', type=str, default='yolov8s.pt', help="Başlangıç modeli (örn: 'yolov8s.pt').") # <-- DEĞİŞİKLİK
    parser_train.add_argument('--data', type=str, default='data.yaml', help='Veri seti yapılandırma dosyası.')
    parser_train.add_argument('--epochs', type=int, default=100, help='Eğitim epoch sayısı.')
    parser_train.add_argument('--batch', type=int, default=4, help='Batch boyutu.')
    parser_train.add_argument('--name', type=str, default='yolov8_run', help='Eğitim sonuç klasörünün adı.') # <-- DEĞİŞİKLİK

    # --- 'predict' Komutu İçin Ayarlar ---
    parser_predict = subparsers.add_parser('predict', help='Eğitilmiş bir modelle tahmin yapar.')
    parser_predict.add_argument('--model', type=str, required=True, help='Kullanılacak modelin yolu (.pt dosyası).')
    parser_predict.add_argument('--source', type=str, required=True, help='Tahmin yapılacak resim/video yolu.')
    parser_predict.add_argument('--conf', type=float, default=0.25, help='Minimum güven eşiği.')
    parser_predict.add_argument('--name', type=str, default='prediction_run', help='Tahmin sonuç klasörünün adı.')

    args = parser.parse_args()

    if args.command == 'train':
        print("--- EĞİTİM MODU (YOLOv8) ---")
        run_training(
            model_name=args.model,
            data_config=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            project_dir='outputs/training_runs',
            experiment_name=args.name
        )
    elif args.command == 'predict':
        print("--- TAHMİN MODU ---")
        run_prediction(
            model_path=args.model,
            source_path=args.source,
            conf_threshold=args.conf,
            project_dir='outputs/predictions',
            experiment_name=args.name
        )

if __name__ == '__main__':
    main()