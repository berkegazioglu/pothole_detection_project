# =============================================================================
# Gerekli Kütüphaneler
# =============================================================================
import os
import random
import shutil
import xml.etree.ElementTree as ET

# =============================================================================
# KULLANICI AYARLARI (Konfigürasyon)
# =============================================================================
# Bu bölüm, ham bir XML veri setini YOLO formatına dönüştürmek için
# script'in davranışını kontrol eder. Yeni bir ham veri seti işleyeceğiniz
# zaman bu ayarları düzenlemeniz gerekebilir.

# Ham resim dosyalarının bulunduğu kaynak klasörün yolu.
SOURCE_IMAGES_DIR = "data/raw/images"

# Ham XML etiket dosyalarının bulunduğu kaynak klasörün yolu.
SOURCE_ANNOTATIONS_DIR = "data/raw/annotations"

# İşlenmiş ve eğitime hazır verilerin kaydedileceği hedef klasör.
DEST_DIR = "data/processed"

# Veri setinin yüzde kaçının doğrulama (validation) için ayrılacağı.
VALID_SPLIT = 0.2

# Tespit edilecek nesnelerin sınıfları.
CLASSES = ["pothole"]


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def create_dirs(destination_dir):
    """
    YOLO'nun beklediği standart klasör yapısını (images/train, labels/valid vb.) oluşturur.
    Eğer klasörler zaten mevcutsa, hata vermeden devam eder.
    """
    print(f"'{destination_dir}' içinde hedef klasör yapısı oluşturuluyor veya kontrol ediliyor...")
    for sub_dir in ["images/train", "images/valid", "labels/train", "labels/valid"]:
        os.makedirs(os.path.join(destination_dir, sub_dir), exist_ok=True)


def convert_xml_to_yolo(xml_path, img_w, img_h):
    """
    Tek bir XML dosyasındaki koordinatları okur ve YOLO'nun .txt formatına dönüştürür.
    Koordinatları, resim boyutlarına göre 0 ile 1 arasında oranlar.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        yolo_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASSES:
                continue
            class_id = CLASSES.index(class_name)
            bndbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = [float(bndbox.find(tag).text) for tag in ['xmin', 'ymin', 'xmax', 'ymax']]

            x_center = (xmin + xmax) / 2.0 / img_w
            y_center = (ymin + ymax) / 2.0 / img_h
            width = (xmax - xmin) / img_w
            height = (ymax - ymin) / img_h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        return "\n".join(yolo_lines)
    except Exception as e:
        print(f"HATA: XML dosyası '{xml_path}' okunurken sorun oluştu: {e}")
        return None


# =============================================================================
# ANA İŞLEM FONKSİYONU
# =============================================================================

def process_dataset():
    """Tüm veri işleme sürecini baştan sona yönetir."""
    create_dirs(DEST_DIR)

    if not os.path.isdir(SOURCE_ANNOTATIONS_DIR):
        print(f"HATA: Kaynak etiket klasörü bulunamadı: '{SOURCE_ANNOTATIONS_DIR}'")
        return

    xml_files = [f for f in os.listdir(SOURCE_ANNOTATIONS_DIR) if f.endswith('.xml')]

    valid_files = []
    for xml_file in xml_files:
        base_name = os.path.splitext(xml_file)[0]
        # Hem .png hem de .jpg uzantılı resimleri kontrol eder.
        for ext in ['.png', '.jpg', '.jpeg']:
            img_path = os.path.join(SOURCE_IMAGES_DIR, base_name + ext)
            if os.path.exists(img_path):
                valid_files.append((xml_file, ext))
                break
        else:  # Döngü 'break' olmadan biterse, yani resim bulunamazsa...
            print(f"UYARI: '{xml_file}' etiketine ait bir resim bulunamadı. Bu etiket atlanacak.")

    random.shuffle(valid_files)
    split_idx = int(len(valid_files) * (1 - VALID_SPLIT))
    train_set = valid_files[:split_idx]
    valid_set = valid_files[split_idx:]

    print(
        f"\n{len(valid_files)} geçerli veri bulundu. {len(train_set)} train, {len(valid_set)} validation olarak ayrılıyor.")

    def copy_and_convert(file_set, set_type):
        for xml_name, img_ext in file_set:
            base_name = os.path.splitext(xml_name)[0]
            src_xml_path = os.path.join(SOURCE_ANNOTATIONS_DIR, xml_name)
            src_img_path = os.path.join(SOURCE_IMAGES_DIR, base_name + img_ext)

            dest_label_path = os.path.join(DEST_DIR, f"labels/{set_type}", base_name + ".txt")
            dest_img_path = os.path.join(DEST_DIR, f"images/{set_type}", base_name + img_ext)

            try:
                tree = ET.parse(src_xml_path)
                size = tree.getroot().find('size')
                img_w, img_h = int(size.find('width').text), int(size.find('height').text)

                yolo_data = convert_xml_to_yolo(src_xml_path, img_w, img_h)
                if yolo_data is not None:
                    with open(dest_label_path, 'w') as f:
                        f.write(yolo_data)
                    shutil.copy(src_img_path, dest_img_path)
            except Exception as e:
                print(f"HATA: '{xml_name}' işlenirken beklenmedik bir hata oluştu: {e}")

    print("\nTrain verileri işleniyor...")
    copy_and_convert(train_set, "train")
    print("Validation verileri işleniyor...")
    copy_and_convert(valid_set, "valid")
    print("\nİşlem tamamlandı! Verileriniz artık 'data/processed' klasöründe eğitime hazır.")


# =============================================================================
# SCRIPT'İN BAŞLANGIÇ NOKTASI
# =============================================================================
if __name__ == '__main__':
    # GÜVENLİK NOTU: Bu script, DEST_DIR içindeki mevcut dosyaların üzerine yazabilir.
    # Çalıştırmak için aşağıdaki satırın başındaki '#' işaretini kaldırın ve
    # terminalde `python prepare_dataset.py` komutunu çalıştırın.

    # process_dataset()

    print("Bu script, 'data/raw' klasöründeki ham XML verilerini 'data/processed' klasörüne işlemek içindir.")
    print("Kullanmak için dosyanın en altındaki 'process_dataset()' satırının yorumunu kaldırmanız gerekir.")