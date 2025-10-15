# =============================================================================
# Gerekli Kütüphanelerin Projeye Dahil Edilmesi (import)
# =============================================================================
# Bu bölümde, script'in çalışması için gereken standart Python araçlarını (kütüphaneleri) çağırıyoruz.

import os  # İşletim sistemiyle ilgili temel fonksiyonları kullanmamızı sağlar.
           # Örneğin: klasör oluşturma (makedirs), dosya yollarını birleştirme (path.join)

import random  # Rastgele sayılar ve işlemler üretmek için kullanılır.
               # Veri setini karıştırarak modelin öğrenme kalitesini artırmak için kullanacağız.

import shutil  # Gelişmiş dosya işlemleri (kopyalama, taşıma, silme) için kullanılır.
               # 'shutil.copy()' ile dosyaları bir yerden bir yere kopyalayacağız.

import xml.etree.ElementTree as ET  # XML (Genişletilebilir İşaretleme Dili) dosyalarını okumak ve
                                    # içindeki verileri ayrıştırmak (parse) için güçlü bir araçtır.
                                    # Bizim .xml etiketlerimizi okumak için bu kütüphaneyi kullanacağız.

# =============================================================================
# KULLANICI AYARLARI (Konfigürasyon)
# =============================================================================
# Bu bölüm, kodun ana mantığına dokunmadan projenin temel ayarlarını
# kolayca değiştirebilmeniz için tasarlanmıştır.

# --- AYARLARI DEĞİŞTİREBİLİRSİNİZ ---

# Ham resim dosyalarının bulunduğu kaynak klasörün yolu.
# "./images", bu script'in çalıştığı dizindeki "images" adlı klasörü ifade eder.
SOURCE_IMAGES_DIR = "./images"

# Ham XML etiket dosyalarının bulunduğu kaynak klasörün yolu.
SOURCE_ANNOTATIONS_DIR = "./annotations"

# İşlenmiş ve eğitime hazır hale getirilmiş verilerin kaydedileceği hedef klasör.
DEST_DIR = "./dataset/"

# Veri setinin yüzde kaçının doğrulama (validation) için ayrılacağını belirler.
# 0.2, verilerin %20'sinin modelin performansını test etmek için kullanılacağı,
# geri kalan %80'inin ise modeli eğitmek için kullanılacağı anlamına gelir.
VALID_SPLIT = 0.2

# Tespit edilmesini istediğimiz nesnelerin sınıflarını içeren bir liste.
# Bizim projemizde sadece "pothole" olduğu için tek elemanlı bir liste.
# Eğer birden fazla nesne olsaydı, ['pothole', 'crack', 'manhole'] şeklinde olurdu.
CLASSES = ["pothole"]

# --- AYARLARI DEĞİŞTİRMEYİ BİTİRİN ---


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def create_dirs():
    """
    YOLO'nun beklediği standart klasör yapısını oluşturur.
    Eğer klasörler zaten mevcutsa, hata vermeden devam eder.
    """
    print("Gerekli hedef klasörler oluşturuluyor veya kontrol ediliyor...")
    # Oluşturulacak tüm alt klasör yollarını bir liste içinde tanımlıyoruz.
    for sub_dir in ["images/train", "images/valid", "labels/train", "labels/valid"]:
        # os.makedirs, iç içe klasörleri tek seferde oluşturabilir.
        # exist_ok=True parametresi, script her çalıştığında "klasör zaten var" hatası
        # almamızı engeller, bu da script'i tekrar tekrar çalıştırmayı güvenli hale getirir.
        os.makedirs(os.path.join(DEST_DIR, sub_dir), exist_ok=True)


def convert_xml_to_yolo(xml_file_path, image_width, image_height):
    """
    Tek bir XML dosyasındaki koordinatları okur ve YOLO'nun .txt formatına dönüştürür.

    Args:
        xml_file_path (str): Okunacak .xml dosyasının tam yolu.
        image_width (int): Orijinal resmin piksel cinsinden genişliği.
        image_height (int): Orijinal resmin piksel cinsinden yüksekliği.

    Returns:
        str: YOLO formatına dönüştürülmüş, yazılmaya hazır metin.
    """
    # Verilen yoldaki XML dosyasını okur ve içeriğini bir ağaç yapısı olarak belleğe yükler.
    tree = ET.parse(xml_file_path)
    # XML dosyasının en dış etiketini (kökünü) alır.
    root = tree.getroot()
    # Dönüştürülen her nesne için bir satır tutacak olan boş bir liste oluştururuz.
    yolo_lines = []

    # XML dosyasındaki her bir '<object>' etiketini bulur. Her etiket bir çukuru temsil eder.
    for obj in root.findall('object'):
        # Nesnenin adını ('pothole') alır.
        class_name = obj.find('name').text
        # Eğer bu nesnenin adı bizim aradığımız sınıflar listesinde değilse, bu nesneyi atlarız.
        if class_name not in CLASSES:
            continue

        # Sınıfın, CLASSES listesindeki sıra numarasını (indeksini) alırız. 'pothole' için bu 0 olacaktır.
        # YOLO formatının ilk değeri bu class_id'dir.
        class_id = CLASSES.index(class_name)

        # '<bndbox>' etiketini buluruz. Bu, sınırlayıcı kutunun (bounding box) koordinatlarını içerir.
        bndbox = obj.find('bndbox')
        # Piksel cinsinden koordinatları okur ve matematiksel işlem yapabilmek için ondalıklı sayıya (float) çeviririz.
        xmin = float(bndbox.find('xmin').text)  # Kutunun sol üst köşesinin x koordinatı
        ymin = float(bndbox.find('ymin').text)  # Kutunun sol üst köşesinin y koordinatı
        xmax = float(bndbox.find('xmax').text)  # Kutunun sağ alt köşesinin x koordinatı
        ymax = float(bndbox.find('ymax').text)  # Kutunun sağ alt köşesinin y koordinatı

        # --- YOLO FORMATINA DÖNÜŞTÜRME MATEMATİĞİ ---
        # YOLO, mutlak piksel koordinatları yerine, resmin genişliğine ve yüksekliğine
        # oranlanmış, 0 ile 1 arasında değerler kullanır. Bu, modelin farklı boyutlardaki
        # resimlerle daha kolay çalışmasını sağlar.

        # 1. Kutunun merkez noktasının x koordinatını bul ve resmin toplam genişliğine bölerek oranla.
        x_center = (xmin + xmax) / 2.0 / image_width
        # 2. Kutunun merkez noktasının y koordinatını bul ve resmin toplam yüksekliğine bölerek oranla.
        y_center = (ymin + ymax) / 2.0 / image_height
        # 3. Kutunun genişliğini bul ve resmin toplam genişliğine bölerek oranla.
        width = (xmax - xmin) / image_width
        # 4. Kutunun yüksekliğini bul ve resmin toplam yüksekliğine bölerek oranla.
        height = (ymax - ymin) / image_height

        # Elde edilen 5 değeri (sınıf_id, x_merkez, y_merkez, genişlik, yükseklik)
        # standart YOLO formatında bir string (metin) haline getiririz.
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Tüm nesneler işlendikten sonra, her bir satırı alt alta birleştirip tek bir metin olarak döndürürüz.
    return "\n".join(yolo_lines)


# =============================================================================
# ANA İŞLEM FONKSİYONU
# =============================================================================
# Bu fonksiyon, tüm veri hazırlama sürecini baştan sona yönetir.

def process_dataset():
    """
    Tüm veri setini işler:
    1. Hedef klasörleri oluşturur.
    2. Veri bütünlüğünü kontrol eder (resmi olmayan etiketleri ayıklar).
    3. Verileri train ve valid olarak ayırır.
    4. Etiketleri XML'den YOLO formatına dönüştürür.
    5. Dosyaları son hedef klasörlerine kopyalar.
    """
    # Adım 1: YOLO'nun ihtiyaç duyduğu klasör yapısını oluştur.
    create_dirs()

    # Adım 2: Kaynak `annotations` klasöründeki tüm .xml uzantılı dosyaları bul ve bir listeye ata.
    xml_files = [f for f in os.listdir(SOURCE_ANNOTATIONS_DIR) if f.endswith('.xml')]

    # Adım 3: Veri Bütünlüğünü Kontrol Et. Bazen bir etiket dosyası olur ama karşılık gelen
    # resim dosyası eksik olabilir. Bu "yetim" etiketleri ayıklamamız gerekir.
    valid_xml_files = []
    for xml_name in xml_files:
        base_name = os.path.splitext(xml_name)[0]  # Dosya adından uzantıyı kaldır ('pothole1.xml' -> 'pothole1')
        img_name_png = base_name + ".png"          # Resim dosyasının adını oluştur.
        src_img_path = os.path.join(SOURCE_IMAGES_DIR, img_name_png)

        # Eğer bu isimde bir resim dosyası gerçekten kaynak klasörde varsa...
        if os.path.exists(src_img_path):
            valid_xml_files.append(xml_name)  # ...etiketi geçerli dosyalar listesine ekle.
        else:
            # ...yoksa, kullanıcıyı uyar ve bu sorunlu etiketi atla.
            print(f"UYARI: {xml_name} etiketine ait {img_name_png} resmi bulunamadı. Bu etiket atlanacak.")

    # Adım 4: Veri Setini Karıştır ve Ayır.
    # Modelin verileri belirli bir sırada görerek yanlış örüntüler öğrenmesini engellemek için
    # listeyi rastgele karıştırmak çok önemlidir.
    random.shuffle(valid_xml_files)

    # Listeyi %80 train, %20 valid olacak şekilde ayıracak olan indeksi hesapla.
    split_index = int(len(valid_xml_files) * (1 - VALID_SPLIT))
    train_files = valid_xml_files[:split_index]  # Listenin başından ayırma noktasına kadar olan kısım.
    valid_files = valid_xml_files[split_index:]  # Ayırma noktasından sonuna kadar olan kısım.

    # Kullanıcıya sürecin durumu hakkında bilgi ver.
    print(f"\nToplam {len(xml_files)} etiket bulundu, ancak sadece {len(valid_xml_files)} tanesinin resmi var.")
    print(f"{len(train_files)} tanesi train, {len(valid_files)} tanesi validation için ayrılıyor.")

    # Adım 5: Dosyaları İşle ve Kopyala. Bu iç içe fonksiyon, hem train hem de valid setleri için
    # aynı işlemleri yapacağımızdan kod tekrarını önlemek için kullanılır (DRY - Don't Repeat Yourself).
    def process_files(file_list, set_type):
        for xml_name in file_list:
            base_name = os.path.splitext(xml_name)[0]
            img_name_png = base_name + ".png"
            txt_name = base_name + ".txt"  # Oluşturulacak yeni .txt etiket dosyasının adı

            # Gerekli tüm kaynak ve hedef yollarını oluştur.
            src_xml_path = os.path.join(SOURCE_ANNOTATIONS_DIR, xml_name)
            src_img_path = os.path.join(SOURCE_IMAGES_DIR, img_name_png)
            dest_label_path = os.path.join(DEST_DIR, f"labels/{set_type}", txt_name)
            dest_img_path = os.path.join(DEST_DIR, f"images/{set_type}", img_name_png)

            try:
                # XML dosyasından resmin genişlik ve yükseklik bilgilerini oku.
                # Bu bilgi, YOLO formatına doğru dönüşüm yapabilmemiz için kritik öneme sahiptir.
                tree = ET.parse(src_xml_path)
                root = tree.getroot()
                size = root.find('size')
                img_width = int(size.find('width').text)
                img_height = int(size.find('height').text)
            except Exception as e:
                # Eğer XML okuma sırasında bir hata olursa (dosya bozuksa vb.),
                # hatayı ekrana yazdır, bu dosyayı atla ve bir sonraki ile devam et.
                print(f"HATA: {xml_name} dosyası okunurken hata oluştu: {e}")
                continue

            # Ana dönüşüm fonksiyonunu çağırarak YOLO formatındaki metni oluştur.
            yolo_data = convert_xml_to_yolo(src_xml_path, img_width, img_height)
            # Oluşturulan veriyi, `with open` bloğu ile güvenli bir şekilde yeni .txt dosyasına yaz.
            with open(dest_label_path, 'w') as f:
                f.write(yolo_data)

            # Orijinal resim dosyasını (.png) doğru hedef klasöre kopyala.
            shutil.copy(src_img_path, dest_img_path)

    # Adım 6: `process_files` fonksiyonunu hem train hem de valid listeleri için çalıştır.
    print("\nTrain verileri işleniyor...")
    process_files(train_files, "train")

    print("\nValidation verileri işleniyor...")
    process_files(valid_files, "valid")

    print("\nİşlem tamamlandı! 'dataset' klasörü artık eğitime hazır.")


# =============================================================================
# SCRIPT'İN BAŞLANGIÇ NOKTASI
# =============================================================================
# Bu, Python'da standart bir yapıdır. Anlamı: "Eğer bu script dosyası
# `python prepare_dataset.py` komutuyla doğrudan çalıştırıldıysa
# (başka bir dosya tarafından kütüphane gibi 'import' edilmediyse),
# o zaman aşağıdaki kodu çalıştır."
if __name__ == '__main__':
    # Ana işlem fonksiyonunu çağırarak tüm süreci başlat.
    process_dataset()