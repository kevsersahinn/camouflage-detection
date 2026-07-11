import sys
import os
import time 
import cv2
import numpy as np
from collections import deque # Yumuşatma filtresi için Kuyruk yapısı
sys.path.append(os.getcwd())

from gaze_tracking import GazeTracking

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

print("Sistem Başlatılıyor... Işık Analizi devrede. Lütfen kameraya bakın.")

# PENCERE KONUMU AYARI
cv2.namedWindow("Goz Takibi Testi")
cv2.moveWindow("Goz Takibi Testi", 1000, 400) 

# --- HAFIZA DEĞİŞKENLERİ ---
hedef_kilitlendi = False
son_bilinen_x = None
son_bilinen_y = None
kilitli_x = None
kilitli_y = None

# --- IŞIK ANALİZİ DEĞİŞKENİ ---
referans_parlaklik = None

# --- YUMUŞATMA (SMOOTHING) HAFIZALARI ---
# Kameranın okuduğu son 10 değeri (yaklaşık 0.3 saniyelik bakış) burada tutacağız.
# maxlen=8 sayesinde yeni veri gelince en eski veri otomatik silinir.
x_gecmisi = deque(maxlen=8)
y_gecmisi = deque(maxlen=8)

# --- OTONOM KİLİTLENME (DWELL TIME) AYARLARI ---
odak_baslangic_zamani = None 
ODAK_SURESI_SINIRI = 1.0 # Hedefe kilitlenmek için gereken saniye
son_goz_gorulme_zamani = time.time() # --- YENİ EKLENDİ ---


# --- YENİ EKLENEN ÇAPA DEĞİŞKENLERİ ---
odak_merkez_x = None
odak_merkez_y = None

# --- GÜVENLİ KAPATMA ZIRHI (ZOMBİ SÜREÇLERİ ENGELLER) ---
try:
    while True:
        _, frame = webcam.read()
        
        # --- PROAKTİF IŞIK ANALİZİ (ÇEVRESEL PARLAKLIK ÖLÇÜMÜ) ---
        gri_kare = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        anlik_parlaklik = np.mean(gri_kare)
        
        if referans_parlaklik is None:
            referans_parlaklik = anlik_parlaklik
            
        if abs(anlik_parlaklik - referans_parlaklik) > 55:
            #print(f"DIKKAT! Işık değişimi algılandı. (Eski: {referans_parlaklik:.1f}, Yeni: {anlik_parlaklik:.1f})")
            cv2.putText(frame, "ISIK DEGISIMI! KALIBRE EDILIYOR...", (30, 200), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)
            gaze = GazeTracking() 
            referans_parlaklik = anlik_parlaklik 

        # --- GÖZ TAKİBİ ANALİZİ (ZIRHLI KORUMA KALKANI) ---
        try:
            gaze.refresh(frame)
            frame = gaze.annotated_frame()
        except Exception as e:
            # Eğer kamera anlık göz kırpmasında boş/parazitli kare yakalarsa program çökmeyecek!
            # Hatayı yutup bir sonraki kareye geçecek.
            print("Sensor koptu, otomatik iyilestirme devrede...")
            gaze = GazeTracking()
            pass
        
        text = "Gozler Araniyor..."
        tetikleme_durumu = "Bekleniyor..."
        tetikleme_renk = (0, 0, 255) # Kırmızı

        x_orani = gaze.horizontal_ratio()
        y_orani = gaze.vertical_ratio()

        # Gözler açıksa ve bulunabiliyorsa anlık olarak yedekle
        oran_metni = "Bakis Bulunamadi"

        # Varsayılan başlangıç kutu boyutu
        dinamik_kutu_boyutu = 400

        # --- GÖZ VERİSİ TOPLAMA VE TOLERANS MİMARİSİ ---
        if x_orani is not None and y_orani is not None:
            son_goz_gorulme_zamani = time.time() # Gözü gördükçe süreyi tazele
            x_gecmisi.append(x_orani)
            y_gecmisi.append(y_orani)
            
            son_bilinen_x = sum(x_gecmisi) / len(x_gecmisi)
            son_bilinen_y = sum(y_gecmisi) / len(y_gecmisi)
        else:
            # Göz anlık kaybolursa (göz kırpma), kuyruğu HEMEN SİLME. 
            # Sadece 0.5 saniyeden uzun süreli kopuşlarda sistemi sıfırla.
            if (time.time() - son_goz_gorulme_zamani) > 0.5:
                x_gecmisi.clear()
                y_gecmisi.clear()
                odak_baslangic_zamani = None
                text = "Gozler Araniyor..."

       # --- OTONOM KARAR VE KİLİTLENME MEKANİZMASI ---
        if len(x_gecmisi) >= 4:
            sapma_x = max(x_gecmisi) - min(x_gecmisi)
            sapma_y = max(y_gecmisi) - min(y_gecmisi)
            toplam_sapma = (sapma_x + sapma_y) / 2

            # --- GERÇEK DİNAMİK ROI MATEMATİĞİ ---
            # Sabit sayılar yerine, gözün taradığı alanı doğrudan piksel boyutuna haritalıyoruz.
            # Göz sabitlendikçe kutu küçülür, etrafa bakındıkça doğrusal olarak büyür.
            oransal_boyut = int(180 + (toplam_sapma * 4500))
            dinamik_kutu_boyutu = max(300, min(750, oransal_boyut)) # 300px ile 750px arasında sınırla

            odak_devam_ediyor = False

            if odak_baslangic_zamani is None:
                if toplam_sapma < 0.05:
                    odak_baslangic_zamani = time.time()
                    odak_merkez_x = son_bilinen_x
                    odak_merkez_y = son_bilinen_y
                    odak_devam_ediyor = True
            else:
                mesafe_x = abs(son_bilinen_x - odak_merkez_x)
                mesafe_y = abs(son_bilinen_y - odak_merkez_y)
                toplam_mesafe = (mesafe_x + mesafe_y) / 2

                if toplam_mesafe < 0.13: 
                    odak_devam_ediyor = True
                    # Odaklanma (geri sayım) esnasında bile kutu donmaz; 
                    # o anki mikro göz hareketlerinin genişliğine göre anlık esnemeye devam eder.
                    oransal_odak_boyutu = int(180 + (toplam_mesafe * 3500))
                    dinamik_kutu_boyutu = max(300, min(500, oransal_odak_boyutu))
                else:
                    odak_baslangic_zamani = None
                    odak_merkez_x = None
                    odak_merkez_y = None

            # 2. Ekrana yazdırma ve Kilitleme İşlemi
            if odak_devam_ediyor:
                gecen_sure = time.time() - odak_baslangic_zamani
                text = f"🎯 ODAKLANILIYOR: {gecen_sure:.1f}s / {ODAK_SURESI_SINIRI}s"

                if gecen_sure >= ODAK_SURESI_SINIRI and not hedef_kilitlendi:
                    hedef_kilitlendi = True
                    kilitli_x = odak_merkez_x 
                    kilitli_y = odak_merkez_y
                    text = "✅ KOORDINAT KILITLENDI! Arayuze Gecin."
                    
                    # Süre dolduğu an kutu o esnadaki ideal boyutu neyse o değerle dosyaya yazılır
                    with open("koordinat.txt", "w") as f:
                        f.write(f"{kilitli_x},{kilitli_y},{dinamik_kutu_boyutu}")
                    print(f"OTONOM HEDEF KİLİTLENDİ! X: {kilitli_x:.2f}, Y: {kilitli_y:.2f}, Boyut: {dinamik_kutu_boyutu}px")

            elif toplam_sapma < 0.09:
                text = "👀 Inceleniyor"
                odak_baslangic_zamani = None 
                
            else:
                text = "🔍 Nesne Araniyor..."
                odak_baslangic_zamani = None 

            oran_metni = f"X: {son_bilinen_x:.2f} | Y: {son_bilinen_y:.2f} | Kutu: {dinamik_kutu_boyutu}px"
        else:
            if not hedef_kilitlendi:
                text = "Sensorler Hazirlaniyor..."
                odak_baslangic_zamani = None

        # --- DURUM METİNLERİ VE ARAYÜZ (UI) ---
        if hedef_kilitlendi:
            text = "HEDEF KILITLENDI ('r' ile ac)"
            tetikleme_durumu = "ANALIZ TETIKLENDI!"
            tetikleme_renk = (0, 255, 0) # Yeşil
            if kilitli_x is not None and kilitli_y is not None:
                oran_metni = f"Kilitli X: {kilitli_x:.2f} | Y: {kilitli_y:.2f}"
                
        # KULLANICI TALİMATLARI GÜNCELLENDİ
        cv2.putText(frame, "OTONOM: 4sn Sabit Bak | 'r': Kilidi Ac | 'c': Kalibre | 'q': Cikis", (30, 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1)    
        
        # Ekrana Yazıları Basma
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (147, 58, 31), 2)
        cv2.putText(frame, oran_metni, (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Sistem: {tetikleme_durumu}", (30, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, tetikleme_renk, 2)
        cv2.putText(frame, f"Ortam Isigi Skoru: {anlik_parlaklik:.0f}/255", (30, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # Görüntüyü %40'a kadar daha da küçültelim ki ekranda hiç yer kaplamasın
        kucuk_ekran = cv2.resize(frame, (0, 0), fx=0.7,fy=0.7)

        # Pencereyi oluştur ve "HER ZAMAN ÜSTTE (TOPMOST)" kalmasını sağla
        cv2.namedWindow("Goz Takibi Testi", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Goz Takibi Testi", cv2.WND_PROP_TOPMOST, 1)

        # Görüntüyü bu özel pencereye yansıt
        cv2.imshow("Goz Takibi Testi", kucuk_ekran)

        # --- TUŞ KONTROLLERİ (YENİ TEMİZ HALİ) ---
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord('q'):
            break
        elif tus == ord('r'): # KİLİDİ AÇMA / SIFIRLAMA TUŞU
            hedef_kilitlendi = False
            odak_baslangic_zamani = None # Sayacı da sıfırlıyoruz
            print("Kilit açıldı, normal takibe dönüldü.")
        elif tus == ord('c'):
            print("Manuel kalibrasyon tetiklendi...")
            gaze = GazeTracking() 
            referans_parlaklik = anlik_parlaklik

finally:
    # --- BURASI HAYAT KURTARAN KISIM ---
    # Sen Q'ya bassan da, hatadan dolayı çökse de, terminali kapatsan da burası çalışır.
    webcam.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # OpenCV'nin Windows'ta arka planda donmasını engeller.
    print("Sistem güvenli bir şekilde kapatıldı ve donanım serbest bırakıldı.")
cv2.destroyAllWindows()