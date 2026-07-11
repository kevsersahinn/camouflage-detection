import gradio as gr
from PIL import Image, ImageDraw
from rembg import remove, new_session
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import sys
import subprocess

from lib.Network_Res2Net_GRA_NCD import Network

print("Yapay Zeka Motorları Yükleniyor... Lütfen bekleyin (10-15 saniye sürebilir)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan donanım: {device}")

# --- MOTOR YÜKLEMELERİ ---
saliency_session = new_session("u2net") # 1. Yapay Zeka: Belirginlik (Saliency)
sinet_model = Network(channel=32)       # 2. Yapay Zeka: Kamuflaj (SINet-V2)
sinet_model.load_state_dict(torch.load('snapshot/SINet_V2/Net_epoch_best.pth', map_location=device))
sinet_model.to(device)
sinet_model.eval()

transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- DONANIM (KAMERA) ENTEGRASYON FONKSİYONU ---
def kamerayi_baslat():
    try:
        # Arka planda goz_takibi.py dosyasını çalıştırır
        subprocess.Popen([sys.executable, "goz_takibi.py"])
        return "✅ Kamera sistemi başarıyla başlatıldı! Lütfen açılan pencereyi kenara alıp boşluk tuşu ile kilitlenin."
    except Exception as e:
        return f"❌ Kamera başlatılamadı. Hata Detayı: {str(e)}"

# --- FOTOĞRAF STANDARTLAŞTIRICI ---
def fotografi_standartlastir(img):
    if img is None: return None
    hedef_genislik = 1280
    hedef_yukseklik = 720
    img.thumbnail((hedef_genislik, hedef_yukseklik), Image.Resampling.LANCZOS)
    yeni_tuval = Image.new('RGB', (hedef_genislik, hedef_yukseklik), (0, 0, 0))
    paste_x = (hedef_genislik - img.width) // 2
    paste_y = (hedef_yukseklik - img.height) // 2
    yeni_tuval.paste(img, (paste_x, paste_y))
    return yeni_tuval

# --- MATEMATİKSEL DÖNÜŞTÜRÜCÜ ---
def oran_haritala(deger, goz_baslangic, goz_bitis, ekran_baslangic=0.0, ekran_bitis=1.0):
    fark = (goz_bitis - goz_baslangic)
    if fark == 0: return 0.5 
    oran = (deger - goz_baslangic) / fark
    ekran_degeri = ekran_baslangic + oran * (ekran_bitis - ekran_baslangic)
    return max(0.0, min(1.0, ekran_degeri))

# --- MERKEZİ ANALİZ FONKSİYONU ---
def goz_odakli_analiz(input_image):
    if input_image is None:
        return None, None, None, "Hata: Önce bir fotoğraf yüklemelisiniz."

    if not os.path.exists("koordinat.txt"):
        return input_image, None, None, "Hata: koordinat.txt bulunamadı! Önce boşluk tuşuyla kilitlenin."
        
    try:
        with open("koordinat.txt", "r") as f:
            coords = f.read().split(",")
            ham_x = float(coords[0]) 
            ham_y = float(coords[1])
            box_size = int(coords[2]) # --- KAMERANIN KARAR VERDİĞİ BOYUT BURADAN OKUNUYOR --- 
    except Exception as e:
        return input_image, None, None, "Hata: Koordinat veya Kutu boyutu verisi okunamadı."

    # --- KALİBRASYON SINIRLARI ---
    X_SOL_LIMIT = 0.800 
    X_SAG_LIMIT = 0.700 
    Y_UST_LIMIT = 0.900 
    Y_ALT_LIMIT = 1.100  
    
    # Haritalama
    gercek_x = oran_haritala(ham_x, X_SOL_LIMIT, X_SAG_LIMIT, 0.0, 1.0)
    gercek_y = oran_haritala(ham_y, Y_UST_LIMIT, Y_ALT_LIMIT, 0.0, 1.0)
    
    # --- SİSTEMATİK HATAYI TERSİNE İTME (OFFSET) ---
    # kaydırma + sağ, - sol
    gercek_x = gercek_x - 0.36
    # Nokta kaydırma + aşağı, - yukarı
    gercek_y = gercek_y - 0.15
    
    # Kalkan: Sınır dışına çıkmasını engelle
    gercek_x = max(0.0, min(1.0, gercek_x))
    gercek_y = max(0.0, min(1.0, gercek_y))
    # -----------------------------------------------
    
    # Piksel Merkezi
    width, height = input_image.size
    center_x = int(width * gercek_x)
    center_y = int(height * gercek_y)
    
    # Çerçeve Sınırları
    half_size = box_size // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(width, center_x + half_size)
    bottom = min(height, center_y + half_size)
    
    # 1. ORİJİNAL KESİTİ AL
    cropped_img = input_image.crop((left, top, right, bottom))
    
    # 2. SALIENCY (BELİRGİNLİK) ANALİZİ
    saliency_mask = remove(cropped_img, session=saliency_session, only_mask=True)
    
    # 3. KAMUFLAJ ANALİZİ (SINet-V2)
    img_tensor = transform(cropped_img).unsqueeze(0).to(device)
    with torch.no_grad():
        res, _, _, _ = sinet_model(img_tensor)
        res = F.interpolate(res, size=(cropped_img.size[1], cropped_img.size[0]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        
    res_normalized = (res - res.min()) / (res.max() - res.min() + 1e-8) * 255
    kamuflaj_mask_img = Image.fromarray(res_normalized.astype(np.uint8))
    
    # --- TESPİT MANTIĞI VE MESAJI ---
    beyaz_piksel = np.sum(res_normalized > 128)
    toplam_piksel = box_size * box_size
    
    if beyaz_piksel > (toplam_piksel * 0.02):
        durum_mesaji = f"🎯 TESPİT: Baktığınız bölgede KAMUFLE OLMUŞ nesne bulundu! (X: {center_x}, Y: {center_y})"
    else:
        durum_mesaji = f"❌ TEMİZ: Baktığınız bölgede gizlenmiş bir nesne yok. (X: {center_x}, Y: {center_y})"

    # --- ANA FOTOĞRAFA ÇİZİM ---
    gorsel_cikti = input_image.copy()
    draw = ImageDraw.Draw(gorsel_cikti)
    r = 8 
    draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), fill='red')
    draw.rectangle((left, top, right, bottom), outline='green', width=4)

    return gorsel_cikti, saliency_mask, kamuflaj_mask_img, durum_mesaji

# --- WEB ARAYÜZÜ (UI) TASARIMI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👁️ Göz Takibi Destekli Kamuflaj ve Saliency Analizi")
    gr.Markdown("İnsan beyninin 'belirgin' bulduğu nesneler ile 'gizlenmiş' nesneleri kıyaslayın.")
    
    # --- YENİ: DONANIM KONTROL PANELİ ---
    with gr.Row():
        kamera_baslat_butonu = gr.Button("📷 1. Adım: Göz Takibi Kamerasını Başlat", variant="secondary")
        kamera_durum_bilgisi = gr.Textbox(label="Donanım Durumu", interactive=False)
    # ------------------------------------

    with gr.Row():
        with gr.Column(scale=2):
            resim_girdi = gr.Image(type="pil", label="2. Adım: Fotoğrafı Yükle", height=600)
            resim_girdi.upload(fn=fotografi_standartlastir, inputs=resim_girdi, outputs=resim_girdi)
            
        with gr.Column(scale=1):
            #box_slider = gr.Slider(minimum=100, maximum=1000, value=400, step=50, label="İnceleme Çerçevesi (ROI)")
            analiz_butonu = gr.Button("3. Adım: Baktığım Bölgeyi Analiz Et", variant="primary")
            sistem_bilgisi = gr.Textbox(label="Sistem Durumu", interactive=False)
            
    with gr.Row():
        roi_cikti = gr.Image(type="pil", label="Hedef (Kırmızı Nokta ve Dinamik Çerçeve)")
        saliency_cikti = gr.Image(type="pil", label="Belirgin Nesne (Saliency)")
        kamuflaj_cikti = gr.Image(type="pil", label="Kamufle Nesne (Kamuflaj)")
        
    # Buton Tetikleyicileri
    kamera_baslat_butonu.click(fn=kamerayi_baslat, inputs=[], outputs=[kamera_durum_bilgisi])
    analiz_butonu.click(fn=goz_odakli_analiz, inputs=[resim_girdi], outputs=[roi_cikti, saliency_cikti, kamuflaj_cikti, sistem_bilgisi])

if __name__ == "__main__":
    demo.launch(share=True)