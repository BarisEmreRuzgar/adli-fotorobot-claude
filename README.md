# 🔍 Adli Fotorobot v4

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch)
![Diffusers](https://img.shields.io/badge/Diffusers-0.35-yellow?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-5-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Tanık ifadelerinden yapay zeka destekli, fotogerçekçi fotorobot üretim sistemi.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BarisEmreRuzgar/adli-fotorobot-claude/blob/main/Adli_Fotorobot_V3.ipynb)

</div>

---

## 📢 Proje Durumu

Sistem **6 GB'lık bir dizüstü GPU'sunda (RTX 3050) uçtan uca test edilmiştir.** A100 gerekmez; ücretsiz Colab T4 GPU'su (16 GB) fazlasıyla yeterlidir.

---

## 📌 Proje Hakkında

Sistem tanık ifadesini analiz ederek yüz özelliklerini (yaş, cinsiyet, saç, göz, sakal, ten, iz, aksesuar...) çıkarır ve bunlardan fotogerçekçi yüz görselleri üretir. Seçilen yüzün **kimliği (FaceID) korunarak** istenen özellikler değiştirilebilir. Tüm süreç iki aşamalı bir Gradio arayüzünden yürütülür ve her işlem vaka kaydına yazılır.

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 🤖 **Gemini analizi** | Tanık ifadesi yapılandırılmış JSON özelliklere çevrilir (model yedekli: `gemini-flash-latest` → `gemini-3.8-flash` → `gemini-flash-lite-latest`, geçici hatalarda yeniden deneme) |
| 🇹🇷 **Gemini'siz çalışma** | API anahtarı yoksa veya kota dolduysa, kural tabanlı **Türkçe analizci** devreye girer |
| 🧬 **Düzenlenebilir özellik tablosu** | Çıkarılan özellikler tabloda gösterilir; elle düzeltip yeniden üretebilirsiniz |
| 🎨 **Realistic Vision V5.1** | Fotogerçekçi, önden, nötr ifadeli portreler (cinsiyete göre dinamik prompt) |
| 🪪 **IP-Adapter FaceID Plus V2** | Kimlik korunarak özellik güncelleme (`sakalı temizle`, `gözlük ekle`, `10 yaş yaşlandır`...) |
| 📏 **Kimlik benzerlik skoru** | Her düzenlenmiş sonuç için ArcFace benzerliği gösterilir |
| 🔁 **Art arda düzenleme** | Sonucu yeni kaynak yapma, geçmiş ve geri alma; kimlik her zaman ilk yüzden alınır (kayma olmaz) |
| 📤 **Fotoğraf yükleme** | Aşama 2'ye kendi fotoğrafınızı yükleyip düzenleyebilirsiniz |
| 📁 **Vaka kaydı** | Tüm görseller, ifade, prompt, tohum ve benzerlikler `ciktilar/` altına yazılır; ZIP olarak indirilebilir |
| 💾 **Düşük VRAM desteği** | 6 GB GPU'larda çalışır; daha düşük bellekte otomatik CPU offload |

---

## 🚀 Kurulum ve Kullanım

### Seçenek 1: Google Colab (en kolayı, ücretsiz)

1. Yukarıdaki **Open in Colab** düğmesine tıklayın.
2. `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU` seçin.
3. *(İsteğe bağlı)* Sol menüdeki 🔑 **Secrets** bölümüne `GOOGLE_API_KEY` adıyla [Gemini API anahtarınızı](https://aistudio.google.com/apikey) ekleyin (ücretsiz katman yeterlidir).
4. Hücreleri sırayla çalıştırın. Son hücrenin verdiği `https://....gradio.live` bağlantısını açın.

### Seçenek 2: Kendi bilgisayarınız (Windows, NVIDIA GPU)

Gereksinimler: NVIDIA GPU (en az 6 GB önerilir), [Python 3.11](https://www.python.org/downloads/release/python-3119/), ~15 GB boş disk.

1. Depoyu indirin: `git clone https://github.com/BarisEmreRuzgar/adli-fotorobot-claude.git`
2. `kurulum.bat` dosyasına çift tıklayın (bir kez).
3. `baslat.bat` dosyasına çift tıklayın. Tarayıcıda `http://127.0.0.1:7860` açılır.

İlk çalıştırmada modeller indirilir (~7 GB). Gemini anahtarını `baslat.bat` içine yazabilir veya arayüzdeki **⚙️ Ayarlar** bölümünden girebilirsiniz.

<details>
<summary>Linux / elle kurulum</summary>

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-deps insightface==1.0.1
export GOOGLE_API_KEY=...   # isteğe bağlı
python app.py               # --share: herkese açık bağlantı, --low-vram: düşük bellek modu
```
</details>

---

## 📋 Kullanım Akışı

**Aşama 1: Tanık ifadesi → Fotorobot**
1. Tanık ifadesini yazın ve **⚡ Analiz et ve üret** düğmesine basın.
2. Özellik tablosundaki değerleri isterseniz düzeltin, sonra **🎨 Tablodan üret** ile yeniden üretin.
3. Galeriden şüpheliye en çok benzeyen yüzü seçip **➡️ Aşama 2'ye gönder** düğmesine basın.

**Aşama 2: Kimlik korumalı düzenleme**
1. Her değişikliği ayrı satıra yazın, örneğin `sakalı temizle`, `kısa kare sakal ekle`, `gözler mavi olsun`, `gözlük ekle`, `saçları sarı yap`, `kel yap`, `10 yaş yaşlandır`.
2. **Kimlik koruma gücü** (varsayılan 0.6): yüksek değerde yüz yapısı daha sıkı korunur ama değişiklikler zayıflar. Testlerde 0.8'de gözlük eklenemedi; 0.4'te ise yaşlandırma güçlü ama kimlik %50'ye düştü.
3. **Değişim miktarı** (varsayılan 0.8): düşük değerde poz ve ışık korunur, `1.0` değerinde poz da yeniden üretilir.
4. Beğendiğiniz sonucu **↪️ yeni kaynak yap** ile seçip bir sonraki değişikliğe geçin.

---

## 🏗️ Mimari

```
app.py                  → giriş noktası (komut satırı seçenekleri)
fotorobot/
  ├─ config.py          → ayarlar (ortam değişkenleri, Colab Secrets)
  ├─ ozellikler.py      → özellik şeması, doğrulama ve prompt üretimi
  ├─ kural_analiz.py    → Gemini'siz Türkçe ifade ve komut analizcisi
  ├─ gemini.py          → Gemini analizi (JSON çıktı, model yedekleme, otomatik geri dönüş)
  ├─ motor.py           → model yükleme, Aşama 1/2 üretimi, kimlik benzerliği
  ├─ kayit.py           → vaka kaydı (PNG + JSON + ZIP)
  └─ arayuz.py          → Gradio arayüzü
tests/                  → GPU gerektirmeyen birim testleri (python -m pytest)
```

**Ölçülen performans** (RTX 3050 Laptop 6 GB, 25 adım, 512×640)

| İşlem | Süre | Tepe VRAM |
|---|---|---|
| Model yükleme (önbellekten) | ~10-15 sn | 3.6 GB |
| Aşama 1, varyasyon başına | ~8.5 sn | 4.4 GB |
| Aşama 2, varyasyon başına | ~9-12 sn | 5.1 GB |

Testlerde kimlik benzerliği (ArcFace) IP-Adapter ile **%70-80**, IP-Adapter olmadan aynı img2img ile %53-56 ölçüldü.

**Otomatik ayarlamalar (test sonuçlarına dayanır)**
- *Saç değişikliği:* Kimlik kırpıntısında saç da görünür ve IP-Adapter saç rengini "kimlik" gibi korur. Saç değiştiğinde kırpıntının saç bölgesi maskelenir; saç rengi ancak böyle değişebildi.
- *Aksesuar kaldırma* (gözlük, şapka...): img2img nesneyi başlangıç görselinden geri getirdiği için değişim miktarı otomatik 1.0 yapılır.
- *Kalıcı düzenleme:* Kimlik her zaman ilk yüzden alınır. Önceki adımlarda yapılan değişiklikler (ör. sarı saç) prompt'ta vurgulanmaya ve eski halleri negatif prompt'ta kalmaya devam eder, böylece sonraki adımlarda kaybolmaz.

**Teknik notlar**
- İki aşama aynı UNet'i paylaşır. Aşama 1 sırasında IP-Adapter katmanları ve FaceID LoRA'sı geçici olarak devre dışı bırakılır, böylece Aşama 1 çıktısı IP-Adapter hiç yüklenmemiş gibi temiz kalır.
- Uzun prompt'lar 77 token'lık parçalara bölünerek kodlanır, yani CLIP sınırı nedeniyle özellikler kesilmez.
- IP-Adapter FaceID Plus V2, diffusers'ın yerleşik desteğiyle kullanılır (ArcFace kimlik gömüsü + hizalanmış yüz kırpıntısının CLIP gömüsü + `shortcut`).

---

## ⚙️ Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `GOOGLE_API_KEY` | – | Gemini API anahtarı |
| `FOTOROBOT_GEMINI_MODELS` | `gemini-flash-latest,gemini-3.8-flash,gemini-flash-lite-latest` | Sırayla denenecek modeller |
| `FOTOROBOT_SD_MODEL` | `SG161222/Realistic_Vision_V5.1_noVAE` | SD 1.5 tabanlı model |
| `FOTOROBOT_LOW_VRAM` | otomatik | `1`: CPU offload'u zorla |
| `FOTOROBOT_OUTPUT_DIR` | `ciktilar` | Vaka kayıtları klasörü |
| `FOTOROBOT_SHARE` | Colab'da `1` | Herkese açık Gradio bağlantısı |

---

## 🛠️ Sorun Giderme

| Sorun | Çözüm |
|---|---|
| `CUDA out of memory` | Varyasyon sayısını azaltın veya `python app.py --low-vram` ile başlatın |
| Gemini hatası / kota | Sistem otomatik olarak kural tabanlı analize geçer; rapor kutusunda nedeni yazar |
| "Yüz bulunamadı" | Yüzün önden ve net göründüğü bir görsel seçin |
| Düzenlemede yüz çok değişiyor | *Kimlik koruma gücünü* artırın, *değişim miktarını* azaltın |
| İstenen değişiklik görünmüyor | *Değişim miktarını* artırın (saç ve yaş için 0.8-1.0) |

---

## ⚠️ Uyarılar

* **Güvenlik:** API anahtarlarınızı kodun içine yazıp commit etmeyin; Colab Secrets veya ortam değişkeni kullanın. `--share` bağlantısını alan herkes arayüzü (ve girdiğiniz anahtarı) kullanabilir.
* **Kullanım amacı:** Bu proje eğitim ve adli teknoloji araştırması için geliştirilmiştir. Üretilen görseller gerçek bir kişinin fotoğrafı değildir, **tek başına teşhis veya delil olarak kullanılmamalıdır.**

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır. Kullanılan modellerin kendi lisansları vardır (Realistic Vision: CreativeML OpenRAIL-M, IP-Adapter FaceID: araştırma amaçlı, InsightFace modelleri: ticari olmayan araştırma).

---

<div align="center">

**Geliştirici**

### Barış Emre Ruzgar

[![GitHub](https://img.shields.io/badge/GitHub-BarisEmreRuzgar-black?style=for-the-badge&logo=github)](https://github.com/BarisEmreRuzgar)

*Bu proje, maddi imkânsızlıklara rağmen adli bilimlerde yapay zeka kullanımını modernize etmek amacıyla geliştirilmeye devam etmektedir.*

</div>
