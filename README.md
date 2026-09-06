# 🔍 Adli Fotorobot V3.0

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?style=for-the-badge&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?style=for-the-badge&logo=nvidia)
![Gradio](https://img.shields.io/badge/Gradio-4.44.0-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**Tanık ifadelerinden yapay zeka destekli, adli standartlarda fotorobot üretim sistemi.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BarisEmreRuzgar/Adli-Fotorobot/blob/main/Adli_Fotorobot_V3.ipynb)

</div>

---

## 📢 Proje Durumu

> [!IMPORTANT]
> **Bu proje aktif geliştirme aşamasındadır ve henüz tamamlanmamıştır.**
>
> Kod mimarisi NVIDIA A100 (CUDA 12.1) altyapısına tam uyumlu olacak şekilde tasarlanmış ve tüm kütüphane versiyonları buna göre seçilmiştir. Ancak **A100 GPU erişiminin yüksek maliyeti nedeniyle projenin nihai testleri bugüne kadar gerçekleştirilememiştir.** Mevcut versiyon, mimari tasarım ve düşük segment GPU ortamlarında yapılan kısmi testler üzerine inşa edilmiştir.
>
> Projeye katkı sağlamak veya test ortamı desteği vermek isteyenler için katkı kapısı her zaman açıktır.

---

## 📌 Proje Hakkında

Bu sistem, tanık ifadelerini doğal dil işleme yöntemiyle analiz ederek fotogerçekçi adli yüz görselleri üretir. Üretilen görseller üzerinde kimlik yapısı korunarak (FaceID) istenen özellikler değiştirilebilir. Tüm süreç iki aşamalı, kullanıcı dostu bir Gradio arayüzü üzerinden yürütülür.

---

## ✨ Teknik Özellikler

| Özellik | Açıklama |
|---|---|
| 🤖 **Gemini 2.5 Flash** | İfade analizi ve dinamik prompt mühendisliği |
| 🎨 **Realistic Vision V5.1** | Fotorealistik adli yüz üretimi |
| 🧬 **IP-Adapter FaceID Plus V2** | Kimlik korumalı yüz özelliği güncelleme |
| ⚡ **A100 & CUDA 12.1** | Yüksek performanslı GPU mimarisi desteği |
| 🛡️ **Hata Yönetimi** | xformers fallback ve çoklu try/except katmanları |
| 🖥️ **Gradio Arayüzü** | İki aşamalı interaktif iş akışı |

---

## 🛠️ Kullanılan Teknolojiler

*   `torch==2.1.0+cu121`
*   `diffusers==0.27.2`
*   `insightface==0.7.3`
*   `onnxruntime-gpu==1.18.0`
*   `google-genai==1.5.0`
*   `xformers==0.0.25.post1`
*   `gradio==4.44.0`

---

## 🚀 Kurulum ve Kullanım

1. **Colab'da Açın:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BarisEmreRuzgar/Adli-Fotorobot/blob/main/Adli_Fotorobot_V3.ipynb)
2. **GPU Ayarı:** `Çalışma Zamanı -> Türü Değiştir -> A100 veya T4` seçin.
3. **Başlatma:** İlk hücreyi çalıştırın, oturumu yeniden başlatın ve `GOOGLE_API_KEY` değişkenini kendi anahtarınızla doldurun.
4. Tüm hücreleri sırayla çalıştırın, Gradio bağlantısı üzerinden arayüze erişin.

---

## 📋 Kullanım Akışı

* **Aşama 1 (Üretim):** Tanık ifadesini girin, Gemini analiz ederek 4 farklı fotorobot varyasyonu üretir.
* **Aşama 2 (Evrimleştirme):** Seçilen yüzün kimliği (FaceID embedding) dondurularak yalnızca belirtilen özellikler güncellenir. (örn: "sakalları kes", "gözlük ekle", "yaşlandır")

---

## ⚠️ Uyarılar

* **Güvenlik:** API anahtarlarınızı kod içinde açık şekilde commit etmeyiniz.
* **Kullanım Amacı:** Bu proje yalnızca eğitim ve adli teknoloji araştırması amacıyla geliştirilmiştir.

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

<div align="center">

**Geliştirici**

### Barış Emre Ruzgar

[![GitHub](https://img.shields.io/badge/GitHub-BarisEmreRuzgar-black?style=for-the-badge&logo=github)](https://github.com/BarisEmreRuzgar)

*Bu proje, maddi imkânsızlıklara rağmen adli bilimlerde yapay zeka kullanımını modernize etmek amacıyla geliştirilmeye devam etmektedir.*

</div>
