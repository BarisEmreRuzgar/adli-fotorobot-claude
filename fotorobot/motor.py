"""Model yükleme ve görsel üretim motoru.

Aşama 1: Realistic Vision V5.1 ile metinden yüz üretimi.
Aşama 2: IP-Adapter FaceID Plus V2 ile kimlik korunarak özellik güncelleme.

İki aşama da aynı UNet'i paylaşır (bellek tasarrufu). Aşama 1 çalışırken
IP-Adapter katmanları ve FaceID LoRA'sı geçici olarak devre dışı bırakılır;
böylece Aşama 1 çıktısı IP-Adapter hiç yüklenmemiş gibi temiz kalır.
"""

import contextlib
import gc
import random
import time

import numpy as np
from PIL import Image

from .config import Ayarlar
from .ozellikler import BLOK_AYIRICI, agirliklari_ayir

FACEID_ADAPTOR = "faceid_0"
# Yalnızca tespit, kimlik (ArcFace) ve yaş/cinsiyet modelleri gerekli
YUZ_MODULLERI = ["detection", "recognition", "genderage"]
# Saç değişikliğinde kimlik kırpıntısının griye boyanan üst oranı
SAC_MASKE_ORANI = 0.3
# Aşama 3: maskenin çevresinde bağlam için bırakılan pay ve maske kenar yumuşatması (piksel)
BOLGE_KENARI = 48
BOLGE_BULANIKLIGI = 4
# Aşama 2 bölgesel düzenleme: maske kenar yumuşatması ve bölge içindeki en düşük değişim
BOLGE2_BULANIKLIGI = 12
BOLGE2_EN_AZ_DEGISIM = 0.9


def _log(cb, mesaj):
    print(mesaj, flush=True)
    if cb:
        cb(mesaj)


def token_parcala(ids, ayirici_id, uzunluk):
    """Token listesini en fazla `uzunluk` token'lık parçalara böler.

    Bölme virgül (`ayirici_id`) sınırlarından yapılır; böylece "thin metal frame
    eyeglasses" gibi bir ifade iki parçaya ayrılıp anlamını kaybetmez. Tek
    başına `uzunluk`tan uzun bir ifade mecburen ortasından bölünür.
    """
    ifadeler, gecerli = [], []
    for t in ids:
        gecerli.append(t)
        if t == ayirici_id:
            ifadeler.append(gecerli)
            gecerli = []
    if gecerli:
        ifadeler.append(gecerli)

    parcalar, parca = [], []
    for ifade in ifadeler:
        if len(parca) + len(ifade) > uzunluk and parca:
            parcalar.append(parca)
            parca = []
        while len(ifade) > uzunluk:
            parcalar.append(ifade[:uzunluk])
            ifade = ifade[uzunluk:]
        parca = parca + ifade
    if parca or not parcalar:
        parcalar.append(parca)
    return parcalar


def yuz_bolge_maskesi(kps, bbox, boyut, bolgeler):
    """InsightFace'in 5 yüz noktası (sol göz, sağ göz, burun, ağız sol/sağ köşe) ve yüz
    kutusundan istenen bölgelerin birleşik maskesini çizer (ek model gerektirmez).

    Bölgeler: goz (kaşlar + gözler + gözlük), burun, agiz, alt_yuz (bıyık + sakal +
    çene), kulak, sac (yüz ovalinin dışı).
    """
    from PIL import ImageDraw

    (lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry) = [tuple(map(float, p)) for p in kps]
    x0, y0, x1, y1 = map(float, bbox)
    w, h = x1 - x0, y1 - y0
    d = max(abs(rx - lx), 1.0)  # göz arası mesafe
    gy, my = (ly + ry) / 2, (mly + mry) / 2
    maske = Image.new("L", boyut, 0)
    ciz = ImageDraw.Draw(maske)
    for b in bolgeler:
        if b == "goz":
            ciz.rectangle((min(lx, rx) - 0.75 * d, gy - 0.6 * d, max(lx, rx) + 0.75 * d, gy + 0.45 * d), fill=255)
        elif b == "burun":
            ciz.ellipse((nx - 0.45 * d, gy + 0.05 * d, nx + 0.45 * d, ny + 0.35 * d), fill=255)
        elif b == "agiz":
            ciz.ellipse((mlx - 0.3 * d, my - 0.4 * d, mrx + 0.3 * d, my + 0.4 * d), fill=255)
        elif b == "alt_yuz":
            ciz.ellipse((x0 - 0.05 * w, ny - 0.15 * d, x1 + 0.05 * w, y1 + 0.15 * h), fill=255)
        elif b == "kulak":
            ciz.rectangle((x0 - 0.35 * w, gy - 0.1 * h, x0 + 0.08 * w, my + 0.15 * h), fill=255)
            ciz.rectangle((x1 - 0.08 * w, gy - 0.1 * h, x1 + 0.35 * w, my + 0.15 * h), fill=255)
        elif b == "sac":
            sac = Image.new("L", boyut, 255)
            ciz_sac = ImageDraw.Draw(sac)
            ciz_sac.ellipse((x0 + 0.04 * w, y0 + 0.1 * h, x1 - 0.04 * w, y1 + 0.05 * h), fill=0)
            # Omuz ve kıyafet saç değildir: çenenin biraz altından aşağısı korunur.
            kesim = y1 + 0.3 * h
            if kesim < boyut[1]:
                ciz_sac.rectangle((0, kesim, boyut[0], boyut[1]), fill=0)
            maske.paste(255, (0, 0), sac)
    return maske


def agirlikli_parcala(prompt, tokenle, ayirici_id, uzunluk):
    """`(ifade:1.3)` ağırlıklarını ayıklayıp prompt'u (token'lar, ağırlıklar) parçalarına böler.

    BLOK_AYIRICI ("BREAK") her zaman yeni bir parça başlatır; bloklar birbirine karışmaz.
    """
    sonuc = []
    for blok in prompt.split(BLOK_AYIRICI.strip()):
        blok = blok.strip(" ,")
        if not blok and sonuc:
            continue
        ids, agirliklar = [], []
        for metin, agirlik in agirliklari_ayir(blok):
            parca_ids = tokenle(metin)
            ids += parca_ids
            agirliklar += [agirlik] * len(parca_ids)
        bas = 0
        for parca in token_parcala(ids, ayirici_id, uzunluk):
            sonuc.append((parca, agirliklar[bas:bas + len(parca)]))
            bas += len(parca)
    return sonuc


class Motor:
    def __init__(self, ayarlar=None):
        self.ayarlar = ayarlar or Ayarlar()
        self.hazir = False
        self.cihaz = "cpu"
        self.dtype = None
        self.pipe = None
        self.pipe_i2i = None
        self.pipe_inp = None
        self.son_maske = None
        self.yuz_app = None
        self._ip_islemcileri = None
        self._ip_proj = None
        self._lora_var = False
        self.durum_notlari = []

    # ── Yükleme ────────────────────────────────────────────────────────

    def yukle(self, ilerleme=None):
        import torch
        from diffusers import (
            AutoencoderKL,
            DPMSolverMultistepScheduler,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionPipeline,
        )
        from transformers import CLIPVisionModelWithProjection

        if self.hazir:
            return
        t0 = time.time()
        if torch.cuda.is_available():
            self.cihaz = "cuda"
            self.dtype = torch.float16
            ad = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            _log(ilerleme, f"🖥️ GPU: {ad} ({vram:.1f} GB)")
            if self.ayarlar.dusuk_vram is None:
                self.ayarlar.dusuk_vram = vram < 5.5
        else:
            self.cihaz = "cpu"
            self.dtype = torch.float32
            _log(ilerleme, "⚠️ GPU bulunamadı, CPU kullanılacak (çok yavaş olur).")
            self.durum_notlari.append("GPU yok: üretim çok yavaş olacak.")

        _log(ilerleme, "⏳ Yüz analiz modeli (InsightFace buffalo_l) yükleniyor...")
        self._yuz_analiz_yukle()

        _log(ilerleme, "⏳ VAE yükleniyor...")
        vae = AutoencoderKL.from_pretrained(self.ayarlar.sd_vae, torch_dtype=self.dtype)

        _log(ilerleme, "⏳ CLIP görüntü kodlayıcı yükleniyor (ilk seferde ~2.5 GB indirilir)...")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.ayarlar.image_encoder,
            subfolder=self.ayarlar.image_encoder_klasor or None,
            torch_dtype=self.dtype,
        )

        _log(ilerleme, "⏳ Realistic Vision V5.1 yükleniyor (ilk seferde ~4 GB indirilir)...")
        pipe = StableDiffusionPipeline.from_pretrained(
            self.ayarlar.sd_model,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe.set_progress_bar_config(disable=True)

        _log(ilerleme, "⏳ IP-Adapter FaceID Plus V2 yükleniyor...")
        pipe.load_ip_adapter(
            self.ayarlar.ip_repo,
            subfolder=None,
            weight_name=self.ayarlar.ip_agirlik,
            image_encoder_folder=None,
        )
        unet = pipe.unet
        self._ip_islemcileri = dict(unet.attn_processors)
        self._ip_proj = unet.encoder_hid_proj
        self._lora_var = FACEID_ADAPTOR in getattr(unet, "peft_config", {})
        if not self._lora_var:
            self.durum_notlari.append("FaceID LoRA yüklenemedi (peft eksik?); kimlik koruma zayıf olabilir.")
            _log(ilerleme, "⚠️ FaceID LoRA yüklenemedi; 'peft' paketi kurulu mu?")

        if self.cihaz == "cuda" and self.ayarlar.dusuk_vram:
            _log(ilerleme, "ℹ️ Düşük VRAM modu: modeller gerektiğinde GPU'ya taşınacak.")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.cihaz)
        pipe.enable_vae_slicing()

        self.pipe = pipe
        self.pipe_i2i = StableDiffusionImg2ImgPipeline(**pipe.components)
        self.pipe_i2i.set_progress_bar_config(disable=True)
        # Aşama 3: aynı bileşenler, ayrı model indirilmez. 4 kanallı UNet ile
        # diffusers maskenin dışını her adımda kaynak görüntüyle karıştırır.
        self.pipe_inp = StableDiffusionInpaintPipeline(**pipe.components)
        self.pipe_inp.set_progress_bar_config(disable=True)
        if self.cihaz == "cuda" and self.ayarlar.dusuk_vram:
            self.pipe_i2i.enable_model_cpu_offload()
            self.pipe_inp.enable_model_cpu_offload()

        self.hazir = True
        _log(ilerleme, f"🎉 Tüm modeller hazır ({time.time() - t0:.0f} sn).")

    def _yuz_analiz_yukle(self):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        saglayicilar = ["CPUExecutionProvider"]
        if self.cihaz == "cuda":
            # Windows/Colab'da CUDA kütüphanelerini PyTorch'un getirdiğinden yükle
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception:  # noqa: BLE001
                    pass
            if "CUDAExecutionProvider" in ort.get_available_providers():
                saglayicilar.insert(0, "CUDAExecutionProvider")
        try:
            app = FaceAnalysis(name="buffalo_l", providers=saglayicilar, allowed_modules=YUZ_MODULLERI)
            app.prepare(ctx_id=0 if self.cihaz == "cuda" else -1, det_size=(640, 640))
        except Exception:  # noqa: BLE001 - CUDA sağlayıcısı sorun çıkarırsa CPU ile dene
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=YUZ_MODULLERI)
            app.prepare(ctx_id=-1, det_size=(640, 640))
            saglayicilar = ["CPUExecutionProvider"]
        self.yuz_app = app
        self.yuz_saglayici = saglayicilar[0]

    # ── Yardımcılar ───────────────────────────────────────────────────

    @contextlib.contextmanager
    def _ip_adapter_kapali(self):
        """IP-Adapter'ı geçici olarak UNet'ten çıkarır (Aşama 1 için)."""
        from diffusers.models.attention_processor import AttnProcessor2_0

        unet = self.pipe.unet
        # Projeksiyon katmanının içinde de attention katmanları var; işlemci
        # sözlüğü projeksiyon takılıyken alındığı için sıra önemlidir.
        unet.set_attn_processor({ad: AttnProcessor2_0() for ad in self._ip_islemcileri})
        unet.encoder_hid_proj = None
        unet.config.encoder_hid_dim_type = None
        if self._lora_var:
            self.pipe.set_adapters([FACEID_ADAPTOR], adapter_weights=[0.0])
        try:
            yield
        finally:
            unet.encoder_hid_proj = self._ip_proj
            unet.config.encoder_hid_dim_type = "ip_image_proj"
            # set_attn_processor verilen sözlüğü boşaltır; saklananı korumak için kopya ver
            unet.set_attn_processor(dict(self._ip_islemcileri))
            if self._lora_var:
                self.pipe.set_adapters([FACEID_ADAPTOR], adapter_weights=[1.0])

    def _calisma_cihazi(self):
        return getattr(self.pipe, "_execution_device", self.cihaz)

    def _prompt_kodla(self, pozitif, negatif):
        """77 token sınırını aşan prompt'ları virgül sınırlarından parçalara bölerek kodlar."""
        import torch

        tok = self.pipe.tokenizer
        enc = self.pipe.text_encoder
        cihaz = self._calisma_cihazi()
        uzunluk = tok.model_max_length - 2  # BOS/EOS hariç
        virgul = tok(",", add_special_tokens=False).input_ids[0]

        def tokenle(metin):
            return tok(metin, truncation=False, add_special_tokens=False).input_ids

        poz = agirlikli_parcala(pozitif, tokenle, virgul, uzunluk)
        neg = agirlikli_parcala(negatif, tokenle, virgul, uzunluk)
        # Pozitif ve negatif gömüler aynı uzunlukta olmalı; kısa olan boş parçayla tamamlanır
        parca_sayisi = max(len(poz), len(neg))
        poz += [([], [])] * (parca_sayisi - len(poz))
        neg += [([], [])] * (parca_sayisi - len(neg))

        def kodla(parcalar):
            gomuler = []
            for dilim, agirliklar in parcalar:
                dolgu = tok.model_max_length - len(dilim) - 2
                dilim = [tok.bos_token_id] + dilim + [tok.eos_token_id]
                dilim += [tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id] * dolgu
                agirliklar = [1.0] + agirliklar + [1.0] * (dolgu + 1)
                tensor = torch.tensor([dilim], device=cihaz)
                with torch.no_grad():
                    gomu = enc(tensor)[0]
                    if any(a != 1.0 for a in agirliklar):
                        # A1111 yöntemi: token gömülerini ağırlıkla çarp, genel ortalamayı koru
                        ortalama = gomu.mean()
                        gomu = gomu * torch.tensor(agirliklar, device=cihaz, dtype=gomu.dtype)[None, :, None]
                        gomu = gomu * (ortalama / gomu.mean())
                gomuler.append(gomu)
            return torch.cat(gomuler, dim=1).to(self.dtype)

        return kodla(poz), kodla(neg)

    def yuz_bul(self, pil_gorsel):
        """En büyük yüzü döndürür (insightface Face nesnesi) veya None."""
        bgr = np.ascontiguousarray(np.array(pil_gorsel.convert("RGB"))[:, :, ::-1])
        yuzler = self.yuz_app.get(bgr)
        if not yuzler:
            # Küçük/kenarda yüzler için kenarlık ekleyip tekrar dene
            pad = max(bgr.shape[:2]) // 4
            genis = np.pad(bgr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            yuzler = self.yuz_app.get(genis)
            for y in yuzler:
                y.bbox = y.bbox - pad
                y.kps = y.kps - pad
        if not yuzler:
            return None
        return max(yuzler, key=lambda y: (y.bbox[2] - y.bbox[0]) * (y.bbox[3] - y.bbox[1]))

    def kimlik_benzerligi(self, gorsel_a, gorsel_b=None, yuz_a=None):
        """İki yüz arasındaki ArcFace kosinüs benzerliği (0-1 arası, ~0.5+ aynı kişi)."""
        if yuz_a is None:
            yuz_a = self.yuz_bul(gorsel_a)
        yuz_b = self.yuz_bul(gorsel_b)
        if yuz_a is None or yuz_b is None:
            return None
        return float(np.dot(yuz_a.normed_embedding, yuz_b.normed_embedding))

    def yuz_ozeti(self, pil_gorsel):
        """Yüklenen fotoğraf için (cinsiyet, yaş) tahmini."""
        yuz = self.yuz_bul(pil_gorsel)
        if yuz is None:
            return None
        cinsiyet = "male" if int(getattr(yuz, "gender", 1)) == 1 else "female"
        return {"cinsiyet": cinsiyet, "yas": int(getattr(yuz, "age", 30))}

    def _kimlik_gomuleri(self, pil_gorsel, yuz, sac_maskele=False):
        """IP-Adapter FaceID Plus V2 için kimlik ve CLIP gömüleri.

        Hizalanmış yüz kırpıntısının üst kısmında saç da görünür ve CLIP gömüsü
        saç rengini/şeklini "kimlik" gibi taşır. Saç değişikliği istendiğinde
        bu bölge griye boyanır; testlerde saç rengi ancak böyle değişebildi,
        kimlik benzerliği ise neredeyse aynı kaldı.
        """
        import torch
        from insightface.utils import face_align

        cihaz = self._calisma_cihazi()
        rgb = np.array(pil_gorsel.convert("RGB"))
        kirpik = face_align.norm_crop(rgb, landmark=yuz.kps, image_size=224)
        if sac_maskele:
            kirpik = kirpik.copy()
            kirpik[: int(224 * SAC_MASKE_ORANI)] = 128

        kimlik = torch.from_numpy(yuz.normed_embedding).reshape(1, 1, 1, -1)
        kimlik = torch.cat([torch.zeros_like(kimlik), kimlik]).to(device=cihaz, dtype=self.dtype)

        clip = self.pipe.prepare_ip_adapter_image_embeds(
            [Image.fromarray(kirpik)], None, cihaz, 1, True,
        )[0]
        return kimlik, clip.to(dtype=self.dtype), Image.fromarray(kirpik)

    @staticmethod
    def _tohumlar(tohum, adet):
        if tohum is None or int(tohum) < 0:
            tohum = random.randint(0, 2 ** 31 - 1)
        return [int(tohum) + i * 1000 for i in range(adet)]

    def _bellek_temizle(self):
        import torch

        gc.collect()
        if self.cihaz == "cuda":
            torch.cuda.empty_cache()

    # ── Aşama 1 ────────────────────────────────────────────────────────

    def metinden_uret(self, pozitif, negatif, adet=4, tohum=-1, adim=25, cfg=6.0,
                      genislik=512, yukseklik=640, ilerleme=None):
        """Dönüş: [(PIL görsel, tohum), ...]."""
        import torch

        poz_emb, neg_emb = self._prompt_kodla(pozitif, negatif)
        sonuclar = []
        with self._ip_adapter_kapali():
            for i, t in enumerate(self._tohumlar(tohum, adet)):
                if ilerleme:
                    ilerleme(i / adet, f"Varyasyon {i + 1}/{adet} üretiliyor...")
                gen = torch.Generator("cpu").manual_seed(t)
                with torch.inference_mode():
                    cikti = self.pipe(
                        prompt_embeds=poz_emb,
                        negative_prompt_embeds=neg_emb,
                        width=genislik, height=yukseklik,
                        guidance_scale=cfg,
                        num_inference_steps=adim,
                        generator=gen,
                    ).images[0]
                sonuclar.append((cikti, t))
        self._bellek_temizle()
        return sonuclar

    # ── Aşama 2 ────────────────────────────────────────────────────────

    def kimlik_koruyarak_uret(self, kaynak, pozitif, negatif, adet=3, tohum=-1, adim=30,
                              cfg=6.0, kimlik_gucu=0.6, degisim=0.8, kimlik_gorseli=None,
                              sac_maskele=False, bolgeler=None, ilerleme=None):
        """Kaynak yüzün kimliğini koruyarak yeni prompt ile üretir.

        `degisim` 1.0'a yakınsa sıfırdan (txt2img) üretilir; daha düşük değerlerde
        kaynak görselin pozu/ışığı korunur (img2img). `kimlik_gorseli` verilirse
        kimlik bu görselden alınır (art arda düzenlemelerde kimlik kaymasını önler).
        `sac_maskele`: saçı kimlik görselindekinden farklı olacaksa True verin.
        `bolgeler` (ör. {"goz"}) verilirse ve kaynakta yüz bulunursa yalnızca o yüz
        bölgeleri FaceID'li inpainting ile yeniden çizilir; geri kalan pikseller korunur.
        Dönüş: ([(PIL görsel, tohum, benzerlik), ...], kimlik yüz kırpıntısı).
        Benzerlik, kimlik görseline göre hesaplanır.
        """
        import torch

        kaynak = kaynak.convert("RGB")
        kimlik_gorseli = (kimlik_gorseli or kaynak).convert("RGB")
        yuz = self.yuz_bul(kimlik_gorseli)
        if yuz is None:
            raise ValueError("Kimlik görselinde yüz bulunamadı. Yüzün net ve önden göründüğü bir görsel seçin.")
        kimlik, clip, kirpik = self._kimlik_gomuleri(kimlik_gorseli, yuz, sac_maskele=sac_maskele)
        proj = self.pipe.unet.encoder_hid_proj.image_projection_layers[0]
        proj.clip_embeds = clip
        proj.shortcut = True  # Plus V2
        proj.shortcut_scale = 1.0
        self.pipe.set_ip_adapter_scale(float(kimlik_gucu))

        poz_emb, neg_emb = self._prompt_kodla(pozitif, negatif)
        # Boyutları 8'in katına yuvarla
        g, y = (kaynak.width // 8) * 8, (kaynak.height // 8) * 8
        if max(g, y) > 768 or min(g, y) < 384:
            oran = 640 / max(g, y)
            g, y = int(g * oran) // 8 * 8, int(y * oran) // 8 * 8
        baslangic = kaynak.resize((g, y), Image.LANCZOS)
        maske = None
        if bolgeler:
            kaynak_yuz = self.yuz_bul(baslangic)
            if kaynak_yuz is not None:
                maske = yuz_bolge_maskesi(kaynak_yuz.kps, kaynak_yuz.bbox, baslangic.size, bolgeler)
                maske = self.pipe_inp.mask_processor.blur(maske, blur_factor=BOLGE2_BULANIKLIGI)
        self.son_maske = maske

        sonuclar = []
        for i, t in enumerate(self._tohumlar(tohum, adet)):
            if ilerleme:
                ilerleme(i / adet, f"Varyasyon {i + 1}/{adet} üretiliyor...")
            gen = torch.Generator("cpu").manual_seed(t)
            ortak = dict(
                prompt_embeds=poz_emb,
                negative_prompt_embeds=neg_emb,
                ip_adapter_image_embeds=[kimlik],
                guidance_scale=cfg,
                generator=gen,
            )
            with torch.inference_mode():
                if maske is not None:
                    # Bölge içinde (neredeyse) sıfırdan çizim; dışı kaynakla aynı kalır
                    guc = max(float(degisim), BOLGE2_EN_AZ_DEGISIM)
                    cikti = self.pipe_inp(image=baslangic, mask_image=maske, strength=guc,
                                          width=g, height=y,
                                          num_inference_steps=max(adim, int(round(adim / guc))),
                                          **ortak).images[0]
                elif degisim >= 0.95:
                    cikti = self.pipe(width=g, height=y, num_inference_steps=adim, **ortak).images[0]
                else:
                    # img2img'de gerçek adım sayısı adim*degisim olur; kaliteyi korumak için telafi et
                    adim_i2i = max(adim, int(round(adim / max(degisim, 0.2))))
                    cikti = self.pipe_i2i(image=baslangic, strength=float(degisim),
                                          num_inference_steps=adim_i2i, **ortak).images[0]
            benzerlik = self.kimlik_benzerligi(None, cikti, yuz_a=yuz)
            sonuclar.append((cikti, t, benzerlik))
        self._bellek_temizle()
        return sonuclar, kirpik

    # ── Aşama 3 ────────────────────────────────────────────────────────

    def bolge_duzenle(self, kaynak, maske, pozitif, negatif, adet=3, tohum=-1, adim=30, cfg=8.0,
                      guc=1.0, kenar=BOLGE_KENARI, ilerleme=None):
        """Yalnızca maskelenen bölgeyi yeniden çizer (yara izi, ben, küpe gibi küçük ayrıntılar).

        Maskenin çevresi `kenar` piksel payla kırpılıp 512 piksele büyütülür,
        orada çizilir ve kaynağa geri yapıştırılır; böylece küçük ayrıntılar tam
        çözünürlükte üretilir, maskenin dışı piksel piksel korunur.
        Dönüş: [(PIL görsel, tohum, benzerlik), ...]; benzerlik kaynağa göredir.
        """
        import torch

        kaynak = kaynak.convert("RGB")
        maske = maske.convert("L")
        if maske.size != kaynak.size:
            maske = maske.resize(kaynak.size, Image.NEAREST)
        if maske.getbbox() is None:
            raise ValueError("Maske boş: değişecek bölgeyi fırçayla boyayın.")
        maske = self.pipe_inp.mask_processor.blur(maske, blur_factor=BOLGE_BULANIKLIGI)
        yuz = self.yuz_bul(kaynak)

        poz_emb, neg_emb = self._prompt_kodla(pozitif, negatif)
        sonuclar = []
        with self._ip_adapter_kapali():
            for i, t in enumerate(self._tohumlar(tohum, adet)):
                if ilerleme:
                    ilerleme(i / adet, f"Varyasyon {i + 1}/{adet} üretiliyor...")
                gen = torch.Generator("cpu").manual_seed(t)
                with torch.inference_mode():
                    cikti = self.pipe_inp(
                        prompt_embeds=poz_emb,
                        negative_prompt_embeds=neg_emb,
                        image=kaynak, mask_image=maske,
                        strength=float(guc), padding_mask_crop=int(kenar),
                        width=512, height=512,
                        # strength adım sayısını kısaltır; kaliteyi korumak için telafi et
                        num_inference_steps=max(adim, int(round(adim / max(float(guc), 0.2)))),
                        guidance_scale=cfg,
                        generator=gen,
                    ).images[0]
                benzerlik = self.kimlik_benzerligi(None, cikti, yuz_a=yuz) if yuz is not None else None
                sonuclar.append((cikti, t, benzerlik))
        self._bellek_temizle()
        return sonuclar
