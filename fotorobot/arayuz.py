"""Gradio arayüzü."""

import traceback

import gradio as gr

from . import __version__
from .config import Ayarlar, api_anahtari_bul
from .gemini import Gemini
from .kayit import VakaKaydi, gorsel_ac, yeni_vaka_id
from .motor import ben_noktasi
from .rotus import yara_konumlari
from .ozellikler import (
    ALAN_ETIKETLERI, ALANLAR, BOLGE_ORNEKLERI, ben_konumlari, bolge_promptu, degisiklik_bolgeleri,
    normalize, ozet_metni, prompt_olustur, varsayilan_ozellikler, yerel_cizimler,
)

ORNEK_IFADELER = [
    "Yaklaşık 25-30 yaşlarında bir erkekti. Kısa siyah kıvırcık saçlı, kirli sakallıydı. "
    "Gözleri ela, burnu kemerliydi. Sağ yanağında yara izi vardı. Esmer tenliydi.",
    "40'lı yaşların sonunda bir kadın. Uzun düz sarı saçlarını at kuyruğu yapmıştı, "
    "mavi gözlüydü. İnce dudaklı, yuvarlak yüzlüydü. Kalın çerçeveli gözlük takıyordu.",
    "60 yaşlarında, kel, kilolu bir adam. Beyaz pala bıyıklı, kalın kaşlı. "
    "Yüzünde kırışıklıklar vardı, gözlerinin altı torbalıydı.",
    "20 yaşında genç bir erkek. Yanları kısa, üstü uzun koyu kahverengi saçlı. "
    "Temiz tıraşlı, iri koyu kahverengi gözlü, belirgin çeneli. Boynunda dövme vardı.",
]

ORNEK_DEGISIKLIKLER = [
    "sakalı temizle",
    "kısa kare sakal ekle",
    "saçları sarı yap",
    "gözlük ekle",
    "10 yaş yaşlandır",
    "kel yap",
]

CSS = """
.gradio-container {max-width: 1400px !important}
#baslik h1 {margin-bottom: 0}
.rapor textarea {font-family: ui-monospace, Consolas, monospace; font-size: 13px}
"""

ETIKET_ALAN = {v: k for k, v in ALAN_ETIKETLERI.items()}

BOLGE_ADLARI = {"goz": "göz çevresi", "burun": "burun", "agiz": "ağız", "alt_yuz": "alt yüz (sakal/bıyık)",
                "kulak": "kulaklar", "sac": "saç"}

# Aksesuar kaldırılırken kullanılacak en düşük değişim miktarı (txt2img'ye geçer)
KALDIRMA_DEGISIMI = 1.0
# Aşama 3'te en düşük CFG: testte varsayılan 6'da işaretli bölge düz cilt olarak
# yeniden çizildi, 8'de ayrıntı (yara izi) çizildi
BOLGE_EN_AZ_CFG = 8.0
# Ben fırçası: Aşama 3 testinde yarıçap 9 + ağırlık 1.35 → 5/5 doğal ben
BEN_YARICAPI = 9
BEN_YER_ADLARI = {"yanak": "yanak", "kas_ustu": "kaş üstü", "alin": "alın", "cene": "çene",
                  "dudak_ustu": "dudak üstü", "burun": "burun"}


def oz_tablo(oz):
    satirlar = []
    for alan in ALANLAR:
        deger = oz.get(alan, "")
        if alan == "cinsiyet":
            deger = "Kadın" if deger == "female" else "Erkek"
        satirlar.append([ALAN_ETIKETLERI[alan], str(deger)])
    return satirlar


def tablodan_oz(tablo, temel=None):
    if tablo is None:
        return normalize({}, temel=temel)
    if hasattr(tablo, "values"):  # pandas DataFrame
        tablo = tablo.values.tolist()
    ham = {}
    for satir in tablo:
        if len(satir) < 2:
            continue
        alan = ETIKET_ALAN.get(str(satir[0]).strip(), str(satir[0]).strip())
        if alan not in ALANLAR:
            continue
        deger = "" if satir[1] is None else str(satir[1]).strip()
        if alan == "cinsiyet":
            deger = "female" if deger.lower() in ("kadın", "kadin", "female", "k") else "male"
        if deger == "—":
            deger = ""
        ham[alan] = deger
    return normalize(ham, temel=temel)


def tahmin_uyarisi(alanlar):
    """Tanığın belirtmediği, YZ'nin tamamladığı özellikler için görselin üstünde gösterilecek uyarı."""
    adlar = [ALAN_ETIKETLERI[a] for a in ALANLAR if a in (alanlar or []) and a != "cinsiyet"]
    if not adlar:
        return ""
    return ("⚠️ **Tanık tarafından belirtilmedi (YZ tahmini):** " + ", ".join(adlar)
            + ". Bu özellikler tanığa ayrıca sorulmadan kesin kabul edilmemelidir.")


def maske_ciz(boyut, noktalar, yaricap, birlestir=True):
    """Tıklanan noktalardan maske çizer: her nokta bir daire, `birlestir` ise ardışık
    noktalar fırça kalınlığında çizgiyle birleşir (yara izi gibi uzun ayrıntılar için).

    gr.ImageEditor yerine kullanılır: Gradio 5.49'un editörü programdan görsel
    verilince ön yüzde yarış durumuna düşüp (hide_resize_ui) sayfayı bozuyordu.
    """
    from PIL import Image, ImageDraw

    maske = Image.new("L", boyut, 0)
    ciz = ImageDraw.Draw(maske)
    r = max(1, int(yaricap))
    for x, y in noktalar or []:
        ciz.ellipse((x - r, y - r, x + r, y + r), fill=255)
    if birlestir and noktalar and len(noktalar) > 1:
        ciz.line([tuple(n) for n in noktalar], fill=255, width=2 * r)
    return maske


def maske_onizleme(kaynak, maske):
    """Kaynağın üstüne maskeyi yarı saydam kırmızı olarak çizer."""
    from PIL import Image

    kaynak = kaynak.convert("RGB")
    kirmizi = Image.new("RGB", kaynak.size, (255, 59, 48))
    return Image.composite(kirmizi, kaynak, maske.point(lambda p: p * 55 // 100))


def _hata_metni(e):
    print(traceback.format_exc(), flush=True)
    mesaj = str(e)
    if "out of memory" in mesaj.lower():
        return ("❌ GPU belleği yetmedi. Varyasyon sayısını azaltın veya "
                "FOTOROBOT_LOW_VRAM=1 ile düşük VRAM modunu açın.\n" + mesaj[:300])
    return f"❌ Hata: {mesaj[:500]}"


def _rotus_raporu(rotuslar):
    """Varyasyon başına rötuş listelerinden rapor satırları."""
    sayac = {}
    for yapilan in rotuslar:
        for y in yapilan:
            sayac[y] = sayac.get(y, 0) + 1
    if not sayac:
        return []
    return ["🎨 Rötuş (prompt'un tutturamadığı ayrıntı piksel düzeyinde işlendi): "
            + ", ".join(f"{y} · {n}/{len(rotuslar)} yüzde" for y, n in sayac.items())]


class Uygulama:
    def __init__(self, motor, ayarlar=None, gemini=None):
        self.motor = motor
        self.ayarlar = ayarlar or motor.ayarlar
        self.gemini = gemini or Gemini(
            api_anahtari_bul(), self.ayarlar.gemini_modelleri,
            onbellek_yolu=(self.ayarlar.cikti_klasoru / "_gemini_onbellek.json")
            if self.ayarlar.gemini_onbellek else None,
        )

    # ── Ortak ─────────────────────────────────────────────────────────

    def _kayit(self, vaka_id):
        return VakaKaydi(self.ayarlar.cikti_klasoru, vaka_id)

    def durum_metni(self):
        m = self.motor
        parcalar = []
        if m.hazir:
            import torch
            if m.cihaz == "cuda":
                parcalar.append(f"🖥️ {torch.cuda.get_device_name(0)}"
                                + (" · düşük VRAM modu" if m.ayarlar.dusuk_vram else ""))
            else:
                parcalar.append("🐢 CPU modu")
        else:
            parcalar.append("❌ Modeller yüklenmedi")
        parcalar.append("🤖 Gemini: " + ("anahtar var" if self.gemini.hazir else "yok (kural tabanlı analiz)"))
        parcalar += [f"⚠️ {n}" for n in m.durum_notlari]
        return " · ".join(parcalar)

    def anahtar_kaydet(self, anahtar):
        anahtar = (anahtar or "").strip()
        if not anahtar:
            self.gemini.anahtar_ayarla(api_anahtari_bul())
            return "ℹ️ Anahtar alanı boş; ortam değişkeni / Colab Secrets kullanılıyor." \
                if self.gemini.hazir else "ℹ️ Gemini kapalı, kural tabanlı analiz kullanılacak.", self.durum_metni()
        self.gemini.anahtar_ayarla(anahtar)
        ok, mesaj = self.gemini.baglanti_testi()
        return ("✅ " if ok else "❌ ") + mesaj, self.durum_metni()

    # ── Aşama 1 ───────────────────────────────────────────────────────

    def analiz_et(self, ifade):
        """Dönüş: (tablo, rapor, özellikler, YZ tahmini uyarısı, tahmin edilen alanlar)."""
        if not (ifade or "").strip():
            return gr.update(), "❌ Tanık ifadesi boş!", None, "", []
        oz, rapor, tahmin = self.gemini.ifade_analiz(ifade)
        rapor.append("")
        rapor.append(ozet_metni(oz))
        return oz_tablo(oz), "\n".join(rapor), oz, tahmin_uyarisi(tahmin), tahmin

    def uret(self, tablo, ifade, vaka_id, adet, adim, cfg, tohum, progress=gr.Progress()):
        if not self.motor.hazir:
            return gr.update(), [], "❌ Modeller yüklenemedi; konsoldaki hatayı kontrol edin.", vaka_id
        oz = tablodan_oz(tablo)
        benler, yaralar, kalan = [], [], oz["ozel"]
        if self.ayarlar.otomatik_ben:
            kalan, benler = ben_konumlari(kalan)
        if self.ayarlar.otomatik_rotus:
            # Prompt'taki yara izi turuncu leke/dövme gibi çiziliyordu; yeri belliyse rötuşla çizilir
            kalan, yaralar = yara_konumlari(kalan)
        pozitif, negatif = prompt_olustur(dict(oz, ozel=kalan) if benler or yaralar else oz)
        vaka_id = vaka_id or yeni_vaka_id()
        kayit = self._kayit(vaka_id)
        try:
            sonuclar = self.motor.metinden_uret(
                pozitif, negatif, adet=int(adet), tohum=int(tohum), adim=int(adim), cfg=float(cfg),
                ilerleme=lambda o, m: progress(o, desc=m),
            )
            cizilen = self._benleri_ciz(sonuclar, benler, int(adim), progress) if benler else 0
            rotuslar = self._rotus(sonuclar, oz, yaralar, int(adim), progress)                 if self.ayarlar.otomatik_rotus else []
        except Exception as e:  # noqa: BLE001
            return gr.update(), [], _hata_metni(e), vaka_id
        galeri, liste = [], []
        for i, (gorsel, t) in enumerate(sonuclar):
            yol = kayit.gorsel_kaydet(gorsel, f"asama1_v{i + 1}", {"prompt": pozitif, "negatif": negatif, "tohum": t})
            galeri.append((yol, f"Varyasyon {i + 1} · tohum {t}"))
            liste.append({"yol": yol, "tohum": t, "oz": oz})
        kayit.olay_ekle("asama1_uretim", {
            "tanik_ifadesi": ifade, "ozellikler": oz, "prompt": pozitif, "negatif_prompt": negatif,
            "tohumlar": [t for _, t in sonuclar], "adim": int(adim), "cfg": float(cfg),
            "gorseller": [x["yol"] for x in liste],
            "otomatik_benler": [{"yer": y, "taraf": tr} for y, tr in benler],
            "rotuslar": rotuslar,
        })
        rapor = [f"✅ {len(sonuclar)} varyasyon üretildi · Vaka: {vaka_id}"]
        if benler:
            yerler = ", ".join(BEN_YER_ADLARI[y] + (f" ({'sağ' if tr == 'sag' else 'sol'})" if tr else "")
                               for y, tr in benler)
            rapor.append(f"🖌️ Ben yüz noktalarına göre çizildi: {yerler} · {cizilen}/{len(sonuclar)} yüzde")
        rapor += _rotus_raporu(rotuslar)
        rapor += ["", "Prompt:", pozitif]
        return galeri, liste, "\n".join(rapor), vaka_id

    def _benleri_ciz(self, sonuclar, benler, adim, progress):
        """Yeri belli benleri her varyasyona Aşama 3 yöntemiyle çizer (sonuclar yerinde
        güncellenir). Yüzü bulunamayan varyasyon olduğu gibi kalır. Dönüş: çizilen yüz sayısı."""
        pozitif, negatif = bolge_promptu(BOLGE_ORNEKLERI["Ben"])
        cizilen = 0
        for i, (gorsel, t) in enumerate(sonuclar):
            progress(i / len(sonuclar), desc=f"Ben çiziliyor ({i + 1}/{len(sonuclar)})")
            yuz = self.motor.yuz_bul(gorsel)
            if yuz is None:
                continue
            noktalar = [ben_noktasi(yuz.kps, yer, taraf) for yer, taraf in benler]
            maske = maske_ciz(gorsel.size, noktalar, BEN_YARICAPI, birlestir=False)
            yeni = self.motor.bolge_duzenle(gorsel, maske, pozitif, negatif, adet=1, tohum=t, adim=adim,
                                            cfg=BOLGE_EN_AZ_CFG)
            sonuclar[i] = (yeni[0][0], t)
            cizilen += 1
        return cizilen

    def _rotus(self, sonuclar, oz, yaralar, adim, progress, benzerlik_kaynagi=None, atla=()):
        """Her varyasyona rötuş uygular (sonuclar yerinde güncellenir). Dönüş: varyasyon başına yapılanlar."""
        yapilanlar = []
        for i, sonuc in enumerate(sonuclar):
            progress(i / len(sonuclar), desc=f"Renk/iz rötuşu ({i + 1}/{len(sonuclar)})")
            gorsel, t = sonuc[0], sonuc[1]
            yeni, yapilan = self.motor.rotus_uygula(gorsel, oz, tohum=t, adim=adim, yaralar=yaralar, atla=atla)
            if yapilan:
                if benzerlik_kaynagi is not None:
                    sonuc = (yeni, t, self.motor.kimlik_benzerligi(None, yeni, yuz_a=benzerlik_kaynagi))
                else:
                    sonuc = (yeni,) + tuple(sonuc[1:])
                sonuclar[i] = sonuc
            yapilanlar.append(yapilan)
        return yapilanlar

    def analiz_ve_uret(self, ifade, vaka_id, adet, adim, cfg, tohum, progress=gr.Progress()):
        tablo, analiz_raporu, oz, uyari, tahmin = self.analiz_et(ifade)
        if oz is None:
            return gr.update(), gr.update(), [], analiz_raporu, vaka_id, uyari
        galeri, liste, uretim_raporu, vaka_id = self.uret(tablo, ifade, vaka_id, adet, adim, cfg, tohum, progress)
        if liste:
            # Hangi özelliklerin tanıktan değil YZ'den geldiği vaka kaydında da dursun
            self._kayit(vaka_id).olay_ekle("tanik_ifadesi_analizi", {
                "tanik_ifadesi": ifade, "tahmin_edilen_alanlar": tahmin,
            })
        return tablo, galeri, liste, analiz_raporu + "\n\n" + uretim_raporu, vaka_id, uyari

    @staticmethod
    def secim(liste, evt: gr.SelectData):
        if not liste or evt.index is None or evt.index >= len(liste):
            return None, None
        return liste[evt.index]["yol"], evt.index

    def asama2ye_gonder(self, liste, secili, gecmis):
        if not liste:
            return (gr.update(),) * 3 + ("❌ Önce Aşama 1'de yüz üretin.", gecmis, gr.update(), gr.update())
        i = secili if secili is not None and secili < len(liste) else 0
        oge = liste[i]
        gecmis = [{"yol": oge["yol"], "not": "Aşama 1 seçimi", "oz": oge["oz"]}]
        mesaj = (f"✅ Varyasyon {i + 1} Aşama 2'ye aktarıldı."
                 + ("" if secili is not None else " (Seçim yapılmadığı için ilk varyasyon alındı.)"))
        return (oge["yol"], oge["yol"], oz_tablo(oge["oz"]), mesaj,
                gecmis, self._gecmis_galeri(gecmis), gr.Tabs(selected=1))

    # ── Aşama 2 ───────────────────────────────────────────────────────

    @staticmethod
    def _gecmis_galeri(gecmis):
        return [(g["yol"], f"{i}. {g['not']}") for i, g in enumerate(gecmis or [])]

    def kaynak_yuklendi(self, gorsel):
        """Kullanıcı Aşama 2'ye kendi fotoğrafını yüklediğinde özellikleri tahmin eder."""
        pil = gorsel_ac(gorsel)
        if pil is None:
            return (gr.update(),) * 5
        rapor = []
        temel = None
        if self.motor.hazir:
            ozet = self.motor.yuz_ozeti(pil)
            if ozet is None:
                return (None, None, gr.update(), "❌ Yüklenen görselde yüz bulunamadı.", [])
            temel = varsayilan_ozellikler(ozet["cinsiyet"])
            temel["yas"] = ozet["yas"]
            rapor.append(f"✅ Yüz bulundu (tahmini yaş {ozet['yas']}, "
                         f"{'kadın' if ozet['cinsiyet'] == 'female' else 'erkek'})")
        oz, mesaj = self.gemini.gorsel_analiz(pil, temel=temel)
        rapor.append(mesaj)
        oz = oz or temel or varsayilan_ozellikler()
        rapor.append("ℹ️ Özellikleri tablodan düzeltebilirsiniz.")
        yol = self._yuklenen_kaydet(pil)
        gecmis = [{"yol": yol, "not": "Yüklenen fotoğraf", "oz": oz}]
        return (yol, yol, oz_tablo(oz), "\n".join(rapor), gecmis)

    def _yuklenen_kaydet(self, pil):
        from datetime import datetime
        klasor = self.ayarlar.cikti_klasoru / "_yuklenenler"
        klasor.mkdir(parents=True, exist_ok=True)
        yol = klasor / datetime.now().strftime("yuklenen_%Y%m%d_%H%M%S_%f.png")
        pil.save(yol)
        return str(yol)

    def duzenle(self, kaynak, kimlik_ref, tablo, degisiklik, vaka_id, gecmis, kimlik_gucu, degisim,
                adet, adim, cfg, tohum, progress=gr.Progress()):
        kaynak_pil = gorsel_ac(kaynak)
        if kaynak_pil is None:
            return [], [], "❌ Önce Aşama 1'den bir yüz gönderin veya bir fotoğraf yükleyin.", vaka_id
        if not (degisiklik or "").strip():
            return [], [], "❌ Ne değişsin? Kutusu boş.", vaka_id
        if not self.motor.hazir:
            return [], [], "❌ Modeller yüklenemedi.", vaka_id
        kaynak_oz = tablodan_oz(tablo)
        yeni_oz, rapor = self.gemini.degisiklik(kaynak_oz, degisiklik)
        if yeni_oz == kaynak_oz:
            rapor.append("⚠️ Özelliklerde değişiklik olmadı; sonuçlar kaynağa çok benzeyecek.")
        # Kimlik ilk yüzden alınır; ona göre biriken değişiklikler de vurgulanır ve
        # saçı o yüzden farklıysa kırpıntıdaki saç maskelenir
        kimlik_oz = (gecmis[0].get("oz") if gecmis else None) or kaynak_oz
        # Yeri belli yara izleri prompt'a yazılmaz (turuncu leke/dövme gibi çıkıyordu): görüntüdeki
        # izler img2img/bölgesel düzenlemeyle korunur, yeni eklenenler rötuşla çizilir
        uretim_yeni, uretim_kaynak, uretim_kimlik, yeni_yaralar = yeni_oz, kaynak_oz, kimlik_oz, []
        if self.ayarlar.otomatik_rotus:
            kalan_yeni, yaralar_yeni = yara_konumlari(yeni_oz["ozel"])
            kalan_eski, yaralar_eski = yara_konumlari(kaynak_oz["ozel"])
            yeni_yaralar = [y for y in yaralar_yeni if y not in yaralar_eski]
            uretim_yeni = dict(yeni_oz, ozel=kalan_yeni)
            uretim_kaynak = dict(kaynak_oz, ozel=kalan_eski)
            uretim_kimlik = dict(kimlik_oz, ozel=yara_konumlari(kimlik_oz.get("ozel"))[0])
        # Sakal ekleme ve yaşlandırma FaceID'siz yerel çizimle yapılır; kalan değişiklikler FaceID'li yolla
        yereller, yerel_alanlar = yerel_cizimler(uretim_kaynak, uretim_yeni, burun=self.ayarlar.yerel_burun) \
            if self.ayarlar.yerel_cizim else ([], set())
        uretim_kalan = dict(uretim_yeni, **{a: uretim_kaynak[a] for a in yerel_alanlar})
        pozitif, negatif = prompt_olustur(uretim_kalan, onceki=uretim_kaynak, referans=uretim_kimlik)
        bolgeler = degisiklik_bolgeleri(uretim_kaynak, uretim_kalan) if self.ayarlar.bolgesel_duzenleme else None
        yalniz_yerel = uretim_kalan == uretim_kaynak and bool(yeni_yaralar or yereller)
        sac_maskele = yeni_oz["sac"] != kimlik_oz.get("sac")
        if sac_maskele:
            rapor.append("ℹ️ Saç değiştiği için kimlik kırpıntısında saç bölgesi maskelendi.")
        # Kaynakta bulunan bir nesne (gözlük, şapka...) kaldırılıyorsa img2img onu
        # başlangıç görselinden geri getirir; testlerde ancak ~0.9+ değişimle kalktı.
        eski_aks = {a.strip() for a in (kaynak_oz.get("aksesuar") or "").split(",") if a.strip()}
        yeni_aks = {a.strip() for a in (yeni_oz.get("aksesuar") or "").split(",") if a.strip()}
        if eski_aks - yeni_aks and float(degisim) < KALDIRMA_DEGISIMI:
            degisim = KALDIRMA_DEGISIMI
            rapor.append(f"ℹ️ Aksesuar kaldırıldığı için değişim miktarı otomatik {KALDIRMA_DEGISIMI} yapıldı.")
        vaka_id = vaka_id or yeni_vaka_id()
        kayit = self._kayit(vaka_id)
        try:
            if yalniz_yerel:
                # Yalnızca yerel değişiklik (yara izi, sakal, yaşlanma): yüzün geri kalanı yeniden üretilmez
                rapor.append("ℹ️ Yalnızca yerel değişiklik olduğu için yüzün geri kalanı yeniden üretilmedi.")
                sonuclar = [(kaynak_pil.convert("RGB"), t, None)
                            for t in self.motor._tohumlar(int(tohum), int(adet))]
            else:
                sonuclar, _ = self.motor.kimlik_koruyarak_uret(
                    kaynak_pil, pozitif, negatif, adet=int(adet), tohum=int(tohum), adim=int(adim),
                    cfg=float(cfg), kimlik_gucu=float(kimlik_gucu), degisim=float(degisim),
                    kimlik_gorseli=gorsel_ac(kimlik_ref), sac_maskele=sac_maskele, bolgeler=bolgeler,
                    ilerleme=lambda o, m: progress(o, desc=m),
                )
            kimlik_yuzu = self.motor.yuz_bul(gorsel_ac(kimlik_ref) or kaynak_pil)
            for y in yereller:
                for i, (gorsel, t, _) in enumerate(sonuclar):
                    progress(i / len(sonuclar), desc=f"Yerel çizim: {y['ad']} ({i + 1}/{len(sonuclar)})")
                    yeni = self.motor.yerel_ciz(gorsel, y["bolgeler"], y["pozitif"], y["negatif"], adet=1,
                                                tohum=t, adim=int(adim), cfg=y["cfg"], guc=y["guc"],
                                                en_cok_koyulasma=y.get("en_cok_koyulasma"))[0][0]
                    sonuclar[i] = (yeni, t, self.motor.kimlik_benzerligi(None, yeni, yuz_a=kimlik_yuzu)
                                   if kimlik_yuzu is not None else None)
            if yereller:
                rapor.append("🖌️ FaceID'siz yerel çizim: " + ", ".join(y["ad"] for y in yereller))
            rotuslar = []
            if self.ayarlar.otomatik_rotus:
                # Yerel çizimle rengi istenerek çizilen sakal yeniden boyanınca sprey boya gibi duruyordu
                rotuslar = self._rotus(sonuclar, yeni_oz, yeni_yaralar, int(adim), progress,
                                       benzerlik_kaynagi=kimlik_yuzu, atla=yerel_alanlar)
                rapor += _rotus_raporu(rotuslar)
        except Exception as e:  # noqa: BLE001
            return [], [], "\n".join(rapor + [_hata_metni(e)]), vaka_id
        galeri, liste = [], []
        for i, (gorsel, t, benzerlik) in enumerate(sonuclar):
            b = "yüz bulunamadı" if benzerlik is None else f"%{benzerlik * 100:.0f}"
            yol = kayit.gorsel_kaydet(gorsel, f"asama2_v{i + 1}", {
                "prompt": pozitif, "negatif": negatif, "tohum": t, "degisiklik": degisiklik,
                "kimlik_benzerligi": benzerlik,
            })
            galeri.append((yol, f"Varyasyon {i + 1} · kimlik benzerliği {b}"))
            liste.append({"yol": yol, "tohum": t, "oz": yeni_oz, "benzerlik": benzerlik, "not": degisiklik})
        kayit.olay_ekle("asama2_duzenleme", {
            "degisiklik": degisiklik, "onceki_ozellikler": kaynak_oz, "yeni_ozellikler": yeni_oz,
            "prompt": pozitif, "kimlik_gucu": float(kimlik_gucu), "degisim": float(degisim),
            "tohumlar": [t for _, t, _ in sonuclar],
            "benzerlikler": [b for _, _, b in sonuclar], "gorseller": [x["yol"] for x in liste],
        })
        rapor.append("")
        rapor.append(f"✅ {len(sonuclar)} varyasyon üretildi.")
        if bolgeler and getattr(self.motor, "son_maske", None) is not None:
            rapor.append("ℹ️ Yalnızca değişen bölge yeniden çizildi: "
                         + ", ".join(BOLGE_ADLARI.get(b, b) for b in sorted(bolgeler)))
        rapor.append("ℹ️ Kimlik benzerliği (ArcFace): %50 üzeri genelde aynı kişi kabul edilir.")
        rapor.append("")
        rapor.append(ozet_metni(yeni_oz))
        return galeri, liste, "\n".join(rapor), vaka_id

    def yeni_kaynak_yap(self, liste, secili, gecmis):
        if not liste:
            return gr.update(), gr.update(), "❌ Önce düzenleme yapın.", gecmis, gr.update()
        i = secili if secili is not None and secili < len(liste) else 0
        oge = liste[i]
        gecmis = list(gecmis or []) + [{"yol": oge["yol"], "not": oge["not"], "oz": oge["oz"]}]
        return (oge["yol"], oz_tablo(oge["oz"]),
                f"✅ Varyasyon {i + 1} yeni kaynak oldu. Sıradaki değişikliği yazabilirsiniz.",
                gecmis, self._gecmis_galeri(gecmis))

    def geri_al(self, gecmis):
        if not gecmis or len(gecmis) < 2:
            return gr.update(), gr.update(), "ℹ️ Geri alınacak adım yok.", gecmis, gr.update()
        gecmis = list(gecmis[:-1])
        son = gecmis[-1]
        return (son["yol"], oz_tablo(son["oz"]), "↩️ Bir önceki kaynağa dönüldü.",
                gecmis, self._gecmis_galeri(gecmis))

    @staticmethod
    def asama3e_gonder(liste, secili, kaynak):
        """Aşama 2'de seçili sonucu (seçim yoksa kaynağı) Aşama 3'ün kaynağı yapar."""
        if liste and secili is not None and secili < len(liste):
            return liste[secili]["yol"], f"✅ Varyasyon {secili + 1} Aşama 3'e aktarıldı."
        if kaynak:
            return kaynak, "✅ Düzenlenen yüz Aşama 3'e aktarıldı."
        return gr.update(), "❌ Önce bir yüz seçin veya yükleyin."

    # ── Aşama 3 ───────────────────────────────────────────────────────

    @staticmethod
    def bolge_onizle(yol, noktalar, yaricap, birlestir):
        kaynak = gorsel_ac(yol)
        if kaynak is None:
            return None
        return maske_onizleme(kaynak, maske_ciz(kaynak.size, noktalar, yaricap, birlestir))

    def bolge_tikla(self, yol, noktalar, yaricap, birlestir, evt: gr.SelectData):
        if not yol or evt.index is None:
            return gr.update(), noktalar
        noktalar = list(noktalar or []) + [[int(evt.index[0]), int(evt.index[1])]]
        return self.bolge_onizle(yol, noktalar, yaricap, birlestir), noktalar

    def bolge_son_nokta_sil(self, yol, noktalar, yaricap, birlestir):
        noktalar = list(noktalar or [])[:-1]
        return self.bolge_onizle(yol, noktalar, yaricap, birlestir), noktalar

    def bolge_duzenle(self, yol, noktalar, yaricap, birlestir, detay, vaka_id, guc, adet, adim, cfg, tohum,
                      progress=gr.Progress()):
        kaynak_pil = gorsel_ac(yol)
        if kaynak_pil is None:
            return [], [], "❌ Önce bir yüz gönderin veya yükleyin.", vaka_id
        if not noktalar:
            return [], [], "❌ Ayrıntının olacağı yere görsel üzerinde tıklayın.", vaka_id
        maske = maske_ciz(kaynak_pil.size, noktalar, yaricap, birlestir)
        cfg = max(float(cfg), BOLGE_EN_AZ_CFG)
        if not (detay or "").strip():
            return [], [], "❌ Eklenecek ayrıntıyı yazın veya listeden seçin.", vaka_id
        if not self.motor.hazir:
            return [], [], "❌ Modeller yüklenemedi.", vaka_id
        pozitif, negatif = bolge_promptu(detay)
        vaka_id = vaka_id or yeni_vaka_id()
        kayit = self._kayit(vaka_id)
        maske_yol = kayit.gorsel_kaydet(maske, "asama3_maske", {"detay": detay})
        try:
            sonuclar = self.motor.bolge_duzenle(
                kaynak_pil, maske, pozitif, negatif, adet=int(adet), tohum=int(tohum), adim=int(adim),
                cfg=float(cfg), guc=float(guc), ilerleme=lambda o, m: progress(o, desc=m),
            )
        except Exception as e:  # noqa: BLE001
            return [], [], _hata_metni(e), vaka_id
        galeri, liste = [], []
        for i, (gorsel, t, benzerlik) in enumerate(sonuclar):
            b = "yüz bulunamadı" if benzerlik is None else f"%{benzerlik * 100:.0f}"
            yol = kayit.gorsel_kaydet(gorsel, f"asama3_v{i + 1}", {
                "prompt": pozitif, "negatif": negatif, "tohum": t, "detay": detay, "guc": float(guc),
                "kimlik_benzerligi": benzerlik,
            })
            galeri.append((yol, f"Varyasyon {i + 1} · kimlik benzerliği {b}"))
            liste.append({"yol": yol, "tohum": t})
        kayit.olay_ekle("asama3_bolge_duzenleme", {
            "detay": detay, "prompt": pozitif, "guc": float(guc), "maske": maske_yol,
            "tohumlar": [t for _, t, _ in sonuclar], "benzerlikler": [b for _, _, b in sonuclar],
            "gorseller": [x["yol"] for x in liste],
        })
        rapor = [f"✅ {len(sonuclar)} varyasyon üretildi · Vaka: {vaka_id}",
                 "ℹ️ Maskenin dışı değişmez; kimlik benzerliği kaynağa göredir.",
                 f"ℹ️ CFG {cfg:g} (Aşama 3'te en az {BOLGE_EN_AZ_CFG:g}). Ayrıntı çıkmazsa fırçayı büyütüp "
                 "tarifi daha belirgin yazın.", "", "Prompt:", pozitif]
        return galeri, liste, "\n".join(rapor), vaka_id

    @staticmethod
    def bolge_yeni_kaynak(liste, secili):
        if not liste:
            return gr.update(), "❌ Önce bölge düzenlemesi yapın."
        i = secili if secili is not None and secili < len(liste) else 0
        return liste[i]["yol"], f"✅ Varyasyon {i + 1} yeni kaynak oldu. Sıradaki bölgeyi işaretleyebilirsiniz."

    def zip_indir(self, vaka_id):
        if not vaka_id:
            return gr.update(value=None, visible=False), "ℹ️ Henüz kayıt yok."
        yol = self._kayit(vaka_id).zip_olustur()
        return gr.update(value=yol, visible=True), f"✅ Vaka arşivi hazır: {vaka_id}.zip"

    # ── Arayüz ────────────────────────────────────────────────────────

    def olustur(self):
        with gr.Blocks(title=f"Adli Fotorobot v{__version__}", theme=gr.themes.Soft(), css=CSS) as demo:
            gr.Markdown(
                f"# 🔍 Adli Fotorobot v{__version__}\n"
                "Tanık ifadesinden fotogerçekçi fotorobot üretimi · Realistic Vision V5.1 + "
                "IP-Adapter FaceID Plus V2 + Gemini",
                elem_id="baslik",
            )
            durum = gr.Markdown(self.durum_metni())

            vaka_id = gr.State(None)
            s1_liste = gr.State([])
            s1_secili = gr.State(None)
            s2_liste = gr.State([])
            s2_secili = gr.State(None)
            kimlik_ref = gr.State(None)
            gecmis = gr.State([])
            s3_liste = gr.State([])
            s3_secili = gr.State(None)
            bolge_kaynak = gr.State(None)
            bolge_noktalar = gr.State([])

            with gr.Tabs() as sekmeler:
                with gr.Tab("1️⃣ Tanık İfadesi → Fotorobot", id=0):
                    with gr.Row():
                        with gr.Column(scale=1):
                            ifade = gr.Textbox(
                                label="📝 Tanık ifadesi", lines=8,
                                placeholder="Örn: 30 yaşlarında, kısa siyah saçlı, kirli sakallı, ela gözlü bir erkek...",
                            )
                            gr.Examples(ORNEK_IFADELER, inputs=ifade, label="Örnek ifadeler")
                            with gr.Row():
                                btn_analiz = gr.Button("🔍 Sadece analiz et")
                                btn_hepsi = gr.Button("⚡ Analiz et ve üret", variant="primary")
                            tablo1 = gr.Dataframe(
                                headers=["Özellik", "Değer (İngilizce)"], datatype=["str", "str"],
                                col_count=(2, "fixed"), interactive=True, wrap=True,
                                label="🧬 Çıkarılan özellikler — değerleri düzenleyip yeniden üretebilirsiniz",
                                value=oz_tablo(varsayilan_ozellikler()),
                            )
                            with gr.Row():
                                adet1 = gr.Slider(1, 8, value=4, step=1, label="Varyasyon sayısı")
                                btn_uret = gr.Button("🎨 Tablodan üret", variant="secondary")
                            rapor1 = gr.Textbox(label="📋 Rapor", lines=10, interactive=False,
                                                elem_classes="rapor")
                        with gr.Column(scale=2):
                            tahmin_bandi = gr.Markdown()
                            galeri1 = gr.Gallery(label="🖼️ Üretilen yüzler — birini seçin", columns=4,
                                                 height=430, object_fit="contain", preview=False)
                            with gr.Row():
                                secilen1 = gr.Image(label="✅ Seçilen", height=330, interactive=False,
                                                    type="filepath")
                                with gr.Column():
                                    btn_gonder = gr.Button("➡️ Seçili yüzü Aşama 2'ye gönder",
                                                           variant="primary", size="lg")
                                    gonder_durum = gr.Markdown()

                with gr.Tab("2️⃣ Kimlik Korumalı Düzenleme", id=1):
                    gr.Markdown(
                        "Kimlik (yüz yapısı) IP-Adapter FaceID ile korunur, yalnızca istenen özellikler "
                        "değişir. Her değişikliği **ayrı satıra** yazın. Kendi fotoğrafınızı da yükleyebilirsiniz."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            kaynak = gr.Image(label="🖼️ Düzenlenecek yüz (yükleyebilirsiniz)", height=340,
                                              type="filepath", sources=["upload", "clipboard"])
                            degisiklik = gr.Textbox(label="✏️ Ne değişsin?", lines=4,
                                                    placeholder="sakalı temizle\ngözlük ekle")
                            gr.Examples(ORNEK_DEGISIKLIKLER, inputs=degisiklik, label="Örnek komutlar")
                            kimlik_gucu = gr.Slider(
                                0.3, 1.2, value=0.6, step=0.05, label="Yüz ne kadar aynı kalsın?",
                                info="Yüksek: yüz daha az değişir ama istenen değişiklik zayıflar. "
                                     "Değişiklik tutmazsa (ör. gözlük) düşürün.",
                            )
                            with gr.Row():
                                adet2 = gr.Slider(1, 6, value=3, step=1, label="Varyasyon sayısı")
                                btn_duzenle = gr.Button("🔄 Güncelle", variant="primary", size="lg")
                            tablo2 = gr.Dataframe(
                                headers=["Özellik", "Değer (İngilizce)"], datatype=["str", "str"],
                                col_count=(2, "fixed"), interactive=True, wrap=True,
                                label="🧬 Kaynağın özellikleri",
                            )
                            rapor2 = gr.Textbox(label="📋 Rapor", lines=10, interactive=False,
                                                elem_classes="rapor")
                        with gr.Column(scale=2):
                            galeri2 = gr.Gallery(label="🖼️ Güncellenmiş yüzler — birini seçin", columns=3,
                                                 height=400, object_fit="contain", preview=False)
                            with gr.Row():
                                secilen2 = gr.Image(label="✅ Seçilen sonuç", height=300, interactive=False,
                                                    type="filepath")
                                with gr.Column():
                                    btn_yeni_kaynak = gr.Button("↪️ Seçili sonucu yeni kaynak yap "
                                                                "(art arda düzenleme)", variant="secondary")
                                    btn_geri = gr.Button("↩️ Önceki kaynağa dön")
                                    btn_asama3 = gr.Button("➡️ Seçili sonucu Aşama 3'e gönder "
                                                           "(yara izi, ben, küpe...)")
                                    duzenle_durum = gr.Markdown()
                            galeri_gecmis = gr.Gallery(label="🕓 Düzenleme geçmişi", columns=6, height=180,
                                                       object_fit="contain", preview=False)

                with gr.Tab("3️⃣ Bölgesel Ayrıntı", id=2):
                    gr.Markdown(
                        "Yara izi, ben, küpe gibi **küçük ve yeri belli** ayrıntılar için: ayrıntının olacağı "
                        "yere görsel üzerinde tıklayın (uzun bir iz için başından sonuna birkaç nokta; noktalar "
                        "çizgiyle birleşir), ayrıntıyı seçin. Yalnızca işaretli bölge yeniden çizilir, yüzün geri "
                        "kalanı aynen korunur. Birden fazla ayrıntı için sonucu yeni kaynak yapıp devam edin."
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            bolge_onizleme = gr.Image(label="🖌️ Ayrıntının yerine tıklayın", type="pil",
                                                      height=480, interactive=False)
                            with gr.Row():
                                yaricap = gr.Slider(3, 40, value=9, step=1, label="Fırça yarıçapı (piksel)",
                                                    info="Ayrıntı fırça kadar büyük çizilir")
                                birlestir = gr.Checkbox(value=True, label="Noktaları çizgiyle birleştir")
                            with gr.Row():
                                btn_nokta_sil = gr.Button("↩️ Son noktayı sil", size="sm")
                                btn_nokta_temizle = gr.Button("🧹 Temizle", size="sm")
                                btn_bolge_yukle = gr.UploadButton("📤 Fotoğraf yükle", file_types=["image"],
                                                                  type="filepath", size="sm")
                            hazir_detay = gr.Dropdown(list(BOLGE_ORNEKLERI), label="Hazır ayrıntı",
                                                      value=None)
                            detay = gr.Textbox(label="✏️ Eklenecek ayrıntı (İngilizce)", lines=2,
                                               placeholder="prominent visible pale healed scar line on the skin")
                            adet3 = gr.Slider(1, 6, value=3, step=1, label="Varyasyon sayısı")
                            btn_bolge = gr.Button("🖌️ Bölgeyi yeniden çiz", variant="primary", size="lg")
                            rapor3 = gr.Textbox(label="📋 Rapor", lines=6, interactive=False,
                                                elem_classes="rapor")
                        with gr.Column(scale=2):
                            galeri3 = gr.Gallery(label="🖼️ Sonuçlar — birini seçin", columns=3,
                                                 height=400, object_fit="contain", preview=False)
                            with gr.Row():
                                secilen3 = gr.Image(label="✅ Seçilen sonuç", height=300, interactive=False,
                                                    type="filepath")
                                with gr.Column():
                                    btn_bolge_kaynak = gr.Button("↪️ Seçili sonucu yeni kaynak yap "
                                                                 "(başka ayrıntı ekle)", variant="secondary")
                                    bolge_durum = gr.Markdown()

                with gr.Tab("📁 Vaka Kaydı", id=3):
                    gr.Markdown(
                        "Üretilen tüm görseller ve işlem raporu (ifade, özellikler, prompt, tohumlar, kimlik "
                        "benzerlikleri) vaka klasörüne kaydedilir. Aynı tohum ve ayarlarla sonuçlar yeniden "
                        "üretilebilir."
                    )
                    btn_zip = gr.Button("📦 Vaka arşivini hazırla (ZIP)")
                    zip_durum = gr.Markdown()
                    zip_dosya = gr.File(label="İndir", visible=False)

            # Varsayılanlar testlerle seçildi; normal kullanımda değiştirmek gerekmez.
            # Kullanılan tohum her sonucun altında ve vaka kaydında yazdığı için
            # yeniden üretilebilirlik için de bu bölüme girmek gerekmez.
            with gr.Accordion("🔧 Uzman ayarları", open=False):
                gr.Markdown("Varsayılanlar testlerle seçilmiştir; normal kullanımda değiştirmeniz gerekmez.")
                with gr.Row():
                    anahtar = gr.Textbox(
                        label="Gemini API anahtarı (yerel kurulum için; Colab'da Secrets'tan gelir)",
                        type="password", scale=3,
                        placeholder="Boş bırakılırsa kural tabanlı Türkçe analiz kullanılır",
                    )
                    btn_anahtar = gr.Button("Kaydet ve test et", scale=1)
                anahtar_durum = gr.Markdown()
                with gr.Row():
                    adim = gr.Slider(15, 60, value=25, step=1, label="Adım sayısı",
                                     info="Yüksek: biraz daha iyi kalite, daha yavaş")
                    cfg = gr.Slider(3, 10, value=6.0, step=0.5, label="Prompt'a bağlılık (CFG)",
                                    info="Yüksek: tarife daha sıkı uyar ama yapaylaşır")
                    tohum = gr.Number(value=-1, precision=0, label="Tohum (-1 = rastgele)",
                                      info="Bir sonucu birebir yeniden üretmek için altındaki tohumu girin")
                with gr.Row():
                    degisim = gr.Slider(0.3, 1.0, value=0.8, step=0.05, label="Aşama 2: değişim miktarı",
                                        info="1.0: poz ve ışık da yeniden üretilir")
                    guc3 = gr.Slider(0.4, 1.0, value=1.0, step=0.05, label="Aşama 3: değişim gücü",
                                     info="Testlerde 1.0 dışında ayrıntı çizilmedi")

            # ── Olaylar ──────────────────────────────────────────────
            btn_anahtar.click(self.anahtar_kaydet, [anahtar], [anahtar_durum, durum])

            btn_analiz.click(lambda m: [self.analiz_et(m)[i] for i in (0, 1, 3)], [ifade],
                             [tablo1, rapor1, tahmin_bandi])
            btn_hepsi.click(self.analiz_ve_uret, [ifade, vaka_id, adet1, adim, cfg, tohum],
                            [tablo1, galeri1, s1_liste, rapor1, vaka_id, tahmin_bandi])
            btn_uret.click(self.uret, [tablo1, ifade, vaka_id, adet1, adim, cfg, tohum],
                           [galeri1, s1_liste, rapor1, vaka_id])
            galeri1.select(self.secim, [s1_liste], [secilen1, s1_secili])

            btn_gonder.click(
                self.asama2ye_gonder, [s1_liste, s1_secili, gecmis],
                [kaynak, kimlik_ref, tablo2, gonder_durum, gecmis, galeri_gecmis, sekmeler],
            )

            kaynak.upload(self.kaynak_yuklendi, [kaynak],
                          [kaynak, kimlik_ref, tablo2, rapor2, gecmis]) \
                .then(self._gecmis_galeri, [gecmis], [galeri_gecmis])

            btn_duzenle.click(
                self.duzenle,
                [kaynak, kimlik_ref, tablo2, degisiklik, vaka_id, gecmis, kimlik_gucu, degisim, adet2, adim, cfg, tohum],
                [galeri2, s2_liste, rapor2, vaka_id],
            )
            galeri2.select(self.secim, [s2_liste], [secilen2, s2_secili])
            btn_yeni_kaynak.click(self.yeni_kaynak_yap, [s2_liste, s2_secili, gecmis],
                                  [kaynak, tablo2, duzenle_durum, gecmis, galeri_gecmis])
            btn_geri.click(self.geri_al, [gecmis], [kaynak, tablo2, duzenle_durum, gecmis, galeri_gecmis])
            btn_asama3.click(self.asama3e_gonder, [s2_liste, s2_secili, kaynak], [bolge_kaynak, duzenle_durum]) \
                .then(lambda: gr.Tabs(selected=2), None, [sekmeler])

            # Kaynak değişince işaretler sıfırlanır ve görsel yeniden gösterilir
            bolge_kaynak.change(lambda y: (gorsel_ac(y), []), [bolge_kaynak], [bolge_onizleme, bolge_noktalar])
            btn_bolge_yukle.upload(lambda d: d, [btn_bolge_yukle], [bolge_kaynak])
            bolge_onizleme.select(self.bolge_tikla, [bolge_kaynak, bolge_noktalar, yaricap, birlestir],
                                  [bolge_onizleme, bolge_noktalar])
            btn_nokta_sil.click(self.bolge_son_nokta_sil, [bolge_kaynak, bolge_noktalar, yaricap, birlestir],
                                [bolge_onizleme, bolge_noktalar])
            btn_nokta_temizle.click(lambda y: (gorsel_ac(y), []), [bolge_kaynak], [bolge_onizleme, bolge_noktalar])
            for bilesen in (yaricap, birlestir):
                bilesen.change(self.bolge_onizle, [bolge_kaynak, bolge_noktalar, yaricap, birlestir],
                               [bolge_onizleme])

            hazir_detay.change(lambda k: BOLGE_ORNEKLERI.get(k, gr.update()), [hazir_detay], [detay])
            btn_bolge.click(
                self.bolge_duzenle,
                [bolge_kaynak, bolge_noktalar, yaricap, birlestir, detay, vaka_id, guc3, adet3, adim, cfg, tohum],
                [galeri3, s3_liste, rapor3, vaka_id],
            )
            galeri3.select(self.secim, [s3_liste], [secilen3, s3_secili])
            btn_bolge_kaynak.click(self.bolge_yeni_kaynak, [s3_liste, s3_secili], [bolge_kaynak, bolge_durum])

            btn_zip.click(self.zip_indir, [vaka_id], [zip_dosya, zip_durum])
        return demo
