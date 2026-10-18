"""GPU gerektirmeyen testler: kural tabanlı analiz, prompt üretimi, tablo dönüşümü.

Çalıştırma:  python -m pytest tests
"""

import pytest

from fotorobot.kural_analiz import degisiklik_uygula, ifade_analiz, ozellik_cikar
from fotorobot.ozellikler import ALANLAR, normalize, prompt_olustur, varsayilan_ozellikler


@pytest.fixture
def ornek_erkek():
    oz, _ = ifade_analiz(
        "Yaklaşık 25-30 yaşlarında bir erkekti. Kısa siyah kıvırcık saçlı, kirli sakallı, "
        "ela gözlü. Burnu kemerliydi, sağ yanağında yara izi vardı. Esmer tenliydi, gözlük takıyordu."
    )
    return oz


def test_erkek_ifadesi(ornek_erkek):
    oz = ornek_erkek
    assert oz["cinsiyet"] == "male"
    assert oz["yas"] == 28
    assert oz["sac"] == "short curly black hair"
    assert oz["sakal"] == "stubble"
    assert oz["goz"] == "hazel eyes"
    assert "aquiline" in oz["burun"]
    assert "scar" in oz["ozel"] and "right cheek" in oz["ozel"]
    assert oz["ten"] == "tan brown skin"
    assert oz["aksesuar"] == "glasses"


def test_kadin_ifadesi():
    oz, _ = ifade_analiz("40lı yaşların sonunda, sarışın, mavi gözlü bir kadın. "
                         "Uzun düz saçları at kuyruğu yapılmıştı. İnce dudaklı, yuvarlak yüzlü.")
    assert oz["cinsiyet"] == "female"
    assert oz["yas"] == 48
    assert "blonde" in oz["sac"] and "long" in oz["sac"] and "ponytail" in oz["sac"]
    assert oz["goz"] == "blue eyes"
    assert oz["dudak"] == "thin lips"
    assert oz["yuz_sekli"] == "round face"
    assert oz["sakal"] == ""


@pytest.mark.parametrize("metin,yas", [
    ("30 yaşında bir adam", 30),
    ("yaklaşık 45 yaşlarında", 45),
    ("20'li yaşların başında", 22),
    ("30'lu yaşların ortasında", 35),
    ("orta yaşlı bir kadın", 45),
    ("genç bir erkek", 24),
    ("yaşlı bir adam", 65),
    ("35 ile 40 yaş arası", 38),
])
def test_yas(metin, yas):
    assert ozellik_cikar(metin)["yas"] == yas


def test_sacli_kelimesinden_sonraki_sifat_saca_yazilmaz():
    # "uzun boylu" saç uzunluğu değildir
    oz = ozellik_cikar("kıvırcık saçlı uzun boylu bir adam")
    assert oz["sac"] == "curly hair"


def test_gozluk_goz_ile_karismaz():
    oz = ozellik_cikar("gözlüklü, yeşil gözlü bir kadın")
    assert oz["goz"] == "green eyes"
    assert oz["aksesuar"] == "glasses"


def test_kel_ve_biyik():
    oz = ozellik_cikar("Yaşlı bir amca, kel, beyaz pala bıyıklı, kalın kaşlı, kilolu.")
    assert oz["sac"] == "bald shaved head"
    assert oz["sakal"] == "thick mustache"
    assert oz["kas"] == "thick eyebrows"
    assert oz["yuz_sekli"] == "chubby round face"


def test_hicbir_sey_bulunamazsa_varsayilan():
    oz, bulunan = ifade_analiz("hiçbir şey hatırlamıyorum")
    assert bulunan == []
    assert set(oz) == set(ALANLAR)


@pytest.mark.parametrize("komut,alan,beklenen", [
    ("sakalı temizle", "sakal", "clean shaven"),
    ("saçları sarı yap", "sac", "short curly blonde hair"),
    ("gözler mavi olsun", "goz", "blue eyes"),
    ("uzun sakal ekle", "sakal", "long thick full beard"),
    ("kel yap", "sac", "bald shaved head"),
])
def test_degisiklik(ornek_erkek, komut, alan, beklenen):
    yeni, uygulanan = degisiklik_uygula(ornek_erkek, komut)
    assert uygulanan
    assert yeni[alan] == beklenen


def test_degisiklik_digerlerini_korur(ornek_erkek):
    yeni, _ = degisiklik_uygula(ornek_erkek, "gözlüğü çıkar")
    assert yeni["aksesuar"] == ""
    for alan in ALANLAR:
        if alan != "aksesuar":
            assert yeni[alan] == ornek_erkek[alan]


def test_yas_komutlari(ornek_erkek):
    assert degisiklik_uygula(ornek_erkek, "daha yaşlı göster")[0]["yas"] == 43
    assert degisiklik_uygula(ornek_erkek, "10 yaş gençleştir")[0]["yas"] == 18
    assert degisiklik_uygula(ornek_erkek, "20 yaş yaşlandır")[0]["yas"] == 48


def test_anlasilmayan_degisiklik(ornek_erkek):
    yeni, uygulanan = degisiklik_uygula(ornek_erkek, "daha iyi yap")
    assert uygulanan == []
    assert yeni == ornek_erkek


def test_cok_satirli_degisiklik(ornek_erkek):
    yeni, uygulanan = degisiklik_uygula(ornek_erkek, "sakalı temizle\ngözlüğü çıkar\nsaçları kızıl yap")
    assert len(uygulanan) == 3
    assert yeni["sakal"] == "clean shaven"
    assert yeni["aksesuar"] == ""
    assert "red" in yeni["sac"]


# ── normalize / prompt ─────────────────────────────────────────────

def test_normalize_eksik_ve_hatali_alanlar():
    oz = normalize({"yas": "25-30 years", "cinsiyet": "Woman", "sakal": "full beard", "bilinmeyen": 1})
    assert oz["yas"] == 28
    assert oz["cinsiyet"] == "female"
    assert oz["sakal"] == ""  # kadında sakal olmaz
    assert "bilinmeyen" not in oz
    assert set(oz) == set(ALANLAR)


def test_normalize_yas_sinirlari():
    assert normalize({"yas": 5})["yas"] == 14
    assert normalize({"yas": 150})["yas"] == 90
    assert normalize({"yas": "bilinmiyor"})["yas"] == 30


def test_prompt_cinsiyete_gore():
    poz, neg = prompt_olustur(varsayilan_ozellikler("female"))
    assert "year-old woman" in poz
    assert "beard" in neg and "woman" not in neg
    poz, neg = prompt_olustur(varsayilan_ozellikler("male"))
    assert "year-old man" in poz
    assert "woman" in neg and "beard" in neg


def test_prompt_sakal_ve_gozluk_negatifte_olmaz(ornek_erkek):
    poz, neg = prompt_olustur(ornek_erkek)
    assert "stubble" in poz and "wearing glasses" in poz
    assert "stubble" not in neg
    assert "glasses" not in neg


def test_tablo_donusumu():
    from fotorobot.arayuz import oz_tablo, tablodan_oz

    oz = varsayilan_ozellikler("female")
    oz["yas"] = 52
    assert tablodan_oz(oz_tablo(oz)) == oz


def test_degisiklik_promptu_vurgular_ve_eskiyi_negatife_yazar(ornek_erkek):
    yeni, _ = degisiklik_uygula(ornek_erkek, "saçları sarı yap")
    poz, neg = prompt_olustur(yeni, onceki=ornek_erkek)
    # Değişen özellik prompt'un başında, eski renk negatifte
    assert poz.index("blonde") < poz.index("tan brown skin")
    assert "dyed blonde hair" in poz
    assert neg.startswith("black hair")


def test_onceki_degisiklikler_referansa_gore_kalici(ornek_erkek):
    # 1. adım: saç sarı; 2. adım: gözlük çıkar. Sarı saç vurgusu sürmeli.
    sari, _ = degisiklik_uygula(ornek_erkek, "saçları sarı yap")
    gozluksuz, _ = degisiklik_uygula(sari, "gözlüğü çıkar")
    poz, neg = prompt_olustur(gozluksuz, onceki=sari, referans=ornek_erkek)
    assert "dyed blonde hair" in poz
    assert "black hair" in neg and "glasses" in neg
    assert "wearing glasses" not in poz


def test_yas_gorunumu_promptta():
    oz = varsayilan_ozellikler()
    oz["yas"] = 60
    poz, neg = prompt_olustur(oz)
    assert "wrinkled" in poz and "youthful" in neg


def test_ayirt_edici_izler_promptun_basinda():
    oz = varsayilan_ozellikler()
    oz["ozel"] = "thin white scar above left eyebrow"
    oz["aksesuar"] = "small silver hoop earring on right ear"
    poz, _ = prompt_olustur(oz)
    from fotorobot.ozellikler import BLOK_AYIRICI

    kafa, yuz_ici, cekim = poz.split(BLOK_AYIRICI)
    # Saç kafa bloğunda; izler ve aksesuar yüz içi bloğunun başında, sakal/gözlerden önce
    assert oz["sac"] in kafa and "scar" not in kafa
    assert yuz_ici.startswith("(thin white scar")
    assert yuz_ici.index("hoop earring") < yuz_ici.index(oz["goz"])
    assert "portrait" in cekim


def test_break_bloklari_ayri_parcada_kodlanir():
    from fotorobot.motor import agirlikli_parcala

    def tokenle(metin):
        return [len(k) for k in metin.replace(",", " , ").split()]

    parcalar = agirlikli_parcala("aa bb BREAK cc, (dd:1.5) BREAK ee", tokenle, 1, 75)
    assert [p for p, _ in parcalar] == [[2, 2], [2, 1, 2], [2]]
    assert parcalar[1][1] == [1.0, 1.0, 1.5]


@pytest.mark.parametrize("gozluk, kalin_vurgu", [
    ("thin rectangular metal frame eyeglasses", False),
    ("rimless glasses", False),
    ("thick black glasses", True),
])
def test_ince_gozluge_kalin_cerceve_vurgusu_eklenmez(gozluk, kalin_vurgu):
    oz = varsayilan_ozellikler()
    oz["aksesuar"] = gozluk
    poz, _ = prompt_olustur(oz)
    assert f"wearing {gozluk}" in poz
    assert ("clearly visible frames" in poz) == kalin_vurgu


VIRGUL = 0


def test_token_parcala_virgulden_boler():
    from fotorobot.motor import token_parcala

    # ifadeler: [1,2,V] [3,4,5,V] [6,7]
    ids = [1, 2, VIRGUL, 3, 4, 5, VIRGUL, 6, 7]
    parcalar = token_parcala(ids, VIRGUL, 5)
    assert parcalar == [[1, 2, VIRGUL], [3, 4, 5, VIRGUL], [6, 7]]
    assert sum(parcalar, []) == ids


def test_token_parcala_sigan_ifadeleri_birlestirir():
    from fotorobot.motor import token_parcala

    ids = [1, VIRGUL, 2, VIRGUL, 3, VIRGUL, 4]
    assert token_parcala(ids, VIRGUL, 6) == [[1, VIRGUL, 2, VIRGUL, 3, VIRGUL], [4]]
    assert token_parcala(ids, VIRGUL, 75) == [ids]


def test_token_parcala_cok_uzun_ifadeyi_mecburen_boler():
    from fotorobot.motor import token_parcala

    ids = [9, VIRGUL] + list(range(1, 13)) + [VIRGUL, 8]
    parcalar = token_parcala(ids, VIRGUL, 5)
    assert all(len(p) <= 5 for p in parcalar)
    assert sum(parcalar, []) == ids
    assert parcalar[0] == [9, VIRGUL]


def test_token_parcala_bos():
    from fotorobot.motor import token_parcala

    assert token_parcala([], VIRGUL, 75) == [[]]


def test_agirliklari_ayir():
    from fotorobot.ozellikler import agirliklari_ayir

    assert agirliklari_ayir("a man, (thin scar, left brow:1.3), (glasses:0.8)") == [
        ("a man, ", 1.0), ("thin scar, left brow", 1.3), (", ", 1.0), ("glasses", 0.8)]
    assert agirliklari_ayir("no weights (just text)") == [("no weights (just text)", 1.0)]


def test_iz_ve_aksesuar_agirlikli(ornek_erkek):
    from fotorobot.ozellikler import AKSESUAR_AGIRLIGI, IZ_AGIRLIGI

    oz = dict(ornek_erkek, ozel="scar on right cheek", aksesuar="thin metal glasses")
    poz, _ = prompt_olustur(oz)
    assert f"(scar on right cheek:{IZ_AGIRLIGI:g})" in poz
    assert f"(wearing thin metal glasses:{AKSESUAR_AGIRLIGI:g})" in poz


def test_agirlikli_parcala_agirliklari_tokenlara_tasir():
    from fotorobot.motor import agirlikli_parcala

    sozluk = {",": VIRGUL}
    def tokenle(metin):
        # Her kelime/virgül bir token; kelimenin uzunluğu id'si olsun
        return [sozluk.get(k, len(k)) for k in metin.replace(",", " , ").split()]

    parcalar = agirlikli_parcala("aa bbb, (cccc dd:1.5), e", tokenle, VIRGUL, 4)
    assert parcalar == [([2, 3, VIRGUL], [1.0, 1.0, 1.0]),
                        ([4, 2, VIRGUL, 1], [1.5, 1.5, 1.0, 1.0])]


def test_bolge_promptu_kisa_agirlikli_ve_cilt_tarifsiz():
    from fotorobot.ozellikler import BOLGE_AGIRLIGI, BOLGE_ORNEKLERI, bolge_promptu

    for detay in BOLGE_ORNEKLERI.values():
        poz, neg = bolge_promptu(detay)
        assert poz.startswith(f"({detay}:{BOLGE_AGIRLIGI:g})")
        # Cilt tarifi modeli düz cilde itiyordu
        assert "skin" not in poz.replace(detay, "")
        # CLIP tokenizer yok; kelime sayısı 75 token'ın çok altında kalmalı
        assert len(poz.replace(",", " ").split()) < 40
    assert bolge_promptu("  scar ,  ")[0] == f"(scar:{BOLGE_AGIRLIGI:g}), photo, realistic"


def test_maske_ciz_nokta_ve_cizgi():
    from fotorobot.arayuz import maske_ciz

    tek = maske_ciz((100, 120), [[50, 60]], 5)
    assert tek.size == (100, 120) and tek.getbbox() == (45, 55, 56, 66)
    assert set(tek.getdata()) <= {0, 255}

    # Birleştirilen iki nokta arasındaki çizgi de maskelenir
    cizgi = maske_ciz((100, 120), [[20, 60], [80, 60]], 4, birlestir=True)
    ayri = maske_ciz((100, 120), [[20, 60], [80, 60]], 4, birlestir=False)
    assert cizgi.getpixel((50, 60)) == 255 and ayri.getpixel((50, 60)) == 0


def test_maske_ciz_bos_ve_onizleme():
    from PIL import Image

    from fotorobot.arayuz import maske_ciz, maske_onizleme

    assert maske_ciz((32, 32), [], 5).getbbox() is None
    kaynak = Image.new("RGB", (32, 32), (0, 0, 255))
    maske = maske_ciz((32, 32), [[16, 16]], 3)
    onizleme = maske_onizleme(kaynak, maske)
    assert onizleme.getpixel((0, 0)) == (0, 0, 255)
    r, _, b = onizleme.getpixel((16, 16))
    assert r > 100 and b < 255  # işaretli bölge kırmızıya döner


def test_arayuz_modelsiz_kurulur():
    from types import SimpleNamespace

    from fotorobot.arayuz import Uygulama
    from fotorobot.config import Ayarlar

    motor = SimpleNamespace(hazir=False, ayarlar=Ayarlar(), durum_notlari=[])
    gemini = SimpleNamespace(hazir=False)
    demo = Uygulama(motor, gemini=gemini).olustur()
    assert demo is not None


def test_erkek_yuzde_erkeklik_vurgulanir(ornek_erkek):
    from fotorobot.ozellikler import ERKEKLIK_AGIRLIGI

    vurgu = f"(masculine male face, masculine facial features:{ERKEKLIK_AGIRLIGI:g})"
    # Sakallı yüzde de (sakal tarif edildiği halde kadın yüzler çıkabiliyordu)
    sakalli = dict(ornek_erkek, sakal="short full beard")
    poz, neg = prompt_olustur(sakalli)
    assert vurgu in poz and "androgynous" in neg

    # Sakal silinip saç sarıya boyandığında (kadınsılaşan durum) erkeklik vurgulanır
    yeni, _ = degisiklik_uygula(sakalli, "sakalı temizle\nsaçları sarı yap")
    poz, neg = prompt_olustur(yeni, onceki=sakalli, referans=sakalli)
    assert vurgu in poz and "androgynous" in neg

    kadin_poz, _ = prompt_olustur(varsayilan_ozellikler("female"))
    assert "masculine" not in kadin_poz


def test_sari_sac_yellow_yazilsa_da_blonde_olur(ornek_erkek):
    oz = normalize({"sac": "short Yellow hair with receding hairline"}, temel=ornek_erkek)
    assert oz["sac"] == "short blonde hair with receding hairline"
    poz, _ = prompt_olustur(oz, onceki=ornek_erkek)
    assert "dyed blonde hair" in poz


def test_gemini_atlanan_model_rapora_yazilir():
    from types import SimpleNamespace

    from fotorobot.gemini import Gemini

    def uret(model, contents, config):
        if model == "ana":
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded")
        if model == "ikinci":
            raise RuntimeError("404 NOT_FOUND. models/ikinci is not found")
        return SimpleNamespace(text='{"cinsiyet": "male", "yas": 30}')

    g = Gemini("deneme-anahtari", ["ana", "ikinci", "yedek"])
    g._istemci = SimpleNamespace(models=SimpleNamespace(generate_content=uret))
    _, rapor, _ = g.ifade_analiz("30 yaşında bir erkek")
    metin = "\n".join(rapor)
    assert "(yedek)" in metin
    assert "ana → kota / istek sınırı doldu (429)" in metin
    assert "ikinci → model bulunamadı (404)" in metin


def test_tahmin_uyarisi():
    from fotorobot.arayuz import tahmin_uyarisi

    assert tahmin_uyarisi([]) == ""
    assert tahmin_uyarisi(["cinsiyet"]) == ""
    u = tahmin_uyarisi(["dudak", "burun", "bilinmeyen"])
    assert "Burun, Dudak" in u and "YZ tahmini" in u


def test_vaka_manifesti(tmp_path):
    import hashlib

    from PIL import Image

    from fotorobot.kayit import VakaKaydi

    kayit = VakaKaydi(tmp_path, "vaka_deneme")
    yol = kayit.gorsel_kaydet(Image.new("RGB", (8, 8)), "asama1_v1", {"tohum": 1})
    kayit.olay_ekle("deneme", {"a": 1})
    zip_yolu = kayit.zip_olustur()
    satirlar = (tmp_path / "vaka_deneme" / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    ozetler = dict(reversed(s.split("  ")) for s in satirlar)
    assert set(ozetler) == {"001_asama1_v1.png", "rapor.json"}
    with open(yol, "rb") as f:
        assert ozetler["001_asama1_v1.png"] == hashlib.sha256(f.read()).hexdigest()
    import zipfile
    assert "vaka_deneme/manifest.sha256" in zipfile.ZipFile(zip_yolu).namelist()


def test_gemini_onbellek_ayni_istegi_tekrar_gondermez(tmp_path):
    from types import SimpleNamespace

    from fotorobot.gemini import Gemini

    cagri = []

    def uret(model, contents, config):
        cagri.append(model)
        return SimpleNamespace(text='{"cinsiyet": "male", "yas": %d}' % (30 + len(cagri)))

    g = Gemini("anahtar", ["m"], onbellek_yolu=tmp_path / "onbellek.json")
    g._istemci = SimpleNamespace(models=SimpleNamespace(generate_content=uret))
    oz1, rapor1, _ = g.ifade_analiz("bir erkek")
    oz2, rapor2, _ = g.ifade_analiz("bir erkek")
    assert len(cagri) == 1 and oz1 == oz2
    assert "önbellekten" in rapor2[0]
    g.ifade_analiz("başka bir erkek")
    assert len(cagri) == 2


def test_degisiklik_bolgeleri(ornek_erkek):
    from fotorobot.ozellikler import degisiklik_bolgeleri

    yeni, _ = degisiklik_uygula(ornek_erkek, "sakalı temizle")
    assert degisiklik_bolgeleri(ornek_erkek, yeni) == {"alt_yuz"}
    yeni, _ = degisiklik_uygula(ornek_erkek, "sakalı temizle\nsaçları sarı yap")
    assert degisiklik_bolgeleri(ornek_erkek, yeni) == {"alt_yuz", "sac"}
    # Gözlük çıkarmak / küpe eklemek bölgesel
    assert degisiklik_bolgeleri(dict(ornek_erkek, aksesuar="glasses"),
                                dict(ornek_erkek, aksesuar="")) == {"goz"}
    assert degisiklik_bolgeleri(dict(ornek_erkek, aksesuar=""),
                                dict(ornek_erkek, aksesuar="silver hoop earring")) == {"kulak"}
    # Yaş veya ten gibi bütünsel değişiklikte bölgesel düzenleme yok
    yeni, _ = degisiklik_uygula(ornek_erkek, "10 yaş yaşlandır")
    assert degisiklik_bolgeleri(ornek_erkek, yeni) is None
    assert degisiklik_bolgeleri(ornek_erkek, dict(ornek_erkek)) is None


def test_yuz_bolge_maskesi():
    import numpy as np

    from fotorobot.motor import yuz_bolge_maskesi

    kps = np.array([[200, 260], [312, 260], [256, 320], [215, 380], [297, 380]])
    bbox = np.array([150, 160, 362, 460])
    goz = np.array(yuz_bolge_maskesi(kps, bbox, (512, 640), {"goz"})) > 0
    assert goz[260, 256] and goz[260, 200] and not goz[380, 256] and not goz[600, 256]
    alt = np.array(yuz_bolge_maskesi(kps, bbox, (512, 640), {"alt_yuz"})) > 0
    assert alt[400, 256] and alt[350, 256] and not alt[200, 256]
    sac = np.array(yuz_bolge_maskesi(kps, bbox, (512, 640), {"sac"})) > 0
    assert sac[20, 256] and sac[300, 20] and not sac[300, 256]
    assert sac[500, 20] and not sac[620, 20] and not sac[620, 256]  # omuz/kıyafet korunur
    birlesik = np.array(yuz_bolge_maskesi(kps, bbox, (512, 640), {"goz", "alt_yuz"})) > 0
    assert birlesik.sum() >= max(goz.sum(), alt.sum())


def test_degisiklik_bolgeleri_yalnizca_degisen_aksesuar(ornek_erkek):
    from fotorobot.ozellikler import degisiklik_bolgeleri

    once = dict(ornek_erkek, aksesuar="thin metal glasses, small silver hoop earring")
    sonra = dict(ornek_erkek, aksesuar="small silver hoop earring")
    assert degisiklik_bolgeleri(once, sonra) == {"goz"}
