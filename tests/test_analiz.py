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
    assert poz.index("scar") < poz.index(oz["ten"])
    assert poz.index("hoop earring") < poz.index(oz["sac"])


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
