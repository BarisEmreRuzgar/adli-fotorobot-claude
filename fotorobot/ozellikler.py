"""Yüz özellikleri sözlüğü ve Stable Diffusion prompt üretimi.

Özellikler her zaman İngilizce değerler taşır (SD modeli İngilizce prompt
bekler); anahtarlar Türkçedir.
"""

import copy
import re

ALANLAR = (
    "cinsiyet", "yas", "ten", "yuz_sekli", "sac", "kas", "goz",
    "burun", "dudak", "cene", "sakal", "aksesuar", "ozel",
)

ALAN_ETIKETLERI = {
    "cinsiyet": "Cinsiyet",
    "yas": "Yaş",
    "ten": "Ten",
    "yuz_sekli": "Yüz şekli",
    "sac": "Saç",
    "kas": "Kaş",
    "goz": "Göz",
    "burun": "Burun",
    "dudak": "Dudak",
    "cene": "Çene",
    "sakal": "Sakal/Bıyık",
    "aksesuar": "Aksesuar",
    "ozel": "Belirgin iz",
}

VARSAYILAN = {
    "cinsiyet": "male",
    "yas": 30,
    "ten": "light olive skin",
    "yuz_sekli": "oval face",
    "sac": "short dark brown hair",
    "kas": "medium thick dark eyebrows",
    "goz": "dark brown eyes",
    "burun": "straight medium nose",
    "dudak": "medium lips",
    "cene": "defined jawline",
    "sakal": "clean shaven",
    "aksesuar": "",
    "ozel": "",
}

YAS_ALT, YAS_UST = 14, 90

# Ayırt edici izler küçük ayrıntılar olduğu için model tarafından kolayca yok sayılır
IZ_AGIRLIGI = 1.3
AKSESUAR_AGIRLIGI = 1.2
# Sakal, erkek yüzün en güçlü işaretidir; sakalsız (özellikle sakal silinip saç
# sarıya boyanınca) yüzler kadınsılaşıyordu, bu yüzden erkeklik ayrıca vurgulanır
ERKEKLIK_AGIRLIGI = 1.2

_AGIRLIK_KALIBI = re.compile(r"\(([^():]+):\s*([0-9]*\.?[0-9]+)\)")


def agirlikli(metin, agirlik):
    """`(metin:agirlik)` sözdizimiyle ağırlıklandırılmış prompt parçası."""
    return f"({metin}:{agirlik:g})" if metin and agirlik != 1 else metin


def agirliklari_ayir(prompt):
    """Prompt'u (metin, ağırlık) parçalarına ayırır; işaretsiz metnin ağırlığı 1.0'dır."""
    parcalar, konum = [], 0
    for m in _AGIRLIK_KALIBI.finditer(prompt):
        if m.start() > konum:
            parcalar.append((prompt[konum:m.start()], 1.0))
        parcalar.append((m.group(1), float(m.group(2))))
        konum = m.end()
    if konum < len(prompt):
        parcalar.append((prompt[konum:], 1.0))
    return [(m, a) for m, a in parcalar if m]


def varsayilan_ozellikler(cinsiyet=None):
    oz = copy.deepcopy(VARSAYILAN)
    if cinsiyet == "female":
        oz["cinsiyet"] = "female"
        oz["sac"] = "long dark brown hair"
        oz["kas"] = "thin arched dark eyebrows"
        oz["sakal"] = ""
    return oz


def _yas_normalize(deger, varsayilan):
    if isinstance(deger, (int, float)):
        yas = int(round(deger))
    else:
        rakamlar = [int(p) for p in "".join(
            c if c.isdigit() else " " for c in str(deger or "")
        ).split()]
        if not rakamlar:
            return varsayilan
        # "25-30" gibi aralıklarda ortalamayı al
        yas = round(sum(rakamlar[:2]) / len(rakamlar[:2]))
    return max(YAS_ALT, min(YAS_UST, yas))


def _cinsiyet_normalize(deger):
    d = str(deger or "").strip().lower()
    if d in ("female", "woman", "f", "kadın", "kadin", "girl", "kız"):
        return "female"
    return "male"


def normalize(ham, temel=None):
    """Gemini'den veya analizciden gelen sözlüğü güvenli hale getirir.

    Eksik alanlar `temel` (yoksa varsayılan) değerlerle tamamlanır, bilinmeyen
    alanlar atılır, tüm değerler string (yaş hariç) olur.
    """
    ham = ham if isinstance(ham, dict) else {}
    cinsiyet = _cinsiyet_normalize(ham.get("cinsiyet", (temel or {}).get("cinsiyet")))
    sonuc = copy.deepcopy(temel) if temel else varsayilan_ozellikler(cinsiyet)
    for alan in ALANLAR:
        if alan not in ham or ham[alan] is None:
            continue
        if alan == "yas":
            sonuc["yas"] = _yas_normalize(ham["yas"], sonuc.get("yas", VARSAYILAN["yas"]))
        elif alan == "cinsiyet":
            sonuc["cinsiyet"] = cinsiyet
        else:
            sonuc[alan] = " ".join(str(ham[alan]).replace("\n", " ").split()).strip(" ,.")
    for alan in ALANLAR:
        sonuc.setdefault(alan, VARSAYILAN[alan])
    sonuc["yas"] = _yas_normalize(sonuc["yas"], VARSAYILAN["yas"])
    if sonuc["cinsiyet"] == "female" and _sakal_var_mi(sonuc["sakal"]):
        sonuc["sakal"] = ""
    return sonuc


def _sakal_var_mi(sakal):
    s = (sakal or "").lower()
    if not s or "clean" in s or ("shaven" in s and "unshaven" not in s):
        return False
    return any(k in s for k in ("beard", "stubble", "goatee", "mustache", "moustache"))


def _kisi_ifadesi(oz):
    yas = oz["yas"]
    if oz["cinsiyet"] == "female":
        kisi = "teenage girl" if yas < 18 else "woman"
    else:
        kisi = "teenage boy" if yas < 18 else "man"
    return f"{yas}-year-old {kisi}"


def _iceriyor(metin, *kelimeler):
    m = (metin or "").lower()
    return any(k in m for k in kelimeler)


def _yas_gorunumu(yas):
    """SD modelleri sayısal yaşa zayıf tepki verir; görünüm ifadeleriyle destekle."""
    if yas < 18:
        return "teenage features"
    if yas < 26:
        return "young adult, youthful smooth skin"
    if yas < 40:
        return ""
    if yas < 55:
        return "middle-aged, fine wrinkles, crow's feet"
    if yas < 70:
        return "older person, wrinkled skin, aged face, sagging skin"
    return "elderly person, deeply wrinkled skin, very old face"


def _yas_negatifi(yas):
    if yas >= 45:
        return "young, youthful, smooth skin, teenager"
    if yas < 25:
        return "old, elderly, wrinkles, aged"
    return ""


def _yuzde_olmayan_kelimeler(eski, yeni):
    """Eski değerde olup yenide olmayan anlamlı kelimeler (negatif prompt için)."""
    bos = {"and", "with", "a", "the", "of", "on", "in", "hair", "eyes", "face", "skin", "nose", "lips"}
    yeni_k = set(re.findall(r"[a-z-]+", (yeni or "").lower()))
    return [k for k in re.findall(r"[a-z-]+", (eski or "").lower()) if k not in yeni_k and k not in bos]


SAC_RENKLERI = {"blonde", "platinum", "red", "ginger", "auburn", "black", "brown", "chestnut",
                "grey", "gray", "white", "silver", "greying"}

DEGISIM_ALANLARI = ("sac", "sakal", "goz", "aksesuar", "ozel", "ten", "kas", "yuz_sekli", "burun",
                    "dudak", "cene")


def _aksesuar_vurgusu(aksesuar):
    parcalar = []
    for a in [p.strip() for p in aksesuar.split(",") if p.strip()]:
        if "glasses" in a and "sun" not in a:
            # "Belirgin çerçeve" vurgusu ince/metal gözlükleri kalın kemik çerçeveye çeviriyor;
            # onlarda gözlüğün görünmesini yalnızca ağırlık sağlar
            ince = _iceriyor(a, "thin", "metal", "wire", "rimless", "frameless")
            parcalar.append(f"wearing {a}" if ince
                            else f"wearing {a}, eyeglasses with clearly visible frames on the face")
        else:
            parcalar.append(f"wearing {a}")
    return ", ".join(agirlikli(p, AKSESUAR_AGIRLIGI) for p in parcalar)


def _fark_vurgulari(oz, onceki, kadin):
    """`onceki`den `oz`e değişen alanlar için (vurgular, negatifler)."""
    vurgular, negatifler = [], []
    for alan in DEGISIM_ALANLARI:
        if (oz.get(alan) or "") == (onceki.get(alan) or ""):
            continue
        if oz.get(alan):
            vurgular.append(oz[alan] if alan != "aksesuar" else _aksesuar_vurgusu(oz[alan]))
            if alan == "sac":
                yeni_renk = [k for k in _yuzde_olmayan_kelimeler(oz[alan], onceki.get(alan))
                             if k in SAC_RENKLERI]
                if yeni_renk:
                    vurgular.append(f"dyed {' '.join(yeni_renk)} hair")
        elif alan == "sakal" and not kadin:
            vurgular.append("clean shaven")
        eski = _yuzde_olmayan_kelimeler(onceki.get(alan), oz.get(alan))
        if eski:
            ad = {"sac": "hair", "goz": "eyes", "ten": "skin"}.get(alan, "")
            negatifler.append(" ".join(eski + ([ad] if ad else [])))
    if oz["yas"] != onceki.get("yas") and _yas_gorunumu(oz["yas"]):
        vurgular.append(_yas_gorunumu(oz["yas"]))
    return vurgular, negatifler


def prompt_olustur(oz, onceki=None, referans=None):
    """Özelliklerden (pozitif, negatif) prompt çifti üretir.

    En ayırt edici özellikler başa yazılır; uzun prompt'lar üretim tarafında
    parçalara bölünerek kodlandığı için 77 token sınırı sorun olmaz.

    Aşama 2 için:
    - `onceki`: düzenlenen kaynağın özellikleri. Bu adımda değişenler prompt'un
      en başına alınır, eski halleri negatif prompt'a yazılır.
    - `referans`: kimliğin alındığı ilk yüzün özellikleri. Kimlik modeli
      görünümü bu yüze geri çekmeye çalıştığından, önceki adımlarda yapılan
      değişiklikler de (ör. sarı saç) vurgulanır; böylece kalıcı olurlar.
    """
    kadin = oz["cinsiyet"] == "female"
    vurgular, eski_negatifler = [], []
    for temel in (onceki, referans):
        if temel:
            v, n = _fark_vurgulari(oz, temel, kadin)
            vurgular += v
            eski_negatifler += n

    parcalar = [f"RAW photo, frontal passport style portrait of a {_kisi_ifadesi(oz)}"]
    if not kadin and not _sakal_var_mi(oz["sakal"]):
        parcalar.append(agirlikli("masculine male face, masculine facial features", ERKEKLIK_AGIRLIGI))
    parcalar += vurgular
    # Yara izi, küpe, gözlük gibi ayırt edici izler ilk parçada kalsın diye öne alınır
    if oz["ozel"]:
        parcalar.append(agirlikli(oz["ozel"], IZ_AGIRLIGI))
    if oz["aksesuar"]:
        parcalar.append(_aksesuar_vurgusu(oz["aksesuar"]))
    parcalar += [_yas_gorunumu(oz["yas"]), oz["ten"], oz["yuz_sekli"], oz["sac"]]
    if not kadin:
        parcalar.append(oz["sakal"] or "clean shaven")
    parcalar += [oz["goz"], oz["kas"], oz["burun"], oz["dudak"], oz["cene"]]
    if not _iceriyor(oz["aksesuar"], "hoodie", "jacket", "suit", "shirt", "coat", "sweater"):
        parcalar.append("wearing a plain dark crew-neck t-shirt")
    parcalar += [
        "head and shoulders portrait, entire head in frame with space above the hair, centered",
        "facing camera, looking at viewer, neutral expression",
        "plain light grey background, even studio lighting",
        "sharp focus, photorealistic, highly detailed natural skin texture, 8k uhd, dslr",
    ]
    tekil = []
    for p in parcalar:
        p = (p or "").strip()
        if p and p not in tekil:
            tekil.append(p)
    pozitif = ", ".join(tekil)

    negatif = eski_negatifler + [
        "worst quality, low quality, lowres, blurry, jpeg artifacts",
        "cartoon, anime, 3d render, cgi, painting, drawing, illustration, sketch",
        "deformed, disfigured, bad anatomy, asymmetric eyes, cross-eyed, extra fingers, mutated",
        "watermark, text, logo, signature, frame, multiple people",
        "cropped head, head out of frame, close-up, shirtless, bare chest, nude",
        "smiling, open mouth, tilted head, profile view, dramatic lighting, heavy makeup",
        "nsfw",
    ]
    if _yas_negatifi(oz["yas"]):
        negatif.append(_yas_negatifi(oz["yas"]))
    if kadin:
        negatif.append("man, male, beard, mustache, stubble")
    else:
        negatif.append("woman, female, feminine, makeup")
        if not _sakal_var_mi(oz["sakal"]):
            negatif.append("beard, mustache, stubble, goatee")
            negatif.append("androgynous, feminine face, soft feminine features")
    aksesuar = oz["aksesuar"] + " " + oz["ozel"]
    if not _iceriyor(aksesuar, "glasses", "spectacles", "eyewear", "sunglasses"):
        negatif.append("glasses, sunglasses")
    if not _iceriyor(aksesuar, "hat", "cap", "beanie", "hood", "helmet", "headscarf", "hijab"):
        negatif.append("hat, cap, headwear")
    if not _iceriyor(oz["sac"], "bald", "shaved head"):
        negatif.append("bald")
    return pozitif, ", ".join(dict.fromkeys(negatif))


# Aşama 3 (bölgesel ayrıntı) için hazır ayrıntılar: Türkçe etiket → İngilizce prompt.
# Colab testinde "thin pale scar" gibi sade tarifler bölgeyi düz cilt olarak yeniden
# çizdi; "prominent visible ..." gibi belirgin tarifler izi gerçekten çizdirdi.
BOLGE_ORNEKLERI = {
    "Yara izi": "prominent visible pale healed scar line, raised scar tissue on the skin",
    "Ben": "prominent dark brown mole on the skin",
    "Halka küpe": "small shiny silver hoop earring pierced through the earlobe",
    "Dövme": "visible black ink tattoo on the skin",
    "Çil": "many visible brown freckles on the skin",
    "Derin kırışıklık": "deep prominent wrinkles in the skin",
    "Kırık burun kemeri": "crooked nose bridge, previously broken nose",
}


def bolge_promptu(detay):
    """Aşama 3 için kısa (pozitif, negatif) prompt çifti.

    Yalnızca maskelenen bölge yeniden çizildiği için prompt o bölgeyi
    tarif eder; yüzün geri kalanını anlatmaya gerek yoktur ve kısa prompt
    77 token'lık tek parçada kalır.
    """
    detay = " ".join((detay or "").split()).strip(" ,.")
    pozitif = ", ".join(p for p in [
        detay, "RAW photo, close-up of human skin, photorealistic",
        "natural skin texture, sharp focus, even studio lighting",
    ] if p)
    negatif = ("worst quality, low quality, blurry, jpeg artifacts, cartoon, painting, drawing, "
               "3d render, deformed, text, watermark, makeup")
    return pozitif, negatif


def ozet_metni(oz):
    """Raporlarda gösterilecek okunabilir özet."""
    satirlar = []
    for alan in ALANLAR:
        deger = oz.get(alan)
        if alan == "cinsiyet":
            deger = "Kadın" if deger == "female" else "Erkek"
        if deger in ("", None):
            deger = "—"
        satirlar.append(f"  {ALAN_ETIKETLERI[alan]:<12}: {deger}")
    return "\n".join(satirlar)
