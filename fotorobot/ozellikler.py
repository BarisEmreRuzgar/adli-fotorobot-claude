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

# Prompt'ta yeni bir 77 token'lık parça başlatan işaret (A1111'deki BREAK)
BLOK_AYIRICI = " BREAK "

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
    # Gemini (özellikle yedek model) "sarı saç"ı "yellow hair" yazabiliyor; SD buna zayıf
    # tepki veriyor ve renk değişikliği vurgusu da yalnızca "blonde"u tanıyor
    sonuc["sac"] = re.sub(r"\byellow\b", "blonde", sonuc["sac"], flags=re.IGNORECASE)
    if sonuc["cinsiyet"] == "female" and _sakal_var_mi(sonuc["sakal"]):
        sonuc["sakal"] = ""
    return sonuc


def _yalniz_kirli_sakal_mi(sakal):
    s = (sakal or "").lower()
    return "stubble" in s and not any(k in s for k in ("beard", "goatee", "mustache", "moustache"))


# "Kirli sakal" isteğinde model keçi sakalı + dolgun bıyık çiziyordu (gerileme testi, esmer senaryosu)
KIRLI_SAKAL_IPUCU = "visible dark short stubble evenly covering cheeks, jaw and upper lip, unshaven"
KIRLI_SAKAL_NEGATIFI = "goatee, full beard, long beard, thick mustache, handlebar mustache"


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


SAC_RENKLERI = {"blonde", "golden", "platinum", "red", "ginger", "auburn", "black", "brown", "chestnut",
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


# Aşama 2'de değişen alan → yeniden çizilecek yüz bölgesi. Listede olmayan bir alan
# (yaş, ten, yüz şekli, çene, belirgin iz...) tüm yüzü etkilediği için bölgesel
# düzenleme yapılmaz.
ALAN_BOLGELERI = {"sakal": "alt_yuz", "dudak": "agiz", "goz": "goz", "kas": "goz",
                  "burun": "burun", "sac": "sac"}


def _aksesuar_bolgeleri(aksesuar):
    bolgeler = set()
    for a in [p.strip().lower() for p in (aksesuar or "").split(",") if p.strip()]:
        if _iceriyor(a, "glasses", "spectacles", "eyewear", "eye patch"):
            bolgeler.add("goz")
        elif _iceriyor(a, "earring", "ear "):
            bolgeler.add("kulak")
        elif _iceriyor(a, "hat", "cap", "beanie", "hood", "helmet", "headscarf", "bandana"):
            bolgeler.add("sac")
        else:
            return None
    return bolgeler


def degisiklik_bolgeleri(onceki, yeni):
    """Aşama 2'de yalnızca hangi yüz bölgelerinin yeniden çizileceği; bütünsel bir
    değişiklik varsa (veya hiç değişiklik yoksa) None (tüm görüntü düzenlenir)."""
    bolgeler = set()
    for alan in ALANLAR:
        if alan == "cinsiyet" or (yeni.get(alan) or "") == (onceki.get(alan) or ""):
            continue
        if alan == "aksesuar":
            # Yalnızca eklenen veya kaldırılan aksesuarların bölgesi değişir; dokunulmayan
            # (ör. gözlük çıkarılırken kalan küpe) bölgeye girmez
            eski = {a.strip() for a in (onceki.get("aksesuar") or "").split(",") if a.strip()}
            yeni_a = {a.strip() for a in (yeni.get("aksesuar") or "").split(",") if a.strip()}
            degisen = _aksesuar_bolgeleri(", ".join(sorted(eski ^ yeni_a)))
            if degisen is None:
                return None
            bolgeler |= degisen
        elif alan in ALAN_BOLGELERI:
            bolgeler.add(ALAN_BOLGELERI[alan])
        else:
            return None
    return bolgeler or None


# Aşama 2'de FaceID kaynak yüzü geri çektiği için sakal ekleme (0/3, kirli sakal) ve yaşlandırma
# (0/3) uygulanmıyordu. Bu değişiklikler FaceID kapalıyken yalnızca ilgili yüz bölgesine çizilir;
# maskenin dışı birebir korunduğu için kimlik bölge dışından gelir (yerel test 2026-10-22:
# sakal 3/3, yaş 3/3, gerçekçilik 4/5).
SAKAL_EKLEME_GUCU = 0.95
# Sakal eklemede denenecek tohum sayısı (renk + kimliğe göre en iyisi seçilir; Motor.yerel_ciz_sec).
# Yaşlandırmada kullanılmaz: kimliğe göre seçim en az yaşlananı seçiyordu (3/3 → 2/3)
YEREL_ADAY = 3
# Sakal çizimi burnu da değiştiriyordu; kaynağın burnu geri konunca kimlik 0.66-0.67 → 0.75 (sakal_analiz)
SAKAL_KORU = ("nose",)
YAS_GUCU = 0.55
# Yaşlandırmada düşük güçlü FaceID (deney yf1/yf2: kapalı id 0.71; 0.45 → 0.73-0.75, yaşlanma korunuyor;
# 0.5 uçtan uca 0/3, 0.6+ → 0.78-0.81 ama yaşlanma kayboluyor)
YAS_KIMLIK_GUCU = 0.45
YAS_ESIGI = 8  # bu kadar ve daha fazla yaşlandırmada yerel yaş çizimi
SAKAL_NEGATIFI = ("worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, "
                  "clean shaven, smooth chin, woman, braid, braided beard, handlebar mustache, curled mustache, "
                  "costume, fake beard, viking, wizard, shaving foam, cotton")
YAS_NEGATIFI = "worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, makeup"
# Burun: FaceID'li bölgesel düzenleme burnu hiç değiştirmiyordu (0/3, piksel farkı ~0). FaceID'siz, burun
# kökünü de kapsayan maskeyle tam güçte deney ızgarasında 2/3, ama uçtan uca testte 0/3 ve kimlik
# 0.92 → 0.60-0.66 (burun değişiyor, "kemerli" görünmüyor; önden kemer seçilmiyor). Kazanç olmadığı için
# varsayılan olarak kapalı (FOTOROBOT_LOCAL_NOSE=1 ile açılır).
BURUN_GUCU = 1.0
BURUN_KOYULASMA_SINIRI = 12.0
BURUN_IPUCLARI = {
    ("aquiline", "hooked", "roman"): "prominent bony bump on the nose bridge catching the light, "
                                     "long nose with a downward pointing tip",
    ("crooked", "broken"): "nose bridge bent to one side, asymmetric nose",
    ("bulbous",): "large rounded bulbous nose tip",
    ("upturned",): "upturned nose tip, visible nostrils",
}
BURUN_NEGATIFI = ("worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, "
                  "makeup, dirt, mud, wound, blood, bruise, dark stain")


KEMER_SOZCUKLERI = ("aquiline", "hooked", "roman")


def kemerli_burun_mu(onceki, yeni):
    """Burun kemerli/kartal burun olarak değişiyor mu (geometrik rötuşla yapılır: rotus.burun_kemerli)."""
    b = (yeni.get("burun") or "").lower()
    return b != (onceki.get("burun") or "").lower() and any(k in b for k in KEMER_SOZCUKLERI)         and not any(k in (onceki.get("burun") or "").lower() for k in KEMER_SOZCUKLERI)


def yerel_cizimler(onceki, yeni, burun=False):
    """FaceID'siz bölgesel çizimle yapılacak değişiklikler.

    Dönüş: ([{"ad", "bolgeler", "pozitif", "negatif", "guc", "cfg"}, ...], ele alınan alanlar).
    """
    cizimler, alanlar = [], set()
    erkek = yeni.get("cinsiyet") != "female"
    if erkek and not _sakal_var_mi(onceki.get("sakal")) and _sakal_var_mi(yeni.get("sakal")):
        s = yeni["sakal"].lower()
        yalniz_biyik = ("mustache" in s or "moustache" in s) and not any(
            k in s for k in ("beard", "goatee", "stubble"))
        cizimler.append({
            "ad": "bıyık" if yalniz_biyik else "sakal",
            "bolgeler": ["biyik"] if yalniz_biyik else ["alt_yuz"],
            "pozitif": f"{agirlikli(yeni['sakal'] + ', natural facial hair', 1.1)}, photo, realistic",
            "negatif": SAKAL_NEGATIFI, "guc": SAKAL_EKLEME_GUCU, "cfg": 6.5,
            "aday": YEREL_ADAY, "renk": yeni["sakal"], "koru": SAKAL_KORU,
        })
        alanlar.add("sakal")
    if int(yeni.get("yas", 0)) >= int(onceki.get("yas", 0)) + YAS_ESIGI:
        tarif = (f"{_kisi_ifadesi(yeni)}, deep forehead wrinkles, crow's feet, under-eye bags, "
                 "nasolabial folds, aged skin")
        cizimler.append({
            "ad": "yaşlanma", "bolgeler": ["yas"],
            "pozitif": f"{agirlikli(tarif, 1.2)}, photo, realistic, natural skin texture",
            "negatif": YAS_NEGATIFI, "guc": YAS_GUCU, "cfg": 7.0, "kimlik_gucu": YAS_KIMLIK_GUCU,
        })
        alanlar.add("yas")
    if burun and yeni.get("burun") and yeni.get("burun") != onceki.get("burun"):
        b = yeni["burun"].lower()
        ipucu = next((v for k, v in BURUN_IPUCLARI.items() if any(x in b for x in k)), "")
        tarif = yeni["burun"] + (", " + ipucu if ipucu else "")
        cizimler.append({
            "ad": "burun", "bolgeler": ["burun_genis"],
            "pozitif": f"{agirlikli(tarif, 1.3)}, photo, realistic, natural skin texture",
            "negatif": BURUN_NEGATIFI, "guc": BURUN_GUCU, "cfg": 8.0,
            "en_cok_koyulasma": BURUN_KOYULASMA_SINIRI,
        })
        alanlar.add("burun")
    return cizimler, alanlar


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
    if not kadin:
        # Yalnızca sakalsız yüzlerde değil: "siyah saç + beyaz sakal" testinde sakal
        # tarif edildiği halde 4 yüzün 3'ü kadın çıktı (sakal çizilmeyince erkeklik kayboluyor)
        parcalar.append(agirlikli("masculine male face, masculine facial features", ERKEKLIK_AGIRLIGI))
    parcalar += vurgular
    # Prompt anlamsal bloklara ayrılır; her blok ayrı 77 token'lık parçada kodlanır
    # (BLOK_AYIRICI). Aynı parçadaki sıfatlar birbirine sızıyordu: "koyu saç" ile
    # "beyaz bıyık" yan yana olunca bıyık da koyu, kısa saç ön kabulü de uzun saçı
    # bastırıyordu. 1: kişi + kafa yapısı + saç, 2: yüz içi + sakal + izler/aksesuar,
    # 3: çekim koşulları.
    blok1 = parcalar + [_yas_gorunumu(oz["yas"]), oz["ten"], oz["yuz_sekli"], oz["sac"]]
    blok2 = []
    # Yara izi, küpe, gözlük gibi ayırt edici izler bloğun başına alınır
    if oz["ozel"]:
        blok2.append(agirlikli(oz["ozel"], IZ_AGIRLIGI))
    if oz["aksesuar"]:
        blok2.append(_aksesuar_vurgusu(oz["aksesuar"]))
    if not kadin:
        if _yalniz_kirli_sakal_mi(oz["sakal"]):
            blok2.append(agirlikli(f"{oz['sakal']}, {KIRLI_SAKAL_IPUCU}", 1.25))
        else:
            blok2.append(oz["sakal"] or "clean shaven")
    blok2 += [oz["goz"], oz["kas"], oz["burun"], oz["dudak"], oz["cene"]]
    blok3 = []
    if not _iceriyor(oz["aksesuar"], "hoodie", "jacket", "suit", "shirt", "coat", "sweater"):
        blok3.append("wearing a plain dark crew-neck t-shirt")
    blok3 += [
        "head and shoulders portrait, entire head in frame with space above the hair, centered",
        "facing camera, looking at viewer, neutral expression",
        "plain light grey background, even studio lighting",
        "sharp focus, photorealistic, highly detailed natural skin texture, 8k uhd, dslr",
    ]
    gorulen, bloklar = set(), []
    for blok in (blok1, blok2, blok3):
        tekil = []
        for p in blok:
            p = (p or "").strip()
            if p and p not in gorulen:
                tekil.append(p)
                gorulen.add(p)
        if tekil:
            bloklar.append(", ".join(tekil))
    pozitif = BLOK_AYIRICI.join(bloklar)

    negatif = eski_negatifler + [
        "worst quality, low quality, lowres, blurry, jpeg artifacts",
        "cartoon, anime, 3d render, cgi, painting, drawing, illustration, sketch",
        "deformed, disfigured, bad anatomy, asymmetric eyes, cross-eyed, extra fingers, mutated",
        "watermark, text, logo, signature, frame, multiple people",
        "cropped head, head out of frame, close-up, shirtless, bare chest, nude",
        "smiling, open mouth, tilted head, profile view, dramatic lighting, heavy makeup",
        # Model "güzelleştirmeye" yatkın; fotorobot sıradan, kusurlu bir yüz olmalı
        "fashion model, airbrushed skin, studio glamour, beauty retouching",
        "nsfw",
    ]
    if _yas_negatifi(oz["yas"]):
        negatif.append(_yas_negatifi(oz["yas"]))
    if kadin:
        negatif.append("man, male, beard, mustache, stubble")
    else:
        negatif.append("woman, female, feminine, makeup")
        negatif.append("androgynous, feminine face, soft feminine features")
        if not _sakal_var_mi(oz["sakal"]):
            negatif.append("beard, mustache, stubble, goatee")
        elif _yalniz_kirli_sakal_mi(oz["sakal"]):
            negatif.append(KIRLI_SAKAL_NEGATIFI)
    aksesuar = oz["aksesuar"] + " " + oz["ozel"]
    if not _iceriyor(aksesuar, "glasses", "spectacles", "eyewear", "sunglasses"):
        negatif.append("glasses, sunglasses")
    if not _iceriyor(aksesuar, "hat", "cap", "beanie", "hood", "helmet", "headscarf", "hijab"):
        negatif.append("hat, cap, headwear")
    if not _iceriyor(oz["sac"], "bald", "shaved head"):
        negatif.append("bald")
    return pozitif, ", ".join(dict.fromkeys(negatif))


# Aşama 3 (bölgesel ayrıntı) için hazır ayrıntılar: Türkçe etiket → İngilizce prompt.
# Kısa nesne adları; bolge_promptu bunları ağırlıklandırır.
BOLGE_ORNEKLERI = {
    "Yara izi": "pale healed scar line, raised scar tissue",
    "Ben": "dark brown mole, beauty spot",
    "Halka küpe": "small shiny silver hoop earring pierced through the earlobe",
    "Dövme": "black ink tattoo",
    "Çil": "brown freckles",
    "Derin kırışıklık": "deep wrinkles",
    "Kırık burun kemeri": "crooked nose bridge, previously broken nose",
}
# Ayrıntının ağırlığı. Colab testi (aynı kaynak/maske/tohum, ben): cilt tarifli eski prompt
# 0/3; ağırlık 1.2 → 1-2/5; 1.35 → 5/5 doğal ben (yarıçap 9); 1.6 → 6/6 ama iri, yara gibi.
# Cilt tarifleri modeli düz cilde itiyor, ağırlıksız nesne kayboluyor.
BOLGE_AGIRLIGI = 1.35


# Aşama 1 prompt'unda yazılan ben neredeyse hiç çizilmiyor (Colab testi, 4 tohum: yanakta
# 0/4, kaş üstünde 0/4). Yeri belli benler prompt'tan çıkarılıp üretilen yüze, yüz
# noktalarından hesaplanan yere Aşama 3 ile çizilir (aynı testte 4/4 ve 4/4 doğru yerde).
BEN_SOZCUKLERI = ("mole", "beauty mark", "beauty spot")
BEN_YERLERI = (  # (anahtar sözcükler, yer); sıra önemli: özelden genele
    (("eyebrow", "brow"), "kas_ustu"),
    (("upper lip", "above the lip", "above lip", "above his lip", "above her lip"), "dudak_ustu"),
    (("cheek",), "yanak"),
    (("chin",), "cene"),
    (("forehead",), "alin"),
    (("nose",), "burun"),
)
# Bu yerlerde taraf (sağ/sol) bilinmeden nokta seçilemez
TARAF_GEREKEN = {"kas_ustu", "yanak"}


def ben_konumlari(ozel):
    """`ozel` alanındaki yeri belli benleri ayıklar.

    Döndürür: (kalan_ozel, [(yer, taraf), ...]). taraf kişinin kendi sağı/solu
    ("sag"/"sol") ya da None. Yeri anlaşılamayan ben `kalan_ozel`de kalır.
    """
    kalan, benler = [], []
    for parca in re.split(r",|;|\band\b", ozel or ""):
        p = parca.strip()
        k = p.lower()
        if not p:
            continue
        if not any(b in k for b in BEN_SOZCUKLERI):
            kalan.append(p)
            continue
        yer = next((y for sozcukler, y in BEN_YERLERI if any(x in k for x in sozcukler)), None)
        taraf = "sag" if re.search(r"\bright\b", k) else "sol" if re.search(r"\bleft\b", k) else None
        if yer is None or (yer in TARAF_GEREKEN and taraf is None):
            kalan.append(p)
        else:
            benler.append((yer, taraf))
    return ", ".join(kalan), benler


def bolge_promptu(detay):
    """Aşama 3 için kısa (pozitif, negatif) prompt çifti.

    Yalnızca maskelenen bölge yeniden çizildiği için prompt yalnızca eklenecek
    ayrıntıyı (ağırlıklı) söyler; "cilt" gibi bölgeyi tarif eden sözcükler
    modeli mevcut düz cildi yeniden çizmeye ittiği için kullanılmaz.
    """
    detay = " ".join((detay or "").split()).strip(" ,.")
    pozitif = ", ".join(p for p in [agirlikli(detay, BOLGE_AGIRLIGI), "photo, realistic"] if p)
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
