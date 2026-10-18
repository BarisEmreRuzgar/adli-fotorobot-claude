"""Gemini'ye erişilemediğinde kullanılan kural tabanlı Türkçe ifade analizcisi.

Amaç tanık ifadesinden en sık geçen yüz özelliklerini (yaş, cinsiyet, saç,
göz, sakal, ten, aksesuar, iz...) yakalamaktır. Gemini kadar esnek değildir
ama internet bağlantısı veya API anahtarı olmadan da sistemi kullanılabilir
kılar. Aynı fonksiyonlar Aşama 2'deki değişiklik komutlarını da yorumlar.
"""

import re

from .ozellikler import YAS_ALT, YAS_UST, normalize


def tr_kucuk(metin):
    return (metin or "").replace("I", "ı").replace("İ", "i").lower()


# ── Sözlükler ───────────────────────────────────────────────────────────
# (türkçe ifade, ingilizce karşılık). Uzun ifadeler önce denenir.

RENK_SAC = [
    ("koyu kahverengi", "dark brown"), ("koyu kahve", "dark brown"),
    ("açık kahverengi", "light brown"), ("açık kahve", "light brown"),
    ("kestane", "chestnut brown"), ("kahverengi", "brown"), ("kahve", "brown"),
    ("simsiyah", "jet black"), ("siyah", "black"), ("kara", "black"),
    ("sarışın", "blonde"), ("platin", "platinum blonde"), ("sarı", "blonde"),
    ("kumral", "light brown"), ("kızıl", "red ginger"), ("bakır", "auburn"),
    ("kırlaşmış", "greying"), ("kırçıl", "salt and pepper"), ("kır", "grey"),
    ("gri", "grey"), ("bembeyaz", "white"), ("beyaz", "white"), ("ak", "grey"),
    ("boyalı", "dyed"),
]
RENK_GOZ = [
    ("koyu kahverengi", "dark brown"), ("koyu kahve", "dark brown"),
    ("açık kahverengi", "light brown"), ("açık kahve", "light brown"),
    ("kahverengi", "brown"), ("kahve", "brown"), ("siyah", "very dark brown"),
    ("kara", "very dark brown"), ("mavi", "blue"), ("yeşil", "green"),
    ("ela", "hazel"), ("gri", "grey"), ("bal rengi", "amber"), ("bal", "amber"),
]
SAC_UZUNLUK = [
    ("çok kısa", "very short"), ("omuz hizasında", "shoulder-length"),
    ("omuzlarına kadar", "shoulder-length"), ("orta uzunlukta", "medium-length"),
    ("orta boy", "medium-length"), ("kısa", "short"), ("uzun", "long"),
]
SAC_DOKU = [
    ("kıvırcık", "curly"), ("kıvır kıvır", "tightly curly"), ("dalgalı", "wavy"),
    ("düz", "straight"), ("dağınık", "messy"), ("seyrek", "thinning"),
    ("gür", "thick"), ("yağlı", "greasy"), ("jöleli", "gelled"),
]
SAC_STIL = [
    ("at kuyruğu", "in a ponytail"), ("topuz", "in a bun"), ("örgü", "braided"),
    ("geriye taranmış", "slicked back"), ("geriye yatırılmış", "slicked back"),
    ("arkaya taranmış", "slicked back"), ("yandan ayrık", "side-parted"),
    ("yana taranmış", "side-parted"), ("ortadan ayrık", "middle-parted"),
    ("perçemli", "with bangs"), ("kakül", "with bangs"), ("kâkül", "with bangs"),
    ("perçem", "with bangs"), ("yanları kısa", "with short faded sides"),
    ("fade", "with fade haircut"), ("asker tıraşı", "buzz cut"),
    ("asker traşı", "buzz cut"), ("kazınmış", "buzz cut"), ("sıfır", "buzz cut"),
    ("rasta", "dreadlocks"), ("afro", "afro"), ("mohawk", "mohawk"),
]
GOZ_SEKIL = [
    ("çekik", "narrow almond-shaped"), ("badem", "almond-shaped"),
    ("iri", "large"), ("büyük", "large"), ("küçük", "small"),
    ("yuvarlak", "round"), ("çukur", "deep-set"), ("derin", "deep-set"),
    ("sarkık", "droopy hooded"), ("düşük", "droopy hooded"),
    ("kısık", "squinting narrow"), ("patlak", "bulging"),
    ("yakın", "close-set"), ("aralıklı", "wide-set"),
]
KAS = [
    ("çatık", "furrowed thick"), ("birleşik", "unibrow"), ("kalın", "thick"),
    ("gür", "bushy"), ("ince", "thin"), ("kemerli", "arched"), ("yay", "arched"),
    ("düz", "straight"), ("seyrek", "sparse"), ("alınmış", "thin plucked"),
]
BURUN = [
    ("kartal", "aquiline hooked"), ("gaga", "aquiline hooked"),
    ("kemerli", "aquiline hooked"), ("kalkık", "upturned"), ("kırık", "crooked broken"),
    ("eğri", "crooked"), ("basık", "flat wide"), ("yayvan", "wide flat"),
    ("geniş", "wide"), ("kalın", "bulbous"), ("iri", "large"), ("büyük", "large"),
    ("kocaman", "very large"), ("küçük", "small"), ("sivri", "pointed"),
    ("ince", "narrow"), ("uzun", "long"), ("düz", "straight"), ("kısa", "short"),
]
DUDAK = [
    ("ince", "thin"), ("kalın", "full thick"), ("dolgun", "full"), ("etli", "full"),
    ("geniş", "wide"), ("küçük", "small"), ("çatlak", "chapped"),
    ("sarkık", "downturned"),
]
CENE = [
    ("gamzeli", "cleft chin"), ("yarık", "cleft chin"), ("çift", "double chin"),
    ("gıdı", "double chin"), ("sivri", "pointed chin"), ("yuvarlak", "rounded chin"),
    ("köşeli", "strong square jawline"), ("kare", "strong square jawline"),
    ("belirgin", "strong defined jawline"), ("güçlü", "strong defined jawline"),
    ("geniş", "wide jaw"), ("çıkık", "protruding chin"), ("geri", "weak receding chin"),
    ("küçük", "small chin"), ("uzun", "long chin"),
]
YUZ = [
    ("oval", "oval face"), ("yuvarlak", "round face"), ("kare", "square face"),
    ("köşeli", "angular square face"), ("uzun", "long narrow face"),
    ("üçgen", "heart-shaped face"), ("kalp", "heart-shaped face"),
    ("çökük", "gaunt face with hollow cheeks"), ("zayıf", "thin face"),
    ("ince", "thin narrow face"), ("dolgun", "full round face"),
    ("tombul", "chubby round face"), ("geniş", "wide face"),
    ("sert", "harsh angular face"), ("yorgun", "tired face"),
]
TEN = [
    ("esmer", "tan brown skin"), ("buğday", "olive skin"), ("kumral", "light olive skin"),
    ("bronz", "tanned skin"), ("güneşten yanmış", "sunburnt tanned skin"),
    ("solgun", "pale skin"), ("bembeyaz", "very fair pale skin"),
    ("beyaz", "fair skin"), ("açık", "fair skin"), ("pembe", "fair rosy skin"),
    ("kırmızı", "ruddy skin"), ("kızarık", "ruddy skin"), ("koyu", "dark brown skin"),
    ("siyahi", "dark brown skin"), ("zenci", "dark brown skin"),
]

# Hedef kelime kökleri: (alan, kök regex'i)
HEDEFLER = [
    ("sac", r"saç(?!ma)\w*"),
    ("goz", r"göz(?!lü[kğ])\w*"),
    ("kas", r"kaş(?!ık)\w*"),
    ("burun", r"bur(?:un|nu)\w*"),
    ("dudak", r"duda[kğ]\w*"),
    ("cene", r"çene\w*"),
    ("yuz", r"yüz(?![üm]?n?[dt][ea])\w*"),
    ("ten", r"(?:ten|teni|tenli|tenini|cilt|cildi|ciltli)\b"),
    ("sakal", r"sakal\w*"),
]
ALAN_SOZLUK = {
    "goz": GOZ_SEKIL, "kas": KAS, "burun": BURUN, "dudak": DUDAK,
    "cene": CENE, "yuz": YUZ, "ten": TEN,
}


def _bul(metin, sozluk, tek=False):
    """Sözlükteki ifadeleri metinde arar; çakışmayan eşleşmeleri sırayla döndürür."""
    bulunan, dolu = [], []
    for tr, en in sorted(sozluk, key=lambda x: -len(x[0])):
        for m in re.finditer(r"(?<!\w)" + re.escape(tr) + r"\w*", metin):
            a, b = m.span()
            if any(a < y and x < b for x, y in dolu):
                continue
            dolu.append((a, b))
            if en not in [e for _, e in bulunan]:
                bulunan.append((a, en))
            break
    bulunan.sort()
    sonuc = [e for _, e in bulunan]
    return sonuc[:1] if tek else sonuc


def _cumlecikler(metin):
    return [c.strip() for c in re.split(r"[.,;:!?\n()]+|\s+ve\s+|\s+ama\s+|\s+fakat\s+", metin) if c.strip()]


_KOKLER = ("saç", "göz", "kaş", "burun", "burn", "dudak", "dudağ", "çene",
           "yüz", "ten", "cilt", "cild", "sakal")


def _iyelik_mi(kelime):
    k = kelime.strip("'’\"")
    for kok in sorted(_KOKLER, key=len, reverse=True):
        if k.startswith(kok):
            ek = k[len(kok):]
            return bool(re.fullmatch(r"(?:ı|i|u|ü|sı|si|su|sü|ları|leri|lar|ler)", ek))
    return False


def _pencereler(cumlecik, alan_regex, once=4, sonra=3):
    """Hedef kelimenin etrafındaki kelime penceresini döndürür."""
    kelimeler = cumlecik.split()
    pencereler = []
    for i, k in enumerate(kelimeler):
        if re.fullmatch(alan_regex, k.strip("'’\"")):
            bas = max(0, i - once)
            # Başka bir hedef kelimeye kadar geri git
            for j in range(i - 1, bas - 1, -1):
                if any(re.fullmatch(r, kelimeler[j]) for _, r in HEDEFLER):
                    bas = j + 1
                    break
            son = min(len(kelimeler), i + 1 + sonra)
            for j in range(i + 1, son):
                if any(re.fullmatch(r, kelimeler[j]) for _, r in HEDEFLER):
                    son = j
                    break
            # "saçları siyah" gibi sonradan gelen sıfatlar yalnızca iyelik/çoğul
            # ekiyle gelir; "saçlı uzun boylu" gibi -lı ekinde sonrası alınmaz.
            if not _iyelik_mi(k):
                son = i + 1
            pencereler.append(" ".join(kelimeler[bas:i] + kelimeler[i + 1:son]))
    return pencereler


def _alan_pencereleri(metin, alan):
    regex = dict(HEDEFLER)[alan]
    sonuc = []
    for c in _cumlecikler(metin):
        sonuc += _pencereler(c, regex)
    return sonuc


# ── Alan çıkarıcıları ──────────────────────────────────────────────────

def _cinsiyet(m):
    kadin = len(re.findall(r"\b(?:kadın|bayan|kız|hanım|teyze|abla|anne|kadını)\w*", m))
    erkek = len(re.findall(r"\b(?:erkek|adam|bay|oğlan|delikanlı|amca|abi|ağabey|dayı)\w*", m))
    erkek += len(re.findall(r"\b(?:sakal|bıyık)\w*", m))
    if kadin == erkek == 0:
        return None
    return "female" if kadin > erkek else "male"


def _yas(m):
    m = m.replace("’", "'")
    aralik = re.search(r"(\d{1,2})\s*(?:-|–|ile|ila|veya|ya da|/)\s*(\d{1,2})\W*\s*(?:yaş|\w*\s*yaş)", m)
    if aralik:
        return round((int(aralik.group(1)) + int(aralik.group(2))) / 2)
    onlu = re.search(r"(\d)0\s*'?\s*(?:lı|li|lu|lü|lar|ler)\w*\s*(?:yaş\w*\s*)?(baş|orta|son)?", m)
    if onlu and "yaş" in m[onlu.start():onlu.end() + 15]:
        taban = int(onlu.group(1)) * 10
        ek = {"baş": 2, "orta": 5, "son": 8}.get(onlu.group(2) or "", 5)
        return taban + ek
    tek = re.search(r"(\d{1,2})\s*(?:'?\s*(?:lık|lik|luk|lük))?\s*yaş", m)
    if tek:
        return int(tek.group(1))
    kelime = [
        (r"orta\s*yaş", 45), (r"yaşlıca", 55), (r"(?:çok\s+)?yaşlı|ihtiyar|ak\s*saçlı", 65),
        (r"amca|teyze", 58), (r"ergen|lise\w*|ortaokul", 16), (r"delikanlı|üniversite\w*", 20),
        (r"genç", 24), (r"abla|abi|ağabey", 32), (r"emekli", 63),
    ]
    for kalip, yas in kelime:
        if re.search(r"\b(?:" + kalip + r")", m):
            return yas
    return None


def _sac(m):
    if re.search(r"\b(?:kel|dazlak|saçsız)\b|saçları?\s*(?:kazınmış|yok)|kafası\s*kazınmış", m):
        return "bald shaved head"
    pencereler = _alan_pencereleri(m, "sac")
    if not pencereler and not re.search(r"sarışın|kumral|kızıl", m):
        return None
    metin = " ".join(pencereler)
    if re.search(r"\b(?:sarışın)", m) and "sarışın" not in metin:
        metin += " sarışın"
    uzunluk = _bul(metin, SAC_UZUNLUK, tek=True)
    doku = _bul(metin, SAC_DOKU)
    renk = _bul(metin, RENK_SAC)
    stil = _bul(metin, SAC_STIL)
    if not (uzunluk or doku or renk or stil):
        return None
    parcalar = uzunluk + doku + renk[:2]
    sonuc = " ".join(parcalar + ["hair"]).strip()
    if "buzz cut" in stil:
        sonuc = " ".join(renk[:1] + ["buzz cut hair"])
        stil = [s for s in stil if s != "buzz cut"]
    if stil:
        sonuc += " " + " ".join(stil)
    if re.search(r"(?:saç\w*\s*)?(?:dökülmüş|seyrelmiş|açılmış|geri\s*çekilmiş)|alnı\s*açık|keli\s*açıl", m):
        sonuc += ", receding hairline"
    return sonuc


def _goz(m):
    metin = " ".join(_alan_pencereleri(m, "goz"))
    if not metin:
        return None
    renk = _bul(metin, RENK_GOZ, tek=True)
    sekil = _bul(metin, GOZ_SEKIL)
    if not (renk or sekil):
        return None
    sonuc = " ".join(sekil[:2] + renk + ["eyes"])
    return sonuc


def _sozluk_alani(m, alan, son_ek):
    metin = " ".join(_alan_pencereleri(m, alan))
    if not metin:
        return None
    bulunan = _bul(metin, ALAN_SOZLUK[alan])
    if not bulunan:
        return None
    if alan in ("cene", "yuz"):
        return ", ".join(bulunan[:2])
    renk = _bul(metin, RENK_SAC, tek=True) if alan == "kas" else []
    return " ".join(bulunan[:2] + renk + [son_ek]).replace("unibrow eyebrows", "unibrow")


def _ten(m):
    metin = " ".join(_alan_pencereleri(m, "ten"))
    bulunan = _bul(metin, TEN, tek=True) if metin else []
    if not bulunan:
        # "esmer", "buğday" gibi kelimeler tek başına da teni anlatır
        bulunan = _bul(m, [t for t in TEN if t[0] in ("esmer", "buğday", "solgun", "bronz", "siyahi")], tek=True)
    return bulunan[0] if bulunan else None


def _yuz(m):
    sonuc = _sozluk_alani(m, "yuz", "face")
    if not sonuc:
        genel = _bul(m, [("kilolu", "chubby round face"), ("şişman", "chubby round face"),
                         ("tombul", "chubby round face"), ("sıska", "thin gaunt face"),
                         ("çok zayıf", "thin gaunt face")], tek=True)
        sonuc = genel[0] if genel else None
    if sonuc and re.search(r"elmacık\w*\s*(?:kemik\w*)?\s*(?:çıkık|belirgin)", m):
        sonuc += ", high cheekbones"
    return sonuc


def _sakal(m):
    if re.search(r"sakalsız|tıraşlı|traşlı|sinekkaydı|temiz\s*(?:tıraş|traş|yüz)|tıraş\s*olmuş|sakalı\s*yok", m):
        sakal = "clean shaven"
    elif re.search(r"kirli\s*sakal|\d+\s*günlük\s*sakal|hafif\s*sakal|uzamış\s*sakal|seyrek\s*sakal", m):
        sakal = "stubble"
    elif re.search(r"keçi\s*sakal", m):
        sakal = "goatee"
    elif re.search(r"(?:uzun|gür|kalın|dolgun)\s*sakal|sakal\w*\s*(?:uzun|gür)", m):
        sakal = "long thick full beard"
    elif re.search(r"(?:kısa|düzgün|bakımlı|kısa\s*kesilmiş)\s*sakal", m):
        sakal = "short trimmed beard"
    elif re.search(r"sakal", m):
        sakal = "full beard"
    else:
        sakal = None
    if sakal and sakal != "clean shaven":
        renk = _bul(" ".join(_alan_pencereleri(m, "sakal")), RENK_SAC, tek=True)
        if renk:
            sakal = f"{renk[0]} {sakal}"
    if re.search(r"bıyıksız", m):
        return sakal
    if re.search(r"pala\s*bıyık|kalın\s*bıyık|gür\s*bıyık", m):
        biyik = "thick mustache"
    elif re.search(r"ince\s*bıyık", m):
        biyik = "thin mustache"
    elif re.search(r"bıyık", m):
        biyik = "mustache"
    else:
        biyik = None
    if biyik:
        if not sakal or sakal == "clean shaven":
            return biyik
        if "beard" in sakal:
            return sakal.replace("beard", "beard and " + biyik)
        return f"{sakal} with {biyik}"
    return sakal


def _aksesuar(m):
    parcalar = []
    if not re.search(r"gözlüksüz|gözlük\w*\s*(?:yok|takmıyor)", m):
        if re.search(r"güneş\s*gözlü", m):
            parcalar.append("sunglasses")
        elif re.search(r"gözlü[kğ]", m):
            if re.search(r"kalın\s*çerçeve|siyah\s*çerçeve|kemik\s*çerçeve", m):
                parcalar.append("thick black-framed glasses")
            elif re.search(r"ince\s*çerçeve|metal\s*çerçeve|çerçevesiz", m):
                parcalar.append("thin metal-framed glasses")
            else:
                parcalar.append("glasses")
    tablo = [
        (r"küpe", "earring"), (r"hızma|burun\w*\s*piercing", "nose ring"),
        (r"piercing", "piercing"), (r"\bkep\b|beyzbol\s*şapka|siperli", "baseball cap"),
        (r"şapka", "hat"), (r"\bbere\b", "beanie"), (r"kapüşon|kapşon", "hoodie with hood up"),
        (r"başörtü|türban|eşarp|tesettür", "headscarf"), (r"kolye|zincir", "chain necklace"),
        (r"takım\s*elbise|ceket", "suit jacket"), (r"kravat", "tie"),
    ]
    for kalip, en in tablo:
        if re.search(kalip, m) and en not in parcalar:
            if en == "piercing" and "nose ring" in parcalar:
                continue
            if en == "hat" and "baseball cap" in parcalar:
                continue
            parcalar.append(en)
    return ", ".join(parcalar) or None


KONUMLAR = [
    (r"sağ\s*yana[kğ]", "on right cheek"), (r"sol\s*yana[kğ]", "on left cheek"),
    (r"yana[kğ]", "on cheek"),
    (r"sağ\s*kaş", "above right eyebrow"), (r"sol\s*kaş", "above left eyebrow"), (r"kaş", "across eyebrow"), (r"aln", "on forehead"),
    (r"çene", "on chin"), (r"duda", "on lip"), (r"göz", "near eye"),
    (r"boyn|boyun", "on neck"), (r"burn|burun", "on nose"), (r"kula", "on ear"),
]


def _konum(parca):
    for kalip, en in KONUMLAR:
        if re.search(kalip, parca):
            return " " + en
    return ""


def _ozel(m):
    parcalar = []
    for c in _cumlecikler(m):
        if re.search(r"yanık\s*iz", c):
            parcalar.append("burn scar" + _konum(c))
        elif re.search(r"yara\s*iz|yara\b|kesik\s*iz|dikiş\s*iz|\byara", c):
            parcalar.append("visible scar" + _konum(c))
        if re.search(r"dövme", c):
            parcalar.append("tattoo" + (_konum(c) or " on neck"))
        if re.search(r"\bben(?:i|leri)?\s+(?:var|vardı|bulunuyor)|\bbenli\b|(?:siyah|kahverengi|büyük|küçük|belirgin)\s+ben\b", c):
            parcalar.append("mole" + _konum(c))
        if re.search(r"doğum\s*leke", c):
            parcalar.append("birthmark" + _konum(c))
        if re.search(r"\bçil", c):
            parcalar.append("freckles")
        if re.search(r"sivilce|akne", c):
            parcalar.append("acne")
        if re.search(r"çiçek\s*bozu|delik\s*delik", c):
            parcalar.append("pockmarked skin")
        if re.search(r"kırışı", c):
            parcalar.append("wrinkles")
        if re.search(r"gamze", c) and not re.search(r"çene", c):
            parcalar.append("dimples")
        if re.search(r"morluk|morarm", c):
            parcalar.append("bruise" + _konum(c))
        if re.search(r"göz\s*altı\w*\s*(?:torba|mor|halka)|torbalı|mor\s*halka", c):
            parcalar.append("dark circles under eyes")
    tekil = []
    for p in parcalar:
        if p not in tekil:
            tekil.append(p)
    return ", ".join(tekil) or None


def ozellik_cikar(metin):
    """Metinde açıkça geçen özellikleri döndürür (geçmeyenler dahil edilmez)."""
    m = tr_kucuk(metin)
    bulunan = {
        "cinsiyet": _cinsiyet(m),
        "yas": _yas(m),
        "sac": _sac(m),
        "goz": _goz(m),
        "kas": _sozluk_alani(m, "kas", "eyebrows"),
        "burun": _sozluk_alani(m, "burun", "nose"),
        "dudak": _sozluk_alani(m, "dudak", "lips"),
        "cene": _sozluk_alani(m, "cene", ""),
        "yuz_sekli": _yuz(m),
        "ten": _ten(m),
        "sakal": _sakal(m),
        "aksesuar": _aksesuar(m),
        "ozel": _ozel(m),
    }
    return {k: v for k, v in bulunan.items() if v not in (None, "")}


def ifade_analiz(metin):
    """Tanık ifadesini özellik sözlüğüne çevirir (eksikler varsayılanla dolar)."""
    bulunan = ozellik_cikar(metin)
    return normalize(bulunan), sorted(bulunan)


# ── Aşama 2: değişiklik komutları ────────────────────────────────────

KALDIR = r"(?:kes|kesil|temizle|kaldır|çıkar|sil|\bal\b|alın|olmasın|yok\s*et|kazı|tıraş\s*et|traş\s*et|gitsin)"


def _listeden_cikar(deger, *anahtarlar):
    parcalar = [p.strip() for p in (deger or "").split(",") if p.strip()]
    return ", ".join(p for p in parcalar if not any(a in p for a in anahtarlar))


def _listeye_ekle(deger, yeni):
    parcalar = [p.strip() for p in (deger or "").split(",") if p.strip()]
    for y in [p.strip() for p in yeni.split(",") if p.strip()]:
        if y not in parcalar:
            parcalar.append(y)
    return ", ".join(parcalar)


def degisiklik_uygula(oz, metin):
    """Türkçe değişiklik komutlarını özelliklere uygular.

    Dönüş: (yeni özellikler, uygulanan değişikliklerin listesi).
    """
    yeni = dict(oz)
    uygulanan = []
    for satir in [s for s in re.split(r"[\n.;]+", tr_kucuk(metin)) if s.strip()]:
        islendi = False
        # Kaldırma komutları
        if re.search(r"sakal|bıyık", satir) and re.search(KALDIR, satir):
            if re.search(r"bıyık", satir) and not re.search(r"sakal", satir):
                s = yeni.get("sakal", "")
                s = re.sub(r"\s*(?:and|with)\s*(?:thick |thin )?mustache", "", s)
                yeni["sakal"] = "clean shaven" if "mustache" in s else s
            else:
                yeni["sakal"] = "clean shaven"
            uygulanan.append("sakal/bıyık kaldırıldı")
            islendi = True
        if re.search(r"gözlü[kğ]", satir) and re.search(KALDIR, satir):
            yeni["aksesuar"] = _listeden_cikar(yeni.get("aksesuar"), "glasses")
            uygulanan.append("gözlük kaldırıldı")
            islendi = True
        for kalip, anahtar, ad in [
            (r"yara", "scar", "yara izi"), (r"dövme", "tattoo", "dövme"),
            (r"\bben\b|\bbeni\b", "mole", "ben"), (r"küpe", "earring", "küpe"),
            (r"şapka|\bkep\b|bere", "cap", "şapka"), (r"kapüşon", "hood", "kapüşon"),
        ]:
            if re.search(kalip, satir) and re.search(KALDIR, satir):
                alan = "ozel" if anahtar in ("scar", "tattoo", "mole") else "aksesuar"
                yeni[alan] = _listeden_cikar(yeni.get(alan), anahtar, "hat", "beanie") \
                    if anahtar == "cap" else _listeden_cikar(yeni.get(alan), anahtar)
                uygulanan.append(f"{ad} kaldırıldı")
                islendi = True
        if islendi:
            continue

        # Yaş komutları
        if re.search(r"yaşlandır|daha\s*yaşlı|yaşlı\s*(?:göster|yap)|yaşını\s*artır", satir):
            fark = re.search(r"(\d{1,2})\s*yaş", satir)
            yeni["yas"] = min(YAS_UST, int(yeni["yas"]) + (int(fark.group(1)) if fark else 15))
            uygulanan.append(f"yaş → {yeni['yas']}")
            continue
        if re.search(r"gençleştir|daha\s*genç|genç\s*(?:göster|yap)|yaşını\s*azalt", satir):
            fark = re.search(r"(\d{1,2})\s*yaş", satir)
            yeni["yas"] = max(YAS_ALT, int(yeni["yas"]) - (int(fark.group(1)) if fark else 10))
            uygulanan.append(f"yaş → {yeni['yas']}")
            continue

        # Ekleme / değiştirme: satırdaki özellikleri çıkarıp üzerine yaz
        bulunan = ozellik_cikar(satir)
        if re.search(r"kel\s*(?:yap|olsun)|saç\w*\s*(?:kazı|sıfırla|tıraş)", satir):
            bulunan["sac"] = "bald shaved head"
        if re.search(r"sakal\w*\s*(?:ekle|bırak|uzat|olsun)", satir) and "sakal" not in bulunan:
            bulunan["sakal"] = "full beard"
        for alan, deger in bulunan.items():
            if alan == "cinsiyet" and not re.search(r"kadın|erkek", satir):
                continue
            if alan in ("aksesuar", "ozel"):
                deger = _listeye_ekle(yeni.get(alan), deger)
            elif alan == "sac":
                deger = _sac_birlestir(yeni.get("sac", ""), deger)
            elif alan == "goz":
                deger = _goz_birlestir(yeni.get("goz", ""), deger)
            yeni[alan] = deger
            uygulanan.append(f"{alan} → {deger}")
    return normalize(yeni, temel=oz), uygulanan


_EN_UZUNLUK = ("very short", "shoulder-length", "medium-length", "short", "long")
_EN_SAC_RENK = sorted({en for _, en in RENK_SAC}, key=len, reverse=True)
_EN_GOZ_RENK = sorted({en for _, en in RENK_GOZ}, key=len, reverse=True)


def _ilk_eslesen(metin, adaylar):
    for a in adaylar:
        if re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", metin):
            return a
    return None


def _sac_birlestir(eski, yeni):
    """'saçları sarı yap' gibi kısmi komutlarda eski uzunluk/renk bilgisini korur."""
    if not eski or yeni.startswith("bald") or eski.startswith("bald") or "buzz cut" in yeni:
        return yeni
    yeni_uzunluk = _ilk_eslesen(yeni, _EN_UZUNLUK)
    yeni_renk = _ilk_eslesen(yeni, _EN_SAC_RENK)
    eski_renk = _ilk_eslesen(eski, _EN_SAC_RENK)
    if yeni_uzunluk and yeni_renk:
        return yeni
    sonuc = eski
    if yeni_renk:
        sonuc = re.sub(re.escape(eski_renk), yeni_renk, sonuc, count=1) if eski_renk \
            else sonuc.replace("hair", f"{yeni_renk} hair", 1)
    if yeni_uzunluk:
        eski_uzunluk = _ilk_eslesen(sonuc, _EN_UZUNLUK)
        sonuc = re.sub(re.escape(eski_uzunluk), yeni_uzunluk, sonuc, count=1) if eski_uzunluk \
            else f"{yeni_uzunluk} {sonuc}"
    # Doku ve stil gibi ek bilgileri de taşı
    for ek in [e for _, e in SAC_DOKU + SAC_STIL] :
        if re.search(r"(?<!\w)" + re.escape(ek) + r"(?!\w)", yeni) and ek not in sonuc:
            sonuc = sonuc.replace("hair", f"{ek} hair", 1) if not ek.startswith(("in ", "with ")) \
                else f"{sonuc} {ek}"
    return sonuc


def _goz_birlestir(eski, yeni):
    yeni_renk = _ilk_eslesen(yeni, _EN_GOZ_RENK)
    if not eski or not yeni_renk or yeni.replace(yeni_renk, "").strip() != "eyes":
        return yeni
    eski_renk = _ilk_eslesen(eski, _EN_GOZ_RENK)
    if eski_renk:
        return re.sub(re.escape(eski_renk), yeni_renk, eski, count=1)
    return eski.replace("eyes", f"{yeni_renk} eyes")
