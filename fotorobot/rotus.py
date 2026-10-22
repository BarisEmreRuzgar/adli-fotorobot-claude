"""Renk ve iz rötuşu: SD 1.5'in prompt'la tutturamadığı ayrıntıları piksel düzeyinde koyar.

SD 1.5'te renk çelişkileri ("siyah saç + beyaz sakal", "kısa sarı saç") ve yara izi
prompt'la güvenilir çıkmıyor (bkz. ILERLEME.md). Burada istenen renk/iz önce doğrudan
görüntüye işlenir, sonra düşük değişim gücüyle inpainting yapılarak doğallaştırılır
(SDEdit): düşük güçte model rengi ve kaba şekli korur, yalnızca dokuyu yeniden çizer.

Saç ve cilt bölgeleri yüz ayrıştırma modeliyle (SegFormer, CelebAMask-HQ sınıfları)
bulunur; sakal ve yara izi yeri InsightFace'in 5 yüz noktasından hesaplanır.
"""

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

AYRISTIRMA_MODELI = "jonathandinu/face-parsing"
SINIF = {"background": 0, "skin": 1, "nose": 2, "eye_g": 3, "l_eye": 4, "r_eye": 5, "l_brow": 6,
         "r_brow": 7, "l_ear": 8, "r_ear": 9, "mouth": 10, "u_lip": 11, "l_lip": 12, "hair": 13,
         "hat": 14, "ear_r": 15, "neck_l": 16, "neck": 17, "cloth": 18}

# Saç/sakal renk sözcükleri → hedef CIE Lab. Uzun ifadeler önce aranır.
RENKLER = {
    "platinum blonde": (80, 1, 10), "salt and pepper": (55, 0, 3), "jet black": (10, 0, 0),
    "dark brown": (22, 5, 9), "light brown": (46, 8, 19), "chestnut brown": (32, 13, 18),
    "red ginger": (45, 28, 38), "white": (79, 0, 4), "silver": (72, 0, 2), "grey": (62, 0, 3),
    "gray": (62, 0, 3), "greying": (55, 1, 6), "blonde": (64, 5, 24), "golden": (62, 9, 32),
    "auburn": (34, 22, 24), "ginger": (48, 26, 38), "red": (40, 30, 30), "black": (13, 1, 1),
    "brown": (30, 8, 14),
}
# Ölçülen renk hedeften bu kadar (ΔE) uzaksa boyanır; yakınsa model zaten tutturmuştur
RENK_ESIGI = 15.0
# Boyama sonrası doğallaştırma gücü. Düşük: renk kalır ama boyalı görünür; yüksek: renk kaybolur.
SAC_INCELTME = 0.45
SAKAL_INCELTME = 0.6
YARA_INCELTME = 0.3
# Beyaz/gri kıl inceltmede sarı-kızıla kayıyordu (Aşama 1 testi, bıyık); bu renklerde negatife eklenir
ACIK_RENK_NEGATIFI = "blonde, yellow, golden, ginger, orange, brown"
INCELTME_NEGATIFI = ("worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, "
                     "makeup, tattoo, ink, stripes, tape, bandage")

_ayristirici = None


def _ayristirici_yukle():
    global _ayristirici
    if _ayristirici is None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        model = SegformerForSemanticSegmentation.from_pretrained(AYRISTIRMA_MODELI).to(cihaz).eval()
        _ayristirici = (SegformerImageProcessor.from_pretrained(AYRISTIRMA_MODELI), model, cihaz)
    return _ayristirici


def ayristir(gorsel):
    """Her piksel için sınıf numarası (H, W)."""
    import torch

    isl, model, cihaz = _ayristirici_yukle()
    gorsel = gorsel.convert("RGB")
    with torch.no_grad():
        girdi = isl(images=gorsel, return_tensors="pt")["pixel_values"].to(cihaz)
        logit = model(pixel_values=girdi).logits
        logit = torch.nn.functional.interpolate(logit, size=gorsel.size[::-1], mode="bilinear")
    return logit.argmax(1)[0].cpu().numpy()


# ── Renk uzayı ────────────────────────────────────────────────────────

_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_BEYAZ = np.array([0.95047, 1.0, 1.08883])


def rgb_lab(rgb):
    """(..., 3) uint8 sRGB → CIE Lab (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _BEYAZ
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_rgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _BEYAZ
    c = xyz @ np.linalg.inv(_M).T
    c = np.where(c > 0.0031308, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055, 12.92 * c)
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)


def renk_bul(metin):
    """Metindeki ilk saç/sakal rengi (anahtar) veya None. 'dyed' gibi sözcükler yok sayılır."""
    m = (metin or "").lower()
    for ad in sorted(RENKLER, key=len, reverse=True):
        if re.search(r"(?<![\w-])" + re.escape(ad) + r"(?![\w-])", m):
            return ad
    return None


def _ortanca(lab, maske):
    return np.median(lab[maske], axis=0) if maske.sum() >= 30 else None


def _boya(lab, agirlik, kaynak_ort, hedef, doku=0.7):
    """Ağırlıklı pikselleri hedef renge taşır; parlaklık farkları (doku) `doku` oranında korunur."""
    yeni = lab.copy()
    yeni[..., 0] = hedef[0] + (lab[..., 0] - kaynak_ort[0]) * doku
    yeni[..., 1] = hedef[1] + (lab[..., 1] - kaynak_ort[1]) * 0.3
    yeni[..., 2] = hedef[2] + (lab[..., 2] - kaynak_ort[2]) * 0.3
    yeni[..., 0] = np.clip(yeni[..., 0], 2, 98)
    w = agirlik[..., None]
    return lab * (1 - w) + yeni * w


def _yumusat(maske, yaricap):
    im = Image.fromarray((np.clip(maske, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(yaricap)), dtype=np.float64) / 255


# ── Bölgeler ─────────────────────────────────────────────────────────

def _noktalar(kps):
    (lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry) = [tuple(map(float, p)) for p in kps]
    return lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry, max(abs(rx - lx), 1.0)


def sakal_bolgesi(kps, boyut, harita, yalniz_biyik=False, siniflar=("skin", "hair")):
    """Bıyık + sakal + çene + favoriler (dudak, burun, göz hariç), yumuşak (H, W) ağırlık.
    `yalniz_biyik`: yalnızca üst dudak ve ağız kenarları (pala bıyığın uçları dahil)."""
    lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry, d = _noktalar(kps)
    my = (mly + mry) / 2
    m = Image.new("L", boyut, 0)
    if yalniz_biyik:
        ImageDraw.Draw(m).polygon([
            (mlx - 0.3 * d, ny + 0.02 * d), (mrx + 0.3 * d, ny + 0.02 * d), (mrx + 0.45 * d, my + 0.15 * d),
            (mrx + 0.35 * d, my + 0.55 * d), (mrx + 0.1 * d, my + 0.1 * d), (mlx - 0.1 * d, my + 0.1 * d),
            (mlx - 0.35 * d, my + 0.55 * d), (mlx - 0.45 * d, my + 0.15 * d)], fill=255)
    else:
        ImageDraw.Draw(m).polygon([
            (lx - 0.75 * d, ly + 0.25 * d), (lx - 0.35 * d, ny + 0.05 * d), (nx, ny + 0.02 * d),
            (rx + 0.35 * d, ny + 0.05 * d), (rx + 0.75 * d, ry + 0.25 * d), (rx + 0.6 * d, my + 0.6 * d),
            (mrx + 0.2 * d, my + 1.05 * d), ((mlx + mrx) / 2, my + 1.2 * d), (mlx - 0.2 * d, my + 1.05 * d),
            (lx - 0.6 * d, my + 0.6 * d)], fill=255)
    # Sınır sert olursa boyanan sakal yanakta yatay bir çizgiyle kesiliyordu: yumuşak geçiş
    bolge = np.asarray(m.filter(ImageFilter.GaussianBlur(0.12 * d)), dtype=np.float64) / 255
    # Boyun (çene altı gölgesi koyu olduğu için kıl sanılıyordu) ve saç dışındaki her şey hariç
    izinli = np.isin(harita, [SINIF[k] for k in siniflar])
    return bolge * izinli


def yanak_cildi(lab, kps, harita):
    lx, ly, rx, ry, *_, d = _noktalar(kps)
    h, w = harita.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.zeros((h, w), bool)
    for gx, gy in ((lx, ly), (rx, ry)):
        m |= ((xx - gx) ** 2 + (yy - gy - 0.5 * d) ** 2) < (0.16 * d) ** 2
    return _ortanca(lab, m & (harita == SINIF["skin"]))


def sac_rengi_olc(gorsel, harita):
    lab = rgb_lab(np.asarray(gorsel.convert("RGB")))
    return _ortanca(lab, harita == SINIF["hair"])


def sakal_kilini_bul(lab, kps, harita, yalniz_biyik=False):
    """Sakal bölgesinde cilde göre koyu (kıl) piksellerin yumuşak ağırlığı ve cilt rengi."""
    cilt = yanak_cildi(lab, kps, harita)
    bolge = sakal_bolgesi(kps, harita.shape[::-1], harita, yalniz_biyik)
    if cilt is None or (bolge > 0.5).sum() < 50:
        return None, None
    # Kıl, aynı ışıktaki cilde göre koyudur; 6 birimden koyular kıl sayılmaya başlar
    koyuluk = np.clip((cilt[0] - lab[..., 0] - 6) / 16, 0, 1)
    agirlik = koyuluk * bolge
    # Ayrıştırıcı gür bıyığın bir kısmını "üst dudak" etiketliyor; dudaktan çok koyu olanlar kıldır
    dudak = sakal_bolgesi(kps, harita.shape[::-1], harita, yalniz_biyik, siniflar=("u_lip",))
    agirlik = np.maximum(agirlik, np.clip((cilt[0] - lab[..., 0] - 25) / 10, 0, 1) * dudak)
    if (agirlik > 0.5).sum() < 0.08 * (bolge > 0.5).sum():
        return None, cilt
    return _yumusat(agirlik, 1.2), cilt


# ── Boyama ───────────────────────────────────────────────────────────

def sac_boya(gorsel, harita, renk):
    """Saç piksellerini `renk`e boyar. Dönüş: (boyalı görsel, değişen bölge maskesi) ya da (None, None)."""
    lab = rgb_lab(np.asarray(gorsel.convert("RGB")))
    maske = harita == SINIF["hair"]
    ort = _ortanca(lab, maske)
    if ort is None:
        return None, None
    hedef = np.array(RENKLER[renk], dtype=np.float64)
    agirlik = _yumusat(maske.astype(np.float64), 1.5) * maske.max()
    # Koyu saçı açarken dokunun kontrastı da artmasın diye korunan oran azaltılır
    # Koyu saç açılırken doku çok azaltılınca düz "boya" gibi görünüyordu (Gemini gerçekçilik 2-3)
    yeni = _boya(lab, agirlik, ort, hedef, doku=0.7 if hedef[0] > ort[0] + 20 else 0.85)
    return Image.fromarray(lab_rgb(yeni)), maske


def renk_kilitle(gorsel, maske, renk, oran=0.8):
    """İnceltme sonrası maskedeki rengi (a*, b*) hedefe geri çeker; parlaklık ve doku olduğu gibi kalır.
    Beyaz bıyık inceltmede altın sarısına kayıyordu."""
    lab = rgb_lab(np.asarray(gorsel.convert("RGB")))
    m = np.asarray(maske, dtype=np.float64)
    m = m / 255 if m.max() > 1 else m
    m = _yumusat(m, 1.0) * oran
    hedef = np.array(RENKLER[renk], dtype=np.float64)
    ort = _ortanca(lab, m > 0.3)
    if ort is None:
        return gorsel
    yeni = lab.copy()
    for k in (1, 2):
        yeni[..., k] = hedef[k] + (lab[..., k] - ort[k]) * 0.4
    sonuc = lab * (1 - m[..., None]) + yeni * m[..., None]
    sonuc[..., 0] = lab[..., 0]
    return Image.fromarray(lab_rgb(sonuc))


def yalniz_biyik_mi(sakal):
    s = (sakal or "").lower()
    return ("mustache" in s or "moustache" in s) and not any(k in s for k in ("beard", "goatee", "stubble"))


def sakal_boya(gorsel, kps, harita, renk, yalniz_biyik=False):
    """Sakal/bıyık kıllarını `renk`e boyar; `yalniz_biyik` ise çene ve yanaklara dokunulmaz."""
    lab = rgb_lab(np.asarray(gorsel.convert("RGB")))
    agirlik, _ = sakal_kilini_bul(lab, kps, harita, yalniz_biyik)
    if agirlik is None:
        return None, None
    kil = agirlik > 0.3
    ort = _ortanca(lab, kil)
    hedef = np.array(RENKLER[renk], dtype=np.float64)
    if renk == "salt and pepper":
        # Kırçıl: kılların yarısı (açık olanlar) griye, koyular olduğu gibi
        agirlik = agirlik * (lab[..., 0] > ort[0])
    # Beyaz/gri kıllar arasındaki gölgeler kalmalı; düz boya "pamuk" gibi görünüyordu
    yeni = _boya(lab, agirlik, ort, hedef, doku=0.85)
    return Image.fromarray(lab_rgb(yeni)), kil


def sakal_rengi_olc(gorsel, kps, harita, yalniz_biyik=False):
    """Sakal/bıyık kıllarının ortanca Lab rengi; kıl bulunamazsa None."""
    lab = rgb_lab(np.asarray(gorsel.convert("RGB")))
    agirlik, _ = sakal_kilini_bul(lab, kps, harita, yalniz_biyik)
    return None if agirlik is None else _ortanca(lab, agirlik > 0.3)


def renk_farki(olculen, renk):
    return float(np.linalg.norm(np.asarray(olculen) - np.asarray(RENKLER[renk])))


def renk_uyumlu(olculen, renk):
    """Ölçülen renk istenen renk ailesinde mi? Tam Lab uzaklığı fazla katıydı: model beyaz sakalı
    doğal kır (L*~60) çizdiğinde yeniden boyanıyor ve gerçekçilik bozuluyordu (Aşama 2 sakal testi)."""
    L, a, b = (float(x) for x in olculen)
    kroma = float(np.hypot(a, b))
    if renk in ("white", "silver", "grey", "gray"):
        return kroma < 14 and L >= RENKLER[renk][0] - 22
    if renk in ("blonde", "golden", "platinum blonde"):
        return L >= 50 and b >= 12
    if renk in ("black", "jet black"):
        return L <= 25
    if renk == "salt and pepper":
        return kroma < 14 and 35 <= L <= 70
    return renk_farki(olculen, renk) <= RENK_ESIGI


# ── Yara izi ─────────────────────────────────────────────────────────

YARA_YERLERI = (  # (anahtar sözcükler, yer); özelden genele
    (("eyebrow", "brow"), "kas"), (("upper lip", "above the lip", "lip"), "dudak"),
    (("cheek",), "yanak"), (("chin",), "cene"), (("forehead",), "alin"), (("nose",), "burun"),
    (("jaw",), "cene_hatti"), (("eye",), "goz_alti"),
)
YARA_SOZCUKLERI = ("scar", "cut mark", "stitches", "burn mark")


def yara_konumlari(ozel):
    """`ozel` alanındaki yeri belli yara izlerini ayıklar. Dönüş: (kalan, [(yer, taraf), ...])."""
    kalan, yaralar = [], []
    for parca in re.split(r",|;|\band\b", ozel or ""):
        p = parca.strip()
        k = p.lower()
        if not p:
            continue
        if not any(s in k for s in YARA_SOZCUKLERI) or "burn" in k:
            kalan.append(p)
            continue
        yer = next((y for sozcukler, y in YARA_YERLERI if any(x in k for x in sozcukler)), None)
        taraf = "sag" if re.search(r"\bright\b", k) else "sol" if re.search(r"\bleft\b", k) else None
        if yer is None:
            kalan.append(p)
        else:
            yaralar.append((yer, taraf))
    return ", ".join(kalan), yaralar


def yara_yolu(kps, yer, taraf=None):
    """Yara izinin görüntüdeki çizgisi [(x, y), ...]. taraf kişinin kendi sağı/solu
    (kişinin solu görüntünün sağında); taraf yoksa kişinin solu."""
    lx, ly, rx, ry, nx, ny, mlx, mly, mrx, mry, d = _noktalar(kps)
    sag = taraf == "sag"
    gx, gy = (lx, ly) if sag else (rx, ry)
    ax, ay = (mlx, mly) if sag else (mrx, mry)
    yon = -1.0 if sag else 1.0  # yüzün dışına doğru
    if yer == "yanak":
        bas = (gx + yon * 0.05 * d, gy + 0.42 * d)
        son = (gx + yon * 0.38 * d, (gy + ay) / 2 + 0.38 * d)
    elif yer == "kas":
        bas = (gx + yon * 0.12 * d, gy - 0.62 * d)
        son = (gx + yon * 0.2 * d, gy - 0.12 * d)
    elif yer == "alin":
        cx = (lx + rx) / 2 + yon * 0.25 * d
        bas, son = (cx - 0.22 * d, (ly + ry) / 2 - 1.05 * d), (cx + 0.18 * d, (ly + ry) / 2 - 0.75 * d)
    elif yer == "cene":
        cx, cy = (mlx + mrx) / 2 + yon * 0.12 * d, (mly + mry) / 2 + 0.55 * d
        bas, son = (cx - 0.18 * d, cy - 0.05 * d), (cx + 0.18 * d, cy + 0.08 * d)
    elif yer == "cene_hatti":
        bas, son = (ax + yon * 0.35 * d, ay + 0.1 * d), (ax + yon * 0.6 * d, ay - 0.3 * d)
    elif yer == "dudak":
        bas, son = (ax - yon * 0.15 * d, ay - 0.45 * d), (ax - yon * 0.05 * d, ay - 0.05 * d)
    elif yer == "burun":
        bas, son = (nx - 0.12 * d, ny - 0.55 * d), (nx + 0.08 * d, ny - 0.3 * d)
    elif yer == "goz_alti":
        bas, son = (gx - 0.1 * d, gy + 0.25 * d), (gx + yon * 0.3 * d, gy + 0.4 * d)
    else:
        raise ValueError(f"bilinmeyen yara yeri: {yer}")
    # Hafif kavisli yol (iyileşmiş kesik izi düz cetvel çizgisi değildir)
    (x0, y0), (x1, y1) = bas, son
    nxv, nyv = -(y1 - y0), x1 - x0
    uz = max(np.hypot(nxv, nyv), 1e-6)
    egri = 0.06 * d
    return [(x0 + (x1 - x0) * t + nxv / uz * egri * np.sin(np.pi * t),
             y0 + (y1 - y0) * t + nyv / uz * egri * np.sin(np.pi * t)) for t in np.linspace(0, 1, 24)]


def yara_ciz(gorsel, kps, yol, harita=None, tohum=0):
    """Yola kabarık, soluk, hafif pembe bir iyileşmiş yara izi çizer.
    Dönüş: (görsel, yara maskesi (L, inceltme için genişletilmiş))."""
    rng = np.random.default_rng(tohum)
    *_, d = _noktalar(kps)
    g = gorsel.convert("RGB")
    lab = rgb_lab(np.asarray(g))
    w, h = g.size
    kalinlik = max(2.0, 0.045 * d)
    # Kalınlık boyunca düzensiz: uçlarda incelir, ortada kalınlaşır
    cekirdek = Image.new("L", (w * 2, h * 2), 0)
    ciz = ImageDraw.Draw(cekirdek)
    n = len(yol)
    for i in range(n - 1):
        t = i / (n - 1)
        k = kalinlik * (0.45 + 0.75 * np.sin(np.pi * t)) * rng.uniform(0.85, 1.15)
        (x0, y0), (x1, y1) = yol[i], yol[i + 1]
        ciz.line((x0 * 2, y0 * 2, x1 * 2, y1 * 2), fill=255, width=max(2, int(round(k * 2))))
        ciz.ellipse((x1 * 2 - k, y1 * 2 - k, x1 * 2 + k, y1 * 2 + k), fill=255)
    cekirdek = cekirdek.resize((w, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6))
    c = np.asarray(cekirdek, dtype=np.float64) / 255
    # Kabarıklık: üst kenar ışık alır, alt kenar gölgede kalır (ışık yukarıdan)
    golge = np.roll(c, 2, axis=0) * (1 - c)
    isik = np.roll(c, -1, axis=0) * (1 - c)
    yerel = np.median(lab[np.asarray(cekirdek.filter(ImageFilter.MaxFilter(15))) > 0], axis=0)
    yeni = lab.copy()
    doku = (lab[..., 0] - yerel[0]) * 0.35  # cilt gözenekleri hafifçe kalsın
    yeni[..., 0] = np.where(c > 0, yerel[0] + 9 + doku, lab[..., 0])
    yeni[..., 1] = np.where(c > 0, yerel[1] * 0.75 + 5, lab[..., 1])
    yeni[..., 2] = np.where(c > 0, yerel[2] * 0.65, lab[..., 2])
    sonuc = lab * (1 - c[..., None]) + yeni * c[..., None]
    sonuc[..., 0] -= golge * 9
    sonuc[..., 0] += isik * 4
    if harita is not None:
        # Kaş, göz, dudak üstüne taşan kısım cilt değildir; kaşta kıllar seyrelir (iz kaşı böler)
        kas = np.isin(harita, [SINIF["l_brow"], SINIF["r_brow"]])
        disari = np.isin(harita, [SINIF["l_eye"], SINIF["r_eye"], SINIF["u_lip"], SINIF["l_lip"],
                                  SINIF["mouth"], SINIF["background"], SINIF["hair"]]) & ~kas
        sonuc = np.where(disari[..., None], lab, sonuc)
    maske = cekirdek.filter(ImageFilter.MaxFilter(int(2 * round(0.06 * d) + 1)))
    return Image.fromarray(lab_rgb(sonuc)), maske.filter(ImageFilter.GaussianBlur(2))
