"""Deneyler için otomatik ölçümler: yüz ayrıştırma (SegFormer, CelebAMask-HQ sınıfları),
saç/sakal rengi (CIE Lab), ArcFace kimlik benzerliği ve sonuç ızgarası."""
import numpy as np
from PIL import Image, ImageDraw

AYRISTIRMA_MODELI = "jonathandinu/face-parsing"
SINIF = {"background": 0, "skin": 1, "nose": 2, "eye_g": 3, "l_eye": 4, "r_eye": 5, "l_brow": 6,
         "r_brow": 7, "l_ear": 8, "r_ear": 9, "mouth": 10, "u_lip": 11, "l_lip": 12, "hair": 13,
         "hat": 14, "ear_r": 15, "neck_l": 16, "neck": 17, "cloth": 18}

_model = None


def ayristir(gorsel):
    """Her piksel için sınıf numarası (H, W) döndürür."""
    global _model
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    if _model is None:
        _model = (SegformerImageProcessor.from_pretrained(AYRISTIRMA_MODELI),
                  SegformerForSemanticSegmentation.from_pretrained(AYRISTIRMA_MODELI).eval())
    isl, model = _model
    gorsel = gorsel.convert("RGB")
    with torch.no_grad():
        logit = model(**isl(images=gorsel, return_tensors="pt")).logits
        logit = torch.nn.functional.interpolate(logit, size=gorsel.size[::-1], mode="bilinear")
    return logit.argmax(1)[0].numpy()


def lab(gorsel):
    """sRGB → CIE Lab (D65), (H, W, 3) float."""
    rgb = np.asarray(gorsel.convert("RGB"), dtype=np.float64) / 255
    rgb = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = rgb @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def bolge_rengi(gorsel, maske):
    """Maskedeki piksellerin ortanca L*, a*, b* değerleri ve piksel sayısı."""
    if maske.sum() < 30:
        return None
    L = lab(gorsel)[maske]
    return {"L": float(np.median(L[:, 0])), "a": float(np.median(L[:, 1])), "b": float(np.median(L[:, 2])),
            "n": int(maske.sum())}


def sac_rengi(gorsel, harita=None):
    harita = ayristir(gorsel) if harita is None else harita
    return bolge_rengi(gorsel, harita == SINIF["hair"])


def sakal_maskesi(kps, boyut, harita):
    """Ağız çevresi + çene (dudaklar hariç): bıyık ve sakalın olduğu yer."""
    (lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry) = [tuple(map(float, p)) for p in kps]
    d = max(abs(rx - lx), 1.0)
    alt = (mly + mry) / 2 + 0.9 * d
    m = Image.new("L", boyut, 0)
    ImageDraw.Draw(m).polygon([(mlx - 0.45 * d, ny + 0.1 * d), (mrx + 0.45 * d, ny + 0.1 * d),
                               (mrx + 0.4 * d, alt), ((mlx + mrx) / 2, alt + 0.25 * d),
                               (mlx - 0.4 * d, alt)], fill=255)
    m = np.asarray(m) > 0
    dudak = np.isin(harita, [SINIF["u_lip"], SINIF["l_lip"], SINIF["mouth"], SINIF["nose"]])
    return m & ~dudak


def yanak_maskesi(kps, boyut):
    (lx, ly), (rx, ry), (nx, ny), _, _ = [tuple(map(float, p)) for p in kps]
    d = max(abs(rx - lx), 1.0)
    m = Image.new("L", boyut, 0)
    c = ImageDraw.Draw(m)
    for gx, gy in ((lx, ly), (rx, ry)):
        c.ellipse((gx - 0.18 * d, gy + 0.35 * d, gx + 0.18 * d, gy + 0.65 * d), fill=255)
    return np.asarray(m) > 0


def sakal_olcumu(gorsel, kps, harita=None):
    """Sakal bölgesi ile yanak cildinin renk farkı. Beyaz sakalda dL ≥ 0, koyu sakalda çok negatif."""
    harita = ayristir(gorsel) if harita is None else harita
    s = bolge_rengi(gorsel, sakal_maskesi(kps, gorsel.size, harita))
    y = bolge_rengi(gorsel, yanak_maskesi(kps, gorsel.size))
    if not s or not y:
        return None
    return {"L": s["L"], "dL": s["L"] - y["L"], "kroma": float(np.hypot(s["a"], s["b"]))}


def izgara(satirlar, etiketler=None, boyut=(256, 320)):
    """satirlar: [[PIL, ...], ...]; etiketler aynı biçimde metin (görselin altına yazılır)."""
    g, y = boyut
    sutun = max(len(s) for s in satirlar)
    tuval = Image.new("RGB", (sutun * g, len(satirlar) * (y + 16)), "white")
    ciz = ImageDraw.Draw(tuval)
    for r, sat in enumerate(satirlar):
        for c, im in enumerate(sat):
            tuval.paste(im.convert("RGB").resize((g, y)), (c * g, r * (y + 16)))
            if etiketler and etiketler[r] and c < len(etiketler[r]):
                ciz.text((c * g + 3, r * (y + 16) + y + 2), str(etiketler[r][c]), fill="black")
    return tuval
