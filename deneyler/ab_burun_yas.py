"""Burun ve yaşlandırma: FaceID'siz bölgesel çizim (FaceID kaynağın burnunu/yaşını geri çekiyordu, 0/3).

Kullanım: python deneyler/ab_burun_yas.py <kaynak> <ad>
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

import numpy as np  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image, ImageChops, ImageDraw  # noqa: E402
from vlm import denetle  # noqa: E402

from fotorobot.motor import Motor, yuz_bolge_maskesi  # noqa: E402
from fotorobot.ozellikler import agirlikli  # noqa: E402

kaynak_yol, ad = sys.argv[1], sys.argv[2]
NEG = "worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, makeup"


def yas_maskeleri(kps, bbox, boyut):
    """Yaşlanmanın görüldüğü bölgeler: alın, göz çevresi (kaz ayağı + göz altı), ağız kenarı çizgileri."""
    (lx, ly), (rx, ry), (nx, ny), (mlx, mly), (mrx, mry) = [tuple(map(float, p)) for p in kps]
    d = rx - lx
    m = Image.new("L", boyut, 0)
    c = ImageDraw.Draw(m)
    c.rectangle((lx - 0.45 * d, ly - 1.25 * d, rx + 0.45 * d, ly - 0.55 * d), fill=255)  # alın
    for gx, gy, yon in ((lx, ly, -1), (rx, ry, 1)):
        c.ellipse((gx - 0.45 * d + yon * 0.15 * d, gy - 0.2 * d, gx + 0.45 * d + yon * 0.15 * d, gy + 0.45 * d),
                  fill=255)  # göz çevresi
    for ax, ay, yon in ((mlx, mly, -1), (mrx, mry, 1)):
        c.ellipse((ax + yon * 0.25 * d - 0.2 * d, ny - 0.1 * d, ax + yon * 0.25 * d + 0.2 * d, ay + 0.3 * d),
                  fill=255)  # nazolabial
    return m


mo = Motor()
mo.yukle()
kaynak = Image.open(kaynak_yol).convert("RGB")
yuz = mo.yuz_bul(kaynak)
burun_m = yuz_bolge_maskesi(yuz.kps, yuz.bbox, kaynak.size, ["burun"])
yas_m = yas_maskeleri(yuz.kps, yuz.bbox, kaynak.size)
VARYANTLAR = [  # (ad, iddia, maske, tarif, ağırlık, güç, cfg) — maske None: tam görüntü img2img
    ("burun g0.75", "nose", burun_m, "aquiline hooked nose with a prominent bump on the nose bridge", 1.25, 0.75, 7.0),
    ("burun g0.9", "nose", burun_m, "aquiline hooked nose with a prominent bump on the nose bridge", 1.25, 0.9, 7.0),
    ("yas bolge g0.55", "yas", yas_m, "50 year old man, deep forehead wrinkles, crow's feet, under-eye bags, "
     "nasolabial folds, aged skin", 1.2, 0.55, 7.0),
    ("yas bolge g0.7", "yas", yas_m, "50 year old man, deep forehead wrinkles, crow's feet, under-eye bags, "
     "nasolabial folds, aged skin", 1.2, 0.7, 7.0),
    ("yas tam i2i g0.45", "yas", None, "50 year old middle-aged man, wrinkles, crow's feet, aged skin, "
     "greying temples", 1.2, 0.45, 6.5),
]
IDDIA = {"nose": ["The man has a clearly hooked, aquiline nose with a visible bump on the bridge"],
         "yas": ["The man clearly looks about 50 years old (middle-aged, visible wrinkles)"]}
satirlar, etiketler = [], []
for vad, tur, maske, tarif, w, guc, cfg in VARYANTLAR:
    poz = f"{agirlikli(tarif, w)}, photo, realistic, natural skin texture"
    if maske is not None:
        s = mo.bolge_duzenle(kaynak, maske, poz, NEG, adet=3, tohum=2026, adim=25, cfg=cfg, guc=guc, kenar=48)
    else:
        import torch

        poz_e, neg_e = mo._prompt_kodla(poz, NEG)
        s = []
        with mo._ip_adapter_kapali():
            for i, t in enumerate(mo._tohumlar(2026, 3)):
                with torch.inference_mode():
                    im = mo.pipe_i2i(image=kaynak, strength=guc, prompt_embeds=poz_e, negative_prompt_embeds=neg_e,
                                     num_inference_steps=int(25 / guc), guidance_scale=cfg,
                                     generator=torch.Generator("cpu").manual_seed(t)).images[0]
                s.append((im, t, None))
    sat, et, ok_say = [kaynak], [vad], 0
    for im, t, _ in s:
        kim = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
        v = denetle(im, IDDIA[tur])
        ok = bool(v["kontroller"]) and all(v["kontroller"])
        ok_say += ok
        sat.append(im)
        et.append(f"{'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}")
        fark = np.asarray(ImageChops.difference(kaynak, im).convert("L")).mean()
        et[-1] += f" d{fark:.0f}"
    satirlar.append(sat)
    etiketler.append(et)
    print("PUAN", vad, f"{ok_say}/3", et[1:], flush=True)
izgara(satirlar, etiketler).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
print("SONUC bitti", flush=True)
