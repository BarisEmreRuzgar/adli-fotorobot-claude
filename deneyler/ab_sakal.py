"""Sakalsız yüze sakal ekleme: FaceID'siz bölgesel çizim (Aşama 3 yolu) + rötuşla renk.

Aşama 2'de FaceID kaynak yüzü (sakalsız) geri çektiği için sakal yerine kirli sakal çıkıyordu (0/3).
Varyantlar: maske (alt_yuz / genişletilmiş), güç, ağırlık. Kullanım: python deneyler/ab_sakal.py <kaynak> <ad>
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from vlm import denetle  # noqa: E402

from fotorobot import rotus  # noqa: E402
from fotorobot.motor import Motor, yuz_bolge_maskesi  # noqa: E402
from fotorobot.ozellikler import agirlikli  # noqa: E402

kaynak_yol, ad = sys.argv[1], sys.argv[2]
NEG = ("worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, "
       "clean shaven, smooth chin, woman, braid, braided beard, handlebar mustache, curled mustache, "
       "costume, fake beard, viking, wizard, brown beard, black beard, dark beard")
IDDIA = ["The man has a white or light grey full beard"]
VARYANTLAR = [  # (ad, tarif, ağırlık, güç, cfg)
    ("G beyaz 1.1 g0.95", "white full beard, grey facial hair", 1.1, 0.95, 6.5),
    ("H beyaz 1.2 g0.95", "snow white full beard, natural white facial hair", 1.2, 0.95, 7.0),
    ("I kir 1.1 g0.9", "short trimmed white and grey full beard, natural facial hair", 1.1, 0.9, 6.5),
]

mo = Motor()
mo.yukle()
kaynak = Image.open(kaynak_yol).convert("RGB")
yuz = mo.yuz_bul(kaynak)
maske = yuz_bolge_maskesi(yuz.kps, yuz.bbox, kaynak.size, ["alt_yuz"])
oz = {"sac": "", "sakal": "white full beard", "cinsiyet": "male"}
satirlar, etiketler = [], []
for vad, tarif, w, guc, cfg in VARYANTLAR:
    s = mo.bolge_duzenle(kaynak, maske, f"{agirlikli(tarif, w)}, photo, realistic", NEG, adet=3, tohum=2026,
                         adim=25, cfg=cfg, guc=guc, kenar=32)
    sat, et, ok_say = [kaynak], [vad], 0
    for im, t, _ in s:
        sat.append(im)
        son, yapilan = mo.rotus_uygula(im, oz, tohum=t, adim=25)
        kim = mo.kimlik_benzerligi(None, son, yuz_a=yuz) or 0
        v = denetle(son, IDDIA)
        ok = bool(v["kontroller"]) and all(v["kontroller"])
        ok_say += ok
        sat.append(son)
        et += ["ham", f"{'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}"]
        son.save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}_{vad[0]}_{t}.png"))
    satirlar.append(sat)
    etiketler.append(et)
    print("PUAN", vad, f"{ok_say}/3", et[1:], flush=True)
izgara(satirlar, etiketler, (200, 250)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
print("SONUC bitti", flush=True)
