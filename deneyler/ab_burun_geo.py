"""Kemerli burun, geometrik yol: rotus.burun_kemerli (bükme + kemer gölgesi) ± düşük güçlü inceltme.
Kullanım: python deneyler/ab_burun_geo.py <ad>
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import TEMEL, TOHUM1, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from uctan_uca import IDDIA2, bos_ilerleme, denetim  # noqa: E402

from fotorobot import rotus  # noqa: E402
from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import agirlikli  # noqa: E402

VARYANTLAR = [(1.0, 0.0), (1.6, 0.0), (1.6, 0.3), (2.2, 0.35)]  # (bükme gücü, inceltme gücü)
IDDIA = IDDIA2["burnunu kemerli yap"]

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad = sys.argv[1]
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    oz = oz_cikar(TEMEL)
    _, liste, _, _ = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, TOHUM1, progress=bos_ilerleme)
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    sat, et = [kaynak], ["kaynak"]
    for bg, ig in VARYANTLAR:
        im, maske = rotus.burun_kemerli(kaynak, yuz.kps, bg)
        if ig:
            im = mo._incelt(im, maske, f"{agirlikli('aquiline hooked nose, bump on the nose bridge', 1.2)}, "
                            "photo, natural skin", ig, TOHUM1, 25)
        kim = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
        ok, v = denetim(im, IDDIA)
        sat.append(im)
        et.append(f"b{bg} i{ig} {'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}")
        print("PUAN", et[-1], v.get("not", "")[:100], flush=True)
    izgara([sat], [et], (256, 256)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
    print("SONUC bitti", flush=True)
