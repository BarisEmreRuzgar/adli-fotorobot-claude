"""Beyaz sakal adaylarında bıyık bölgesinin sarılığı: bölge ortanca Lab'ı ile görsel karşılaştırma için
aday ızgarası. Kullanım: python deneyler/biyik_renk.py [aday_sayisi]
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

import numpy as np  # noqa: E402
from benchmark_and_test import TEMEL, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from uctan_uca import bos_ilerleme  # noqa: E402

from fotorobot import rotus  # noqa: E402
from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import yerel_cizimler  # noqa: E402

if __name__ == "__main__":
    from fotorobot.motor import Motor

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    mo = Motor()
    mo.yukle()
    gem = Gemini("", mo.ayarlar.gemini_modelleri)
    app = Uygulama(mo, gemini=gem)
    oz = oz_cikar(TEMEL)
    _, liste, _, _ = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, 4242, progress=bos_ilerleme)
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    yeni_oz, _ = gem.degisiklik(liste[0]["oz"], "beyaz sakal ekle")
    y = dict(yerel_cizimler(liste[0]["oz"], yeni_oz)[0][0], aday=1)
    sat, et = [], []
    for k in range(n):
        im, kim = mo.yerel_ciz_sec(kaynak, y, 2026 + k * 7919, 25, yuz)
        yz = mo.yuz_bul(im)
        harita = rotus.ayristir(im)
        lab = rotus.rgb_lab(np.asarray(im))
        biyik = rotus.sakal_bolgesi(yz.kps, im.size, harita, yalniz_biyik=True) > 0.5
        cene = (rotus.sakal_bolgesi(yz.kps, im.size, harita) > 0.5) & ~biyik
        b, c = rotus._ortanca(lab, biyik), rotus._ortanca(lab, cene)
        cilt = rotus.yanak_cildi(lab, yz.kps, harita)
        kil = rotus._ortanca(lab, biyik & (lab[..., 0] > cilt[0] + 4)) if cilt is not None else None
        kil = kil if kil is not None else [0, 0, 0]
        olc = rotus.sakal_rengi_olc(im, yz.kps, harita)
        sat.append(im)
        et.append(f"{k} id{kim:.2f} kil L{kil[0]:.0f} b{kil[2]:.0f} cene L{c[0]:.0f} b{c[2]:.0f}")
        print("ADAY", et[-1], "olc", None if olc is None else np.round(olc).tolist(), flush=True)
    izgara([sat[:3], sat[3:]], [et[:3], et[3:]], (260, 260)).save(os.path.join(KOK, "deneyler", "sonuc", "biyik_renk.png"))
    print("SONUC bitti", flush=True)
