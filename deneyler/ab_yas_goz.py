"""Yaşlandırmada kimlik: yerel çizimden sonra gözlerin (ve kaşların) kaynaktan geri konması.
Kullanım: python deneyler/ab_yas_goz.py <ad> [tohum_sayisi]
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

import numpy as np  # noqa: E402
from benchmark_and_test import TEMEL, TOHUM1, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from uctan_uca import IDDIA2, bos_ilerleme, denetim  # noqa: E402

from fotorobot import rotus  # noqa: E402
from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import yerel_cizimler  # noqa: E402


def geri_koy(kaynak, yeni, harita, siniflar, genislet):
    m = np.isin(harita, [rotus.SINIF[s] for s in siniflar]).astype(np.uint8) * 255
    m = Image.fromarray(m).filter(ImageFilter.MaxFilter(genislet)).filter(ImageFilter.GaussianBlur(2))
    a = (np.asarray(m, float) / 255)[..., None]
    return Image.fromarray((np.asarray(kaynak, float) * a + np.asarray(yeni, float) * (1 - a)).astype(np.uint8))


if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mo = Motor()
    mo.yukle()
    gem = Gemini("", mo.ayarlar.gemini_modelleri)
    app = Uygulama(mo, gemini=gem)
    oz = oz_cikar(TEMEL)
    _, liste, _, _ = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, TOHUM1, progress=bos_ilerleme)
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    harita = rotus.ayristir(kaynak)
    yeni_oz, _ = gem.degisiklik(liste[0]["oz"], "15 yaş yaşlandır")
    y = yerel_cizimler(liste[0]["oz"], yeni_oz)[0][0]
    VARYANT = {"A": None, "B göz": (("l_eye", "r_eye"), 5), "C göz+kaş": (("l_eye", "r_eye", "l_brow", "r_brow"), 5)}
    satirlar, etiketler = [], []
    for k in range(n):
        im = mo.yerel_ciz(kaynak, y["bolgeler"], y["pozitif"], y["negatif"], adet=1, tohum=2026 + k,
                          adim=25, cfg=y["cfg"], guc=y["guc"])[0][0]
        sat, et = [kaynak], ["kaynak"]
        for vad, ayar in VARYANT.items():
            son = im if ayar is None else geri_koy(kaynak, im, harita, *ayar)
            kim = mo.kimlik_benzerligi(None, son, yuz_a=yuz) or 0
            ok, v = denetim(son, IDDIA2["15 yaş yaşlandır"])
            sat.append(son)
            et.append(f"{vad} {'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}")
            print("PUAN", k, et[-1], flush=True)
        satirlar.append(sat)
        etiketler.append(et)
    izgara(satirlar, etiketler, (220, 220)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
    print("SONUC bitti", flush=True)
