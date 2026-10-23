"""Beyaz sakal eklemede kimlik kaybı nereden geliyor: üretilen görsele kaynağın bölgelerini (gözler+kaşlar,
burun, dudaklar, sakal dışı her şey) geri yapıştırıp ArcFace benzerliğindeki değişimi ölçer.
Kullanım: python deneyler/sakal_kimlik_analiz.py [varyasyon_sayisi]
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

BOLGELER = {
    "goz_kas": ("l_eye", "r_eye", "l_brow", "r_brow"),
    "burun": ("nose",),
    "kulak_sac": ("l_ear", "r_ear", "hair"),
    "dudak": ("u_lip", "l_lip", "mouth"),
}


def yapistir(hedef, kaynak, maske):
    m = rotus._yumusat(maske.astype(np.float64), 2.0)[..., None]
    return Image.fromarray((np.asarray(hedef) * (1 - m) + np.asarray(kaynak) * m).astype(np.uint8))


if __name__ == "__main__":
    from fotorobot.motor import Motor

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    tarif = os.environ.get("TARIF", TEMEL)
    oz = oz_cikar(tarif)
    _, liste, _, vaka = app.uret(oz_tablo(oz), tarif, None, 1, 25, 6.0, 4242, progress=bos_ilerleme)
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    hk = rotus.ayristir(kaynak)
    _, liste2, _, _ = app.duzenle(liste[0]["yol"], liste[0]["yol"], oz_tablo(oz), "beyaz sakal ekle", vaka, [],
                                  0.6, 0.8, n, 25, 6.0, 2026, progress=bos_ilerleme)
    satirlar, etiketler = [], []
    for x in liste2:
        im = Image.open(x["yol"]).convert("RGB")
        hy = rotus.ayristir(im)
        sat, et = [im], [f"cikti id{mo.kimlik_benzerligi(None, im, yuz_a=yuz):.2f}"]
        for ad, siniflar in BOLGELER.items():
            sn = [rotus.SINIF[s] for s in siniflar]
            v = yapistir(im, kaynak, np.isin(hk, sn) | np.isin(hy, sn))
            sat.append(v)
            et.append(f"{ad} id{mo.kimlik_benzerligi(None, v, yuz_a=yuz):.2f}")
        # sakal dışında her şey kaynaktan: yalnızca sakal pikselleri (belirgin fark) kalır
        fark = np.abs(np.asarray(im, np.float64) - np.asarray(kaynak, np.float64)).mean(-1)
        v = yapistir(kaynak, im, rotus._yumusat((fark > 25).astype(np.float64), 1.5) > 0.3)
        sat.append(v)
        et.append(f"yalniz_sakal id{mo.kimlik_benzerligi(None, v, yuz_a=yuz):.2f}")
        print("ANALIZ", " | ".join(et), flush=True)
        satirlar.append(sat)
        etiketler.append(et)
    izgara(satirlar, etiketler, (220, 220)).save(os.path.join(KOK, "deneyler", "sonuc", "sakal_analiz.png"))
    print("SONUC bitti", flush=True)
