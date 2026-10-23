"""Kemerli burun bükmesinin farklı yüzlerde genellenmesi: son Aşama 1 çıktılarına uygulanır.
Kullanım: python deneyler/burun_genel.py <ad> <görsel>...
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from uctan_uca import IDDIA2, denetim  # noqa: E402

from fotorobot import rotus  # noqa: E402

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad, yollar = sys.argv[1], sys.argv[2:]
    mo = Motor()
    mo._yuz_analiz_yukle()
    sat, et, ok_say = [], [], 0
    for yol in yollar:
        g = Image.open(yol).convert("RGB")
        yuz = mo.yuz_bul(g)
        im, _ = rotus.burun_kemerli(g, yuz.kps, rotus.BURUN_BUKME_GUCU)
        kim = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
        ok, v = denetim(im, [IDDIA2["burnunu kemerli yap"][0].replace("The man", "The person")])
        ok_say += ok
        sat += [g, im]
        et += ["kaynak", f"{'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}"]
        print("PUAN", os.path.basename(os.path.dirname(yol)), et[-1], v.get("not", "")[:90], flush=True)
    print("TOPLAM", f"{ok_say}/{len(yollar)}", flush=True)
    izgara([sat], [et], (200, 200)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
