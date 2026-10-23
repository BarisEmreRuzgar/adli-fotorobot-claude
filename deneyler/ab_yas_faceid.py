"""Yaşlandırma: FaceID kapalı yerel çizim (A) ile düşük güçlü FaceID'li bölgesel inpainting (B: kimlik gücü
0.15/0.3/0.45) karşılaştırması; aynı prompt, maske ("yas") ve güç (YAS_GUCU).
Kullanım: python deneyler/ab_yas_faceid.py <ad> [tohum_sayisi]
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

from fotorobot import motor as motor_mod  # noqa: E402
from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import yerel_cizimler  # noqa: E402

if __name__ == "__main__":
    ad = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    mo = motor_mod.Motor()
    mo.yukle()
    gem = Gemini("", mo.ayarlar.gemini_modelleri)
    app = Uygulama(mo, gemini=gem)
    oz = oz_cikar(TEMEL)
    _, liste, _, _ = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, TOHUM1, progress=bos_ilerleme)
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    yeni_oz, _ = gem.degisiklik(liste[0]["oz"], "15 yaş yaşlandır")
    y = yerel_cizimler(liste[0]["oz"], yeni_oz)[0][0]
    motor_mod.BOLGE2_EN_AZ_DEGISIM = 0.0  # yerel çizimle aynı güç kullanılsın
    satirlar, etiketler = [], []
    for k in range(n):
        t = 2026 + k
        sat, et = [kaynak], ["kaynak"]
        adaylar = [("A faceid yok", mo.yerel_ciz(kaynak, y["bolgeler"], y["pozitif"], y["negatif"], adet=1, tohum=t,
                                                 adim=25, cfg=y["cfg"], guc=y["guc"])[0][0])]
        for kg in [float(x) for x in os.environ.get("KG", "0.15,0.3,0.45").split(",")]:
            im = mo.kimlik_koruyarak_uret(kaynak, y["pozitif"], y["negatif"], adet=1, tohum=t, adim=25,
                                          cfg=y["cfg"], kimlik_gucu=kg, degisim=y["guc"],
                                          bolgeler=y["bolgeler"])[0][0][0]
            adaylar.append((f"B fid{kg}", im))
        for vad, im in adaylar:
            kim = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
            ok, v = denetim(im, IDDIA2["15 yaş yaşlandır"])
            sat.append(im)
            et.append(f"{vad} {'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}")
            print("PUAN", k, et[-1], flush=True)
        satirlar.append(sat)
        etiketler.append(et)
    izgara(satirlar, etiketler, (220, 220)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
    print("SONUC bitti", flush=True)
