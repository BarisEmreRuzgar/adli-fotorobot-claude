"""Yerel çizimde (sakal ekle, yaşlandır) kimlik skoru tohumdan tohuma ne kadar oynuyor?
Oynuyorsa N aday çizip ilk yüze en benzeyeni seçmek kimliği ucuza yükseltir.
Kullanım: python deneyler/ab_kimlik_secim.py <ad> [aday_sayisi]
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import DEGISIKLIKLER, TEMEL, TOHUM1, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from uctan_uca import bos_ilerleme  # noqa: E402

from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import yerel_cizimler  # noqa: E402

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    mo = Motor()
    mo.yukle()
    gem = Gemini("", mo.ayarlar.gemini_modelleri)
    app = Uygulama(mo, gemini=gem)
    oz = oz_cikar(TEMEL)
    _, liste, _, _ = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, TOHUM1, progress=bos_ilerleme)
    from PIL import Image
    kaynak = Image.open(liste[0]["yol"]).convert("RGB")
    yuz = mo.yuz_bul(kaynak)
    satirlar, etiketler = [], []
    for degisiklik in ("beyaz sakal ekle", "15 yaş yaşlandır"):
        yeni, _ = gem.degisiklik(liste[0]["oz"], degisiklik)
        yereller, _ = yerel_cizimler(liste[0]["oz"], yeni)
        y = yereller[0]
        sat, et, skorlar = [kaynak], ["kaynak"], []
        for k in range(n):
            im = mo.yerel_ciz(kaynak, y["bolgeler"], y["pozitif"], y["negatif"], adet=1, tohum=TOHUM1 + k,
                              adim=25, cfg=y["cfg"], guc=y["guc"])[0][0]
            kim = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
            skorlar.append(round(kim, 3))
            sat.append(im)
            et.append(f"t{TOHUM1 + k} id{kim:.2f}")
        satirlar.append(sat)
        etiketler.append(et)
        print("PUAN", degisiklik, skorlar, "en iyi", max(skorlar), "ort", round(sum(skorlar) / n, 3), flush=True)
    izgara(satirlar, etiketler, (200, 200)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
    print("SONUC bitti", flush=True)
