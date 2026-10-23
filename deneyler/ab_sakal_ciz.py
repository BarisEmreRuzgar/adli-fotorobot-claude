"""Aşama 1 sakal rengi tutmadığında: piksel boyama (rötuş) ile sakal bölgesini yerel çizimle yeniden çizme
(Aşama 2 sakal ekleme yolu: FaceID'siz, renge göre aday, burun+dudak korunur) karşılaştırması.
Kullanım: python deneyler/ab_sakal_ciz.py <ad> <senaryo>
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import SENARYOLAR, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from uctan_uca import IDDIA1, bos_ilerleme, denetim  # noqa: E402

from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402
from fotorobot.ozellikler import yerel_cizimler  # noqa: E402

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad, senaryo = sys.argv[1], sys.argv[2]
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    app.ayarlar.otomatik_rotus = False
    oz = oz_cikar(SENARYOLAR[senaryo])
    _, liste, _, _ = app.uret(oz_tablo(oz), SENARYOLAR[senaryo], None, 2, 25, 6.0, 4242, progress=bos_ilerleme)
    satirlar, etiketler = [], []
    for x in liste:
        ham = Image.open(x["yol"]).convert("RGB")
        yuz = mo.yuz_bul(ham)
        boyali, _ = mo.rotus_uygula(ham, x["oz"], tohum=x["tohum"])
        y = yerel_cizimler(dict(x["oz"], sakal=""), x["oz"])[0][0]
        cizili, kim = mo.yerel_ciz_sec(ham, y, x["tohum"], 25, yuz)
        # Saç rengi vb. diğer rötuşlar yine uygulanır; sakal rengi atlanır
        cizili, _ = mo.rotus_uygula(cizili, x["oz"], tohum=x["tohum"], atla=("sakal",))
        sat, et = [ham], ["ham"]
        for vad, im in (("boya", boyali), ("ciz", cizili)):
            ok, v = denetim(im, IDDIA1[senaryo])
            k = mo.kimlik_benzerligi(None, im, yuz_a=yuz) or 0
            sat.append(im)
            et.append(f"{vad} {'OK' if ok else 'X'} id{k:.2f} r{v.get('gercekcilik')}")
            print("PUAN", et[-1], v.get("model"), (v.get("not") or "")[:100], flush=True)
        satirlar.append(sat)
        etiketler.append(et)
    izgara(satirlar, etiketler, (300, 375)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
