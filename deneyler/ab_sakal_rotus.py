"""Sakal rötuşu A/B: aynı ham görsellere (rötuş kapalı) rötuş, rotus modülündeki bir sabitin (PARAM ortam
değişkeni, varsayılan SAKAL_YAN_UST) farklı değerleriyle.
Kullanım: python deneyler/ab_sakal_rotus.py <ad> <senaryo> <değer...>   (senaryo: uctan_uca SENARYOLAR anahtarı)
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

from fotorobot import rotus  # noqa: E402
from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad, senaryo = sys.argv[1], sys.argv[2]
    param = os.environ.get("PARAM", "SAKAL_YAN_UST")
    esikler = [float(x) for x in sys.argv[3:]]
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    app.ayarlar.otomatik_rotus = False
    oz = oz_cikar(SENARYOLAR[senaryo])
    _, liste, _, _ = app.uret(oz_tablo(oz), SENARYOLAR[senaryo], None, 2, 25, 6.0, 4242, progress=bos_ilerleme)
    satirlar, etiketler = [], []
    for x in liste:
        ham = Image.open(x["yol"]).convert("RGB")
        sat, et = [ham], ["ham"]
        for esik in esikler:
            setattr(rotus, param, esik)
            im, yapilan = mo.rotus_uygula(ham, x["oz"], tohum=x.get("tohum", 4242))
            ok, v = denetim(im, IDDIA1[senaryo])
            sat.append(im)
            et.append(f"e{esik} {'OK' if ok else 'X'} r{v.get('gercekcilik')}")
            print("PUAN", et[-1], v.get("model"), (v.get("not") or "")[:100], flush=True)
        satirlar.append(sat)
        etiketler.append(et)
    izgara(satirlar, etiketler, (300, 375)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
