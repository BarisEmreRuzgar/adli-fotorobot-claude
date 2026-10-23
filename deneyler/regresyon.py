"""Gerileme testi: rötuş açık/kapalı, ölçüm senaryolarının dışındaki yüzlerde (kadın, koyu ten,
kızıl saç, gözlük, kaşta yara) Aşama 1 uyumu ve gerçekçiliği. Kullanım: python deneyler/regresyon.py [ad]
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from uctan_uca import bos_ilerleme, denetim  # noqa: E402

from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402

SENARYOLAR = {
    "kadin_sari": ("30 yaşında bir kadın, uzun sarı saçlı, mavi gözlü.",
                   ["The person is a woman", "She has long blonde hair"]),
    "esmer_kivircik": ("45 yaşında esmer tenli bir erkek, kısa kıvırcık siyah saçlı, kirli sakallı.",
                       ["The man has dark/brown skin", "He has short curly black hair", "He has stubble"]),
    "kizil_kadin": ("25 yaşında bir kadın, omuz hizasında kızıl saçlı, yeşil gözlü.",
                    ["The person is a woman", "She has red/ginger hair"]),
    "kir_gozluk": ("50 yaşında bir erkek, kısa kır saçlı, gözlüklü, kalın bıyıklı.",
                   ["He has grey hair", "He wears glasses", "He has a thick mustache"]),
    "kas_yara": ("30 yaşında bir erkek, kısa kahverengi saçlı, sakalsız. Sağ kaşında yara izi vardı.",
                 ["There is a visible scar through or near the person's right eyebrow (left side of the image)"]),
}

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad = sys.argv[1] if len(sys.argv) > 1 else "reg"
    if len(sys.argv) > 2:
        SENARYOLAR = {k: v for k, v in SENARYOLAR.items() if k in sys.argv[2:]}
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    satirlar, etiketler = [], []
    for anahtar, (ifade, iddialar) in SENARYOLAR.items():
        oz = oz_cikar(ifade)
        for rotus in (False, True):
            app.ayarlar.otomatik_rotus = rotus
            _, liste, metin, _ = app.uret(oz_tablo(oz), ifade, None, 2, 25, 6.0, 777, progress=bos_ilerleme)
            sat, et, ok_say, r = [], [], 0, []
            for x in liste:
                im = Image.open(x["yol"]).convert("RGB")
                ok, v = denetim(im, iddialar)
                ok_say += ok
                r.append(v.get("gercekcilik"))
                sat.append(im)
                et.append(f"{'rotus' if rotus else 'kapali'} {'OK' if ok else 'X'} r{v.get('gercekcilik')}")
            satirlar.append(sat)
            etiketler.append(et)
            rapor = " | ".join(s for s in metin.splitlines() if s.startswith("🎨"))
            print("PUAN", anahtar, "rotus" if rotus else "kapali", f"{ok_say}/2", "r", r, rapor[:120], flush=True)
    izgara(satirlar, etiketler).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
    print("SONUC bitti", flush=True)
