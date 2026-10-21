"""Aşama 2: istenen değişikliğin uygulanması (sakal ekleme, kemerli burun, yaşlandırma) için varyantlar.

Her komut × varyant × 3 tohum; ölçüm: ArcFace kimlik + Gemini denetimi. Rötuş (renk) hepsine uygulanır.
Kullanım: python deneyler/ab_asama2.py <kaynak_png> <ad>
"""
import json
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import TEMEL, degistir, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from vlm import denetle  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import agirlikli, degisiklik_bolgeleri, prompt_olustur  # noqa: E402

kaynak_yol, ad = sys.argv[1], sys.argv[2]
IDDIA = {
    "beyaz sakal ekle": ["The man has a white or light grey full beard"],
    "burnunu kemerli yap": ["The man has a clearly hooked, aquiline nose with a visible bump on the bridge"],
    "15 yaş yaşlandır": ["The man clearly looks about 50 years old (middle-aged, visible wrinkles)"],
}


def vurgula(poz, ifade, agirlik):
    return poz.replace(ifade, agirlikli(ifade, agirlik), 1) if ifade in poz else f"{agirlikli(ifade, agirlik)}, {poz}"


def varyantlar(komut, oz, yeni):
    """(ad, üretim özellikleri, prompt dönüştürücü, kimlik gücü, değişim)"""
    v = [("V0 mevcut", yeni, None, 0.6, 0.8)]
    if komut == "beyaz sakal ekle":
        renksiz = dict(yeni, sakal="full beard")
        v += [("V1 renksiz", renksiz, None, 0.6, 0.8), ("V2 renksiz k.45", renksiz, None, 0.45, 0.8)]
    elif komut == "burnunu kemerli yap":
        f = lambda p: vurgula(p, yeni["burun"], 1.3) + ", prominent bump on the nose bridge"  # noqa: E731
        v += [("V1 agirlik", yeni, f, 0.6, 0.8), ("V2 agirlik k.45", yeni, f, 0.45, 0.8)]
    else:
        from fotorobot.ozellikler import _yas_gorunumu
        f = lambda p: vurgula(p, _yas_gorunumu(yeni["yas"]), 1.3)  # noqa: E731
        v += [("V1 agirlik", yeni, f, 0.6, 0.8), ("V2 agirlik d.9 k.5", yeni, f, 0.5, 0.9)]
    return v


mo = Motor()
mo.yukle()
kaynak = Image.open(kaynak_yol).convert("RGB")
kaynak_yuz = mo.yuz_bul(kaynak)
oz = oz_cikar(TEMEL)
satirlar, etiketler, rapor = [], [], {}
for komut in IDDIA:
    yeni = degistir(oz, komut)
    for vad, uretim, donustur, kg, dg in varyantlar(komut, oz, yeni):
        poz, neg = prompt_olustur(uretim, onceki=oz, referans=oz)
        if donustur:
            poz = donustur(poz)
        s, _ = mo.kimlik_koruyarak_uret(kaynak, poz, neg, adet=3, tohum=2026, adim=25, cfg=6.0, kimlik_gucu=kg,
                                        degisim=dg, kimlik_gorseli=kaynak, sac_maskele=False,
                                        bolgeler=degisiklik_bolgeleri(oz, uretim))
        sat, et, puan = [kaynak], [vad], []
        for im, t, _ in s:
            im, yapilan = mo.rotus_uygula(im, yeni, tohum=t, adim=25)
            kim = mo.kimlik_benzerligi(None, im, yuz_a=kaynak_yuz) or 0
            v = denetle(im, IDDIA[komut])
            ok = bool(v["kontroller"]) and all(v["kontroller"])
            puan.append({"ok": ok, "kimlik": round(kim, 3), "r": v.get("gercekcilik"), "rotus": yapilan})
            sat.append(im)
            et.append(f"{'OK' if ok else 'X'} id{kim:.2f} r{v.get('gercekcilik')}")
        rapor[f"{komut} | {vad}"] = puan
        satirlar.append(sat)
        etiketler.append(et)
        print("PUAN", komut, vad, f"{sum(p['ok'] for p in puan)}/3", "kimlik", [p["kimlik"] for p in puan],
              "r", [p["r"] for p in puan], flush=True)
izgara(satirlar, etiketler).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
json.dump(rapor, open(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("SONUC bitti", flush=True)
