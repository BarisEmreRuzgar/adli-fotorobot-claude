"""Uçtan uca ölçüm: arayüzün kendi akışı (kural analizi → Aşama 1 → Aşama 2 → rötuş).

Her çıktı Gemini ile denetlenir (istenen ayrıntı var mı + gerçekçilik), Aşama 2'de ArcFace
kimlik benzerliği de raporlanır. Kullanım: python deneyler/uctan_uca.py [asama1|asama2|hepsi] [ad]
"""
import json
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import DEGISIKLIKLER, SENARYOLAR, TEMEL, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from PIL import Image  # noqa: E402
from vlm import denetle  # noqa: E402

from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402

IDDIA1 = {
    "beyaz_sakal": ["The man has a white or very light grey full beard (not dark)",
                    "The mustache is white or very light grey (not dark)"],
    "sari_kisa": ["The man has short blonde hair"],
    "yara": ["There is a clearly visible healed scar on the person's left cheek (right side of the image)"],
    "pala": ["The man has a thick white or very light grey mustache (not dark)", "The man is bald"],
}
IDDIA2 = {
    "saçları sarı yap": ["The man has blonde hair"],
    "beyaz sakal ekle": ["The man has a white or light grey full beard"],
    "burnunu kemerli yap": ["The man has a clearly hooked, aquiline nose with a visible bump on the bridge"],
    "15 yaş yaşlandır": ["The man clearly looks about 50 years old (middle-aged, visible wrinkles)"],
    "sol yanağına yara izi ekle": ["There is a clearly visible healed scar on the person's left cheek "
                                   "(right side of the image)"],
}
CIKTI = os.path.join(KOK, "deneyler", "sonuc")


def bos_ilerleme(*a, **k):
    return None


def denetim(im, iddialar):
    """OYLAMA ortam değişkeni (varsayılan 1) kadar yargı alır: her iddia çoğunlukla, gerçekçilik ortanca.
    Aynı görsele Gemini bazen farklı karar veriyordu (yara uu5 4/4, uu27 aynı görseller 0/2)."""
    oylar = [denetle(im, iddialar) for _ in range(int(os.environ.get("OYLAMA", 1)))]
    gecerli = [o for o in oylar if len(o["kontroller"]) == len(iddialar)] or oylar[:1]
    v = dict(gecerli[0])
    if len(gecerli) > 1:
        v["kontroller"] = [sum(o["kontroller"][i] for o in gecerli) * 2 > len(gecerli) for i in range(len(iddialar))]
        puanlar = sorted(o["gercekcilik"] for o in gecerli if o.get("gercekcilik") is not None)
        v["gercekcilik"] = puanlar[len(puanlar) // 2] if puanlar else None
        v["oylar"] = [o["kontroller"] for o in gecerli]
    ok = bool(v["kontroller"]) and len(v["kontroller"]) == len(iddialar) and all(v["kontroller"])
    return ok, v


def asama1(app, ad):
    rapor, satirlar, etiketler = {}, [], []
    for anahtar, ifade in SENARYOLAR.items():
        oz = oz_cikar(ifade)
        _, liste, metin, _ = app.uret(oz_tablo(oz), ifade, None, int(os.environ.get("ADET", 4)), 25, 6.0, 4242,
                                       progress=bos_ilerleme)
        print("RAPOR", anahtar, " | ".join(s for s in metin.splitlines() if s.startswith(("🎨", "🖌", "❌"))),
              flush=True)
        sat, et, sonuc = [], [], []
        for x in liste:
            im = Image.open(x["yol"]).convert("RGB")
            ok, v = denetim(im, IDDIA1[anahtar])
            sonuc.append({"ok": ok, "v": v})
            sat.append(im)
            et.append(f"{'OK' if ok else 'X'} r{v.get('gercekcilik')}")
        rapor[anahtar] = sonuc
        satirlar.append(sat)
        etiketler.append(et)
        print("PUAN1", anahtar, f"{sum(s['ok'] for s in sonuc)}/{len(sonuc)}",
              "gercekcilik", [s["v"].get("gercekcilik") for s in sonuc], flush=True)
    izgara(satirlar, etiketler).save(os.path.join(CIKTI, f"{ad}_uu1.png"))
    return rapor


def asama2(app, ad):
    oz = oz_cikar(TEMEL)
    _, liste, _, vaka = app.uret(oz_tablo(oz), TEMEL, None, 1, 25, 6.0, 4242, progress=bos_ilerleme)
    kaynak = liste[0]["yol"]
    rapor, satirlar, etiketler = {}, [], []
    secili = [k for k in os.environ.get("KOMUT", "").split(",") if k]
    for komut in [k for k in DEGISIKLIKLER if not secili or k in secili]:
        _, liste2, metin, _ = app.duzenle(kaynak, kaynak, oz_tablo(oz), komut, vaka, [], 0.6, 0.8, int(os.environ.get("ADET", 3)), 25, 6.0,
                                          2026, progress=bos_ilerleme)
        print("RAPOR", komut, " | ".join(s for s in metin.splitlines() if s.startswith(("🎨", "ℹ️ Yalnız", "❌"))),
              flush=True)
        sat, et, sonuc = [Image.open(kaynak)], ["kaynak"], []
        for x in liste2:
            im = Image.open(x["yol"]).convert("RGB")
            ok, v = denetim(im, IDDIA2[komut])
            sonuc.append({"ok": ok, "kimlik": x.get("benzerlik"), "v": v})
            print("NOT", komut, v.get("model"), v.get("kontroller"), (v.get("not") or "")[:140], flush=True)
            sat.append(im)
            et.append(f"{'OK' if ok else 'X'} id{x.get('benzerlik') or 0:.2f} r{v.get('gercekcilik')}")
        rapor[komut] = sonuc
        satirlar.append(sat)
        etiketler.append(et)
        print("PUAN2", komut, f"{sum(s['ok'] for s in sonuc)}/{len(sonuc)}",
              "kimlik", [round(s["kimlik"] or 0, 2) for s in sonuc],
              "gercekcilik", [s["v"].get("gercekcilik") for s in sonuc], flush=True)
    izgara(satirlar, etiketler).save(os.path.join(CIKTI, f"{ad}_uu2.png"))
    return rapor


if __name__ == "__main__":
    from fotorobot.motor import Motor

    hangi = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    ad = sys.argv[2] if len(sys.argv) > 2 else "uu"
    mo = Motor()
    mo.yukle()
    # Kural tabanlı analiz: deterministik ve Gemini kotası denetime kalsın
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    sonuc = {}
    if hangi in ("asama1", "hepsi"):
        sonuc["asama1"] = asama1(app, ad)
    if hangi in ("asama2", "hepsi"):
        sonuc["asama2"] = asama2(app, ad)
    with open(os.path.join(CIKTI, f"{ad}_uu.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=1, default=str)
    print("SONUC bitti", flush=True)
