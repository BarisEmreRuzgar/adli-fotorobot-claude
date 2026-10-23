"""Aşama 2 sakal eklemenin farklı yüzlerde genellenmesi: her tarif için bir yüz üretilir, sonra komut uygulanır.
Kullanım: python deneyler/sakal_genel.py <ad> [tarif_no...]   (KOMUT ortam değişkeni, varsayılan "beyaz sakal ekle")
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
from uctan_uca import IDDIA2, bos_ilerleme, denetim  # noqa: E402

from fotorobot.arayuz import Uygulama, oz_tablo  # noqa: E402
from fotorobot.gemini import Gemini  # noqa: E402

TARIFLER = [
    "45 yaşında esmer tenli bir erkek, kısa kıvırcık siyah saçlı.",
    "35 yaşında bir erkek, kısa sarı saçlı, mavi gözlü.",
    "50 yaşında kel bir erkek, kalın kaşlı.",
    "28 yaşında bir erkek, orta uzunlukta kahverengi dalgalı saçlı, ince yüzlü.",
]

if __name__ == "__main__":
    from fotorobot.motor import Motor

    ad = sys.argv[1]
    secili = [TARIFLER[int(i)] for i in sys.argv[2:]] or TARIFLER
    komut = os.environ.get("KOMUT", "beyaz sakal ekle")
    mo = Motor()
    mo.yukle()
    app = Uygulama(mo, gemini=Gemini("", mo.ayarlar.gemini_modelleri))
    sat, et, ok_say = [], [], 0
    for tarif in secili:
        oz = oz_cikar(tarif)
        _, liste, _, vaka = app.uret(oz_tablo(oz), tarif, None, 1, 25, 6.0, 4242, progress=bos_ilerleme)
        kaynak = liste[0]["yol"]
        _, liste2, metin, _ = app.duzenle(kaynak, kaynak, oz_tablo(oz), komut, vaka, [], 0.6, 0.8, 1, 25, 6.0, 2026,
                                      progress=bos_ilerleme)
        print("RAPOR", " | ".join(s for s in metin.splitlines() if s.startswith(("🎨", "🖌️", "ℹ️ Yalnız", "❌"))), flush=True)
        im = Image.open(liste2[0]["yol"]).convert("RGB")
        ok, v = denetim(im, IDDIA2[komut])
        ok_say += ok
        sat += [Image.open(kaynak), im]
        et += ["kaynak", f"{'OK' if ok else 'X'} id{liste2[0].get('benzerlik') or 0:.2f} r{v.get('gercekcilik')}"]
        print("PUAN", tarif[:30], et[-1], v.get("not", "")[:90], flush=True)
    print("TOPLAM", f"{ok_say}/{len(secili)}", flush=True)
    izgara([sat[:4], sat[4:]] if len(sat) > 4 else [sat], [et[:4], et[4:]] if len(et) > 4 else [et],
           (220, 220)).save(os.path.join(KOK, "deneyler", "sonuc", f"{ad}.png"))
