"""Mevcut boru hattının adli senaryolarla ölçümü.

Aşama 1: Türkçe tanık ifadesi → kural analizi → 4 tohum. Ölçüm: saç/sakal rengi (Lab).
Aşama 2: temel yüz + değişiklik komutu → 3 varyasyon. Ölçüm: ArcFace kimlik benzerliği + renk.
Çıktı: deneyler/sonuc/<ad>_*.png ve <ad>.json
Kullanım: python deneyler/benchmark_and_test.py [asama1|asama2|hepsi] [ad]
"""
import json
import os
import sys
import time

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from olcum import ayristir, izgara, sac_rengi, sakal_olcumu  # noqa: E402

from fotorobot.kural_analiz import degisiklik_uygula, ozellik_cikar  # noqa: E402
from fotorobot.ozellikler import degisiklik_bolgeleri, normalize, prompt_olustur  # noqa: E402

CIKTI = os.path.join(KOK, "deneyler", "sonuc")
os.makedirs(CIKTI, exist_ok=True)

SENARYOLAR = {
    "beyaz_sakal": "55 yaşlarında bir adamdı. Kısa siyah saçları vardı ama sakalı bembeyaz ve gürdü. Kalın kaşlıydı.",
    "sari_kisa": "Yaklaşık 35 yaşında bir erkek. Kısa sarı saçlı, sakalsız, mavi gözlü.",
    "yara": "40 yaşlarında bir adam, kısa kahverengi saçlı, sakalsız. Sol yanağında belirgin bir yara izi vardı.",
    "pala": "60 yaşında kel bir adam, beyaz pala bıyıklı, kalın kara kaşlı.",
}
TEMEL = "35 yaşında bir erkek, kısa koyu kahverengi saçlı, sakalsız, kahverengi gözlü."
DEGISIKLIKLER = ["saçları sarı yap", "beyaz sakal ekle", "burnunu kemerli yap", "15 yaş yaşlandır",
                 "sol yanağına yara izi ekle"]
TOHUM1, TOHUM2 = 4242, 2026


def oz_cikar(metin):
    s = ozellik_cikar(metin)
    return normalize(s[0] if isinstance(s, tuple) else s)


def degistir(oz, komut):
    s = degisiklik_uygula(oz, komut)
    return s[0] if isinstance(s, tuple) else s


def renk_ozeti(mo, im):
    harita = ayristir(im)
    yuz = mo.yuz_bul(im)
    s = sac_rengi(im, harita)
    k = sakal_olcumu(im, yuz.kps, harita) if yuz is not None else None
    return {"sac": s and {x: round(s[x], 1) for x in ("L", "a", "b")},
            "sakal": k and {x: round(k[x], 1) for x in k}}


def etiket(r, ek=""):
    s, k = r["sac"], r["sakal"]
    t = ek
    if s:
        t += f" sacL{s['L']:.0f} b{s['b']:.0f}"
    if k:
        t += f" skdL{k['dL']:.0f}"
    return t.strip()


def asama1(mo, ad):
    sonuc, satirlar, etiketler = {}, [], []
    for anahtar, ifade in SENARYOLAR.items():
        oz = oz_cikar(ifade)
        poz, neg = prompt_olustur(oz)
        t0 = time.time()
        gorseller = [im for im, _ in mo.metinden_uret(poz, neg, adet=4, tohum=TOHUM1, adim=25, cfg=6.0)]
        olcumler = [renk_ozeti(mo, im) for im in gorseller]
        sonuc[anahtar] = {"oz": oz, "olcum": olcumler, "sure": round(time.time() - t0, 1)}
        for i, im in enumerate(gorseller):
            im.save(os.path.join(CIKTI, f"{ad}_a1_{anahtar}_{i}.png"))
        satirlar.append(gorseller)
        etiketler.append([etiket(r, anahtar[:6]) for r in olcumler])
        print("SONUC", anahtar, json.dumps(olcumler), flush=True)
    izgara(satirlar, etiketler).save(os.path.join(CIKTI, f"{ad}_asama1.png"))
    return sonuc


def asama2(mo, ad, duzenle=None):
    """duzenle(mo, kaynak, eski_oz, yeni_oz, komut) → [PIL]; None ise mevcut Aşama 2."""
    oz = oz_cikar(TEMEL)
    poz, neg = prompt_olustur(oz)
    kaynak = mo.metinden_uret(poz, neg, adet=1, tohum=TOHUM1, adim=25, cfg=6.0)[0][0]
    kaynak.save(os.path.join(CIKTI, f"{ad}_a2_kaynak.png"))
    kaynak_yuz = mo.yuz_bul(kaynak)
    sonuc, satirlar, etiketler = {}, [], []
    for komut in DEGISIKLIKLER:
        yeni = degistir(oz, komut)
        if duzenle is None:
            p2, n2 = prompt_olustur(yeni, onceki=oz, referans=oz)
            s, _ = mo.kimlik_koruyarak_uret(kaynak, p2, n2, adet=3, tohum=TOHUM2, adim=25, cfg=6.0,
                                            kimlik_gucu=0.6, degisim=0.8, kimlik_gorseli=kaynak,
                                            sac_maskele=yeni["sac"] != oz["sac"],
                                            bolgeler=degisiklik_bolgeleri(oz, yeni))
            gorseller = [im for im, _, _ in s]
        else:
            gorseller = duzenle(mo, kaynak, oz, yeni, komut)
        olcum = []
        for im in gorseller:
            r = renk_ozeti(mo, im)
            r["kimlik"] = round(mo.kimlik_benzerligi(None, im, yuz_a=kaynak_yuz) or 0, 3)
            olcum.append(r)
        sonuc[komut] = {"degisen": {k: yeni[k] for k in yeni if yeni[k] != oz[k]}, "olcum": olcum}
        for i, im in enumerate(gorseller):
            im.save(os.path.join(CIKTI, f"{ad}_a2_{DEGISIKLIKLER.index(komut)}_{i}.png"))
        satirlar.append([kaynak] + gorseller)
        etiketler.append(["kaynak " + etiket(renk_ozeti(mo, kaynak))]
                         + [etiket(r, f"id{r['kimlik']:.2f}") for r in olcum])
        print("SONUC", komut, json.dumps(sonuc[komut], ensure_ascii=False), flush=True)
    izgara(satirlar, etiketler).save(os.path.join(CIKTI, f"{ad}_asama2.png"))
    return sonuc


if __name__ == "__main__":
    from fotorobot.motor import Motor

    hangi = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    ad = sys.argv[2] if len(sys.argv) > 2 else "temel"
    mo = Motor()
    mo.yukle()
    rapor = {}
    if hangi in ("asama1", "hepsi"):
        rapor["asama1"] = asama1(mo, ad)
    if hangi in ("asama2", "hepsi"):
        rapor["asama2"] = asama2(mo, ad)
    with open(os.path.join(CIKTI, f"{ad}.json"), "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print("SONUC bitti", flush=True)
