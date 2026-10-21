"""Rötuş deneyi: temel Aşama 1 çıktılarına renk boyama / yara çizimi + düşük güçlü inceltme.

Satırlar: senaryo × (temel, yalnız boyama, boyama + inceltme güç g1, g2 ...).
Kullanım: python deneyler/ab_rotus.py <temel_ad> <ad> [guc1,guc2,...]
"""
import os
import sys

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from PIL import Image  # noqa: E402
from benchmark_and_test import CIKTI, SENARYOLAR, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402
from vlm import denetle  # noqa: E402

from fotorobot import rotus  # noqa: E402
from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import agirlikli  # noqa: E402

temel_ad, ad = sys.argv[1], sys.argv[2]
gucler = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0.3,0.45").split(",")]
NEG = ("worst quality, low quality, blurry, cartoon, painting, drawing, 3d render, deformed, "
       "makeup, tattoo, ink, stripes, tape, bandage")

SECILI = [x for x in os.environ.get("SENARYO", "").split(",") if x] or list(SENARYOLAR)
IDDIALAR = {
    "beyaz_sakal": ["The man has a white or very light grey full beard (not dark)",
                    "The mustache is white or very light grey (not dark)"],
    "sari_kisa": ["The man has short blonde hair"],
    "yara": ["There is a clearly visible healed scar on the person's left cheek (right side of the image)"],
    "pala": ["The man has a thick white or very light grey mustache (not dark)", "The man is bald"],
}
puan = {}
mo = Motor()
mo.yukle()
satirlar, etiketler = [], []
for anahtar in SECILI:
    oz = oz_cikar(SENARYOLAR[anahtar])
    for i in range(4):
        yol = os.path.join(CIKTI, f"{temel_ad}_a1_{anahtar}_{i}.png")
        if not os.path.exists(yol):
            continue
        im = Image.open(yol).convert("RGB")
        yuz = mo.yuz_bul(im)
        if yuz is None:
            continue
        harita = rotus.ayristir(im)
        isler = []  # (boyalı görsel, maske, inceltme prompt'u)
        sac_renk = rotus.renk_bul(oz["sac"])
        if sac_renk and "bald" not in oz["sac"]:
            olc = rotus.sac_rengi_olc(im, harita)
            if olc is not None and rotus.renk_farki(olc, sac_renk) > rotus.RENK_ESIGI:
                b, m = rotus.sac_boya(im, harita, sac_renk)
                if b is not None:
                    isler.append((b, m, f"{agirlikli(sac_renk + ' hair', 1.2)}, natural hair texture, photo"))
        sakal_renk = rotus.renk_bul(oz["sakal"])
        if sakal_renk:
            b, m = rotus.sakal_boya(isler[-1][0] if isler else im, yuz.kps, harita, sakal_renk,
                                    rotus.yalniz_biyik_mi(oz["sakal"]))
            if b is not None:
                isler.append((b, m, f"{agirlikli(oz['sakal'], 1.15)}, natural facial hair strands, photo"))
        _, yaralar = rotus.yara_konumlari(oz["ozel"])
        for yer, taraf in yaralar:
            b, m = rotus.yara_ciz(isler[-1][0] if isler else im, yuz.kps,
                                  rotus.yara_yolu(yuz.kps, yer, taraf), harita, tohum=i)
            isler.append((b, m, f"{agirlikli('healed scar', 1.1)}, skin, photo"))
        if not isler:
            print("SONUC", anahtar, i, "rötuş gerekmedi", flush=True)
            continue
        boyali = isler[-1][0]
        # Tüm değişen bölgelerin birleşimi tek inceltmede (her iş kendi prompt'uyla sırayla)
        sat, et = [im, boyali], ["temel", "boyali"]
        for g in gucler:
            cur = boyali
            for _, m, poz in isler:
                mk = m if isinstance(m, Image.Image) else Image.fromarray((m * 255).astype("uint8"))
                mk = mk.filter(__import__("PIL.ImageFilter", fromlist=["x"]).MaxFilter(7))
                cur = mo.bolge_duzenle(cur, mk, poz, NEG, adet=1, tohum=20261021 + i, adim=25, cfg=6.0,
                                       guc=g)[0][0]
            ben = mo.kimlik_benzerligi(None, cur, yuz_a=yuz)
            v = denetle(cur, IDDIALAR[anahtar])
            p = puan.setdefault(f"{anahtar} g{g}", [0, 0, []])
            p[0] += all(v["kontroller"]) and len(v["kontroller"]) == len(IDDIALAR[anahtar])
            p[1] += 1
            p[2].append(v.get("gercekcilik"))
            print("VLM", anahtar, i, g, v, flush=True)
            sat.append(cur)
            h2 = rotus.ayristir(cur)
            s = rotus.sac_rengi_olc(cur, h2)
            et.append(f"g{g} id{ben or 0:.2f} {'OK' if all(v['kontroller']) and v['kontroller'] else 'X'}"
                      f" r{v.get('gercekcilik')}" + (f" sacL{s[0]:.0f}b{s[2]:.0f}" if s is not None else ""))
            cur.save(os.path.join(CIKTI, f"{ad}_{anahtar}_{i}_g{g}.png"))
        satirlar.append(sat)
        etiketler.append(et)
        print("SONUC", anahtar, i, et, flush=True)
for k, (ok, n, r) in puan.items():
    print("PUAN", k, f"{ok}/{n}", "gercekcilik", r, flush=True)
izgara(satirlar, etiketler).save(os.path.join(CIKTI, f"{ad}.png"))
print("SONUC bitti", flush=True)
