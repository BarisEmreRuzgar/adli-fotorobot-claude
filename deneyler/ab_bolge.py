"""Aşama 2: genel düzenleme ile bölgesel düzenlemeyi aynı kaynak/tohumla karşılaştırır."""
import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)
from PIL import Image  # noqa: E402

from fotorobot.kural_analiz import degisiklik_uygula  # noqa: E402
from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import degisiklik_bolgeleri, normalize, prompt_olustur  # noqa: E402


def asama1(vaka, sira):
    ol = json.load(open(f'ciktilar/{vaka}/rapor.json'))
    o = [o for o in ol if o['islem'] == 'asama1_uretim'][-1]
    return Image.open(o['gorseller'][sira]).convert('RGB'), normalize(o['ozellikler'])


vakalar = [
    ('sakal_sac', 'vaka_20261018_035450', 1, 'sakalı temizle\nsaçları sarı yap', 0.8),
    ('gozluk_cikar', 'vaka_20261018_035450', 1, 'gözlüğü çıkar', 1.0),
    ('gozluk_ekle', 'vaka_20261018_041513', 1, 'gözlük ekle', 0.8),
]
mo = Motor()
mo.yukle()
satirlar = []
for ad, vaka, sira, komut, degisim in vakalar:
    src, oz = asama1(vaka, sira)
    yeni, _ = degisiklik_uygula(oz, komut)
    poz, neg = prompt_olustur(yeni, onceki=oz, referans=oz)
    bolgeler = degisiklik_bolgeleri(oz, yeni)
    for etiket, b in (('GENEL', None), ('BOLGE', bolgeler)):
        s, _ = mo.kimlik_koruyarak_uret(src, poz, neg, adet=3, tohum=2026, adim=25, cfg=6.0,
                                         kimlik_gucu=0.6, degisim=degisim, kimlik_gorseli=src,
                                         sac_maskele=yeni['sac'] != oz['sac'], bolgeler=b)
        satirlar.append([src] + [im for im, _, _ in s])
        print('SONUC', ad, etiket, sorted(b) if b else '-', [round(x or 0, 3) for _, _, x in s], flush=True)
g = Image.new('RGB', (4 * 160, len(satirlar) * 200), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((160, 200)), (c * 160, r * 200))
g.save('/content/ab_bolge.png')
print('SONUC grid')
