import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)  # vaka dosyaları burada
from PIL import Image, ImageFilter  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import BOLGE_ORNEKLERI, bolge_promptu  # noqa: E402

ESKI_YARA = ('prominent visible pale white scar, raised scar tissue, old knife wound scar line on the '
             'forehead skin, RAW photo, close-up of human skin, photorealistic, natural skin texture, '
             'sharp focus, even studio lighting')
NEG = 'worst quality, low quality, blurry, jpeg artifacts, cartoon, painting, drawing, 3d render, deformed, text, watermark, makeup'


def vaka(ad):
    ol = json.load(open(f'ciktilar/{ad}/rapor.json'))
    return ol


mo = Motor()
mo.yukle()
satirlar = []

# Ben: genç erkek, ağız yanı, normal fırça
ol = vaka('vaka_20261018_041513')
a2 = [o for o in ol if o['islem'] == 'asama2_duzenleme'][-1]
a3 = [o for o in ol if o['islem'] == 'asama3_bolge_duzenleme'][-1]
src = Image.open(a2['gorseller'][0]).convert('RGB')
m0 = Image.open(a3['maske']).convert('L')
poz, neg = bolge_promptu(BOLGE_ORNEKLERI['Ben'])
for ad, m in (('ben_r12', m0), ('ben_r16', m0.filter(ImageFilter.MaxFilter(9)))):
    s = mo.bolge_duzenle(src, m, poz, neg, adet=6, tohum=777, adim=25, cfg=8, guc=1.0)
    satirlar.append([im.crop((330, 300, 440, 410)) for im, _, _ in s])
    print('SONUC', ad, [round(b, 3) for _, _, b in s], flush=True)

# Yara izi: dün çalışan vaka (alın), aynı geniş maske; eski tarif vs yeni hazır tarif
ol = vaka('vaka_20261018_035450')
a1 = [o for o in ol if o['islem'] == 'asama1_uretim'][-1]
a3 = [o for o in ol if o['islem'] == 'asama3_bolge_duzenleme'][-1]
src = Image.open(a1['gorseller'][1]).convert('RGB')
m = Image.open(a3['maske']).convert('L')
x0, y0, x1, y1 = m.getbbox()
kutu = (max(0, (x0 + x1) // 2 - 55), max(0, (y0 + y1) // 2 - 55))
kutu = kutu + (kutu[0] + 110, kutu[1] + 110)
for ad, (p, n) in (('yara_eski', (ESKI_YARA, NEG)), ('yara_yeni', bolge_promptu(BOLGE_ORNEKLERI['Yara izi']))):
    s = mo.bolge_duzenle(src, m, p, n, adet=6, tohum=20261018, adim=25, cfg=8, guc=1.0)
    satirlar.append([im.crop(kutu) for im, _, _ in s])
    print('SONUC', ad, [round(b, 3) for _, _, b in s], flush=True)

g = Image.new('RGB', (6 * 150, len(satirlar) * 150), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((150, 150)), (c * 150, r * 150))
g.save('/content/ab_asama3.png')
print('SONUC grid')
