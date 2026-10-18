"""Aşama 3 yara izi: değişim gücü / tarif / negatif taraması."""
import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)
from PIL import Image  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import agirlikli, bolge_promptu  # noqa: E402

NEG = bolge_promptu('x')[1] + ', tattoo, ink, tape, bandage, paint, stripes, sticker, flowers'
mo = Motor()
mo.yukle()
ol = json.load(open('ciktilar/vaka_20261018_035450/rapor.json'))
src = Image.open([o for o in ol if o['islem'] == 'asama1_uretim'][-1]['gorseller'][1]).convert('RGB')
m = Image.open([o for o in ol if o['islem'] == 'asama3_bolge_duzenleme'][-1]['maske']).convert('L')
x0, y0, x1, y1 = m.getbbox()
kutu = (max(0, (x0 + x1) // 2 - 55), max(0, (y0 + y1) // 2 - 55))
kutu = kutu + (kutu[0] + 110, kutu[1] + 110)
satirlar = [[src.crop(kutu)] * 5]
for tarif, w, guc in (('healed scar', 1.2, 0.5), ('healed scar', 1.2, 0.65), ('old healed scar on the skin', 1.2, 0.65),
                      ('pale raised scar tissue', 1.35, 0.65), ('healed scar', 1.35, 0.8)):
    poz = f"{agirlikli(tarif, w)}, photo, realistic"
    s = mo.bolge_duzenle(src, m, poz, NEG, adet=5, tohum=20261018, adim=25, cfg=7, guc=guc)
    satirlar.append([im.crop(kutu) for im, _, _ in s])
    print('SONUC yara', tarif, w, guc, flush=True)
g = Image.new('RGB', (5 * 130, len(satirlar) * 130), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((130, 130)), (c * 130, r * 130))
g.save('/content/ab_yara.png')
print('SONUC grid')
