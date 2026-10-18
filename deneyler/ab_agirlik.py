"""Aşama 3: ayrıntı ağırlığı ve fırça boyutu taraması (ben + yara izi)."""
import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)
from PIL import Image  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.arayuz import maske_ciz  # noqa: E402
from fotorobot.ozellikler import agirlikli, bolge_promptu  # noqa: E402

NEG = bolge_promptu('x')[1]
mo = Motor()
mo.yukle()
satirlar = []

ol = json.load(open('ciktilar/vaka_20261018_041513/rapor.json'))
src_ben = Image.open([o for o in ol if o['islem'] == 'asama2_duzenleme'][-1]['gorseller'][0]).convert('RGB')
ol = json.load(open('ciktilar/vaka_20261018_035450/rapor.json'))
src_yara = Image.open([o for o in ol if o['islem'] == 'asama1_uretim'][-1]['gorseller'][1]).convert('RGB')
m_yara = Image.open([o for o in ol if o['islem'] == 'asama3_bolge_duzenleme'][-1]['maske']).convert('L')
x0, y0, x1, y1 = m_yara.getbbox()
kutu_yara = (max(0, (x0 + x1) // 2 - 55), max(0, (y0 + y1) // 2 - 55))
kutu_yara = kutu_yara + (kutu_yara[0] + 110, kutu_yara[1] + 110)

for w in (1.2, 1.35):
    for r in (6, 9):
        m = maske_ciz(src_ben.size, [[383, 344]], r)
        poz = f"{agirlikli('dark brown mole, beauty spot', w)}, photo, realistic"
        s = mo.bolge_duzenle(src_ben, m, poz, NEG, adet=5, tohum=777, adim=25, cfg=8, guc=1.0)
        satirlar.append([im.crop((330, 300, 440, 410)) for im, _, _ in s])
        print('SONUC ben w', w, 'r', r, flush=True)
    for tarif in ('faint pale healed scar', 'thin white scar line'):
        poz = f"{agirlikli(tarif, w)}, photo, realistic"
        s = mo.bolge_duzenle(src_yara, m_yara, poz, NEG, adet=5, tohum=20261018, adim=25, cfg=8, guc=1.0)
        satirlar.append([im.crop(kutu_yara) for im, _, _ in s])
        print('SONUC yara w', w, tarif, flush=True)

g = Image.new('RGB', (5 * 130, len(satirlar) * 130), 'white')
for rr, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((130, 130)), (c * 130, rr * 130))
g.save('/content/ab_agirlik.png')
print('SONUC grid')
