"""Aşama 2 saç rengi: bölgesel düzenlemede değişim gücü 0.9 vs 1.0 (kısa, seyrek saçlı yüz)."""
import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)
from PIL import Image  # noqa: E402

import fotorobot.motor as motor_mod  # noqa: E402
from fotorobot.kural_analiz import degisiklik_uygula  # noqa: E402
from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import degisiklik_bolgeleri, normalize, prompt_olustur  # noqa: E402

ol = json.load(open('ciktilar/vaka_20261018_035450/rapor.json'))
o = [o for o in ol if o['islem'] == 'asama1_uretim'][-1]
src, oz = Image.open(o['gorseller'][1]).convert('RGB'), normalize(o['ozellikler'])
ol = json.load(open('ciktilar/vaka_20261018_041513/rapor.json'))
o = [o for o in ol if o['islem'] == 'asama1_uretim'][-1]
src2, oz2 = Image.open(o['gorseller'][1]).convert('RGB'), normalize(o['ozellikler'])
mo = Motor()
mo.yukle()
satirlar = []
for ad, s0, o0 in (('kisa_sac', src, oz), ('uzun_sac', src2, oz2)):
    yeni, _ = degisiklik_uygula(o0, 'saçları sarı yap')
    poz, neg = prompt_olustur(yeni, onceki=o0, referans=o0)
    b = degisiklik_bolgeleri(o0, yeni)
    for guc in (0.9,):
        motor_mod.BOLGE2_EN_AZ_DEGISIM = guc
        s, _ = mo.kimlik_koruyarak_uret(s0, poz, neg, adet=3, tohum=2026, adim=25, cfg=6.0, kimlik_gucu=0.6,
                                         degisim=0.8, kimlik_gorseli=s0, sac_maskele=True, bolgeler=b)
        satirlar.append([s0] + [im for im, _, _ in s])
        print('SONUC', ad, guc, sorted(b or []), [round(x or 0, 3) for _, _, x in s], flush=True)
g = Image.new('RGB', (4 * 160, len(satirlar) * 200), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((160, 200)), (c * 160, r * 200))
g.save('/content/ab_sac2.png')
print('SONUC grid')
