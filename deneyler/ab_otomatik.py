"""Ben: Aşama 1 prompt'unda (eski) vs prompt'suz Aşama 1 + yüz noktalarından otomatik Aşama 3 (yeni)."""
import os
import sys

D = '/content/deneme/'
os.chdir(D)
sys.path.insert(0, D)
from PIL import Image  # noqa: E402

from fotorobot.arayuz import maske_ciz  # noqa: E402
from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import bolge_promptu, normalize, prompt_olustur  # noqa: E402


def sol_yanak(k):  # kişinin solu = görüntünün sağı
    d = k[1][0] - k[0][0]
    return [(k[1][0] + k[4][0]) / 2 + 0.25 * d, (k[1][1] + k[4][1]) / 2]


def sag_kas_ustu(k):
    d = k[1][0] - k[0][0]
    return [k[0][0], k[0][1] - 0.6 * d]


temel = {'cinsiyet': 'male', 'yas': 35, 'ten': 'fair skin', 'sac': 'short brown hair', 'sakal': 'clean shaven'}
vakalar = [('yanak', 'small dark mole on the left cheek', sol_yanak),
           ('kas', 'small dark mole above the right eyebrow', sag_kas_ustu)]
mo = Motor()
mo.yukle()
satirlar = []
for ad, ozel, nokta in vakalar:
    poz, neg = prompt_olustur(normalize(dict(temel, ozel=ozel)))
    satirlar.append([im for im, _ in mo.metinden_uret(poz, neg, adet=4, tohum=4242, adim=25, cfg=6.0)])
    print('SONUC', ad, 'ESKI', flush=True)
    poz, neg = prompt_olustur(normalize(temel))
    yeni = []
    bpoz, bneg = bolge_promptu('dark brown mole, beauty spot')
    for im, _ in mo.metinden_uret(poz, neg, adet=4, tohum=4242, adim=25, cfg=6.0):
        yuz = mo.yuz_bul(im)
        m = maske_ciz(im.size, [nokta(yuz.kps.tolist())], 9)
        yeni.append(mo.bolge_duzenle(im, m, bpoz, bneg, adet=1, tohum=777, adim=25, cfg=8.0)[0][0])
    satirlar.append(yeni)
    print('SONUC', ad, 'YENI', flush=True)
g = Image.new('RGB', (4 * 256, len(satirlar) * 320), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((256, 320)), (c * 256, r * 320))
g.save('/content/ab_otomatik.png')
print('SONUC grid')
