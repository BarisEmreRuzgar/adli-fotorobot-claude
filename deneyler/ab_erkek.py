"""Erkeklik vurgusu: yalnızca sakalsız (eski) vs her erkek yüzde (yeni), sakallı vakalarda."""
import importlib.util
import os
import sys

D = '/content/deneme/'
os.chdir(D)
sys.path.insert(0, D)
from PIL import Image  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import normalize, prompt_olustur  # noqa: E402

spec = importlib.util.spec_from_file_location('eski', '/content/onceki_ozellikler.py')
eski = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eski)
vakalar = {
    'siyah_sac_beyaz_sakal': {'cinsiyet': 'male', 'yas': 50, 'ten': 'olive skin',
                              'sac': 'short black hair', 'sakal': 'full white beard'},
    'kivircik_kirli_sakal': {'cinsiyet': 'male', 'yas': 27, 'ten': 'tan skin', 'sac': 'short curly black hair',
                             'sakal': 'stubble', 'goz': 'hazel eyes', 'ozel': 'scar on right cheek'},
    'genc_biyikli': {'cinsiyet': 'male', 'yas': 24, 'ten': 'fair skin', 'sac': 'medium length wavy brown hair',
                     'sakal': 'thin mustache', 'dudak': 'full lips'},
}
mo = Motor()
mo.yukle()
satirlar = []
for ad, ham in vakalar.items():
    for etiket, fn in (('ESKI', eski.prompt_olustur), ('YENI', prompt_olustur)):
        poz, neg = fn(normalize(ham))
        sonuc = mo.metinden_uret(poz, neg, adet=4, tohum=4242, adim=25, cfg=6.0)
        satirlar.append([im for im, _ in sonuc])
        print('SONUC', ad, etiket, flush=True)
g = Image.new('RGB', (4 * 160, len(satirlar) * 200), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((160, 200)), (c * 160, r * 200))
g.save('/content/ab_erkek.png')
print('SONUC grid')
