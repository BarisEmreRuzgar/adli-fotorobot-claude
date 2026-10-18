import importlib.util
import os
import sys

D = '/content/deneme/'
os.chdir(D)
sys.path.insert(0, D)
from PIL import Image  # noqa: E402

from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import normalize, prompt_olustur  # noqa: E402

spec = importlib.util.spec_from_file_location('eski', '/content/eski_ozellikler.py')
eski = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eski)
# Yalnızca blok etkisini ölçmek için eski prompt'a yeni negatifler de eklenir
EK = ', fashion model, airbrushed skin, studio glamour, beauty retouching'
vakalar = {
    'uzun_sac': {'cinsiyet': 'male', 'yas': 21, 'ten': 'fair pale skin',
                 'yuz_sekli': 'thin long face, prominent high cheekbones',
                 'sac': 'long straight black hair reaching the shoulders, parted in the middle',
                 'kas': 'thin straight eyebrows', 'goz': 'large light brown eyes',
                 'burun': 'thin aquiline nose', 'dudak': 'full lips', 'cene': 'narrow jawline',
                 'sakal': 'clean shaven'},
    'beyaz_biyik': {'cinsiyet': 'male', 'yas': 60, 'ten': 'tanned skin',
                    'yuz_sekli': 'round fleshy face, heavy build', 'sac': 'bald',
                    'kas': 'thick bushy black eyebrows', 'goz': 'dark brown eyes, under-eye bags',
                    'sakal': 'thick white handlebar mustache'},
    'siyah_sac_beyaz_sakal': {'cinsiyet': 'male', 'yas': 50, 'ten': 'olive skin',
                              'sac': 'short black hair', 'sakal': 'full white beard'},
}
mo = Motor()
mo.yukle()
satirlar = []
for ad, ham in vakalar.items():
    for etiket, fn in (('ESKI', eski.prompt_olustur), ('YENI', prompt_olustur)):
        oz = normalize(ham)
        poz, neg = fn(oz)
        if etiket == 'ESKI':
            neg += EK
        sonuc = mo.metinden_uret(poz, neg, adet=4, tohum=4242, adim=25, cfg=6.0)
        satirlar.append([im for im, _ in sonuc])
        print('SONUC', ad, etiket, flush=True)
g = Image.new('RGB', (4 * 160, len(satirlar) * 200), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((160, 200)), (c * 160, r * 200))
g.save('/content/ab_blok.png')
print('SONUC grid kaydedildi')
