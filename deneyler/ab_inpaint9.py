"""Aşama 3: 4 kanallı UNet (mevcut) vs Realistic Vision 9 kanallı inpainting UNet'i."""
import json
import os
import sys

D = '/content/deneme/'
R = '/content/adli-fotorobot-claude/'
sys.path.insert(0, D)
os.chdir(R)
import torch  # noqa: E402
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel  # noqa: E402
from PIL import Image  # noqa: E402

from fotorobot.arayuz import maske_ciz  # noqa: E402
from fotorobot.motor import Motor  # noqa: E402
from fotorobot.ozellikler import agirlikli, bolge_promptu  # noqa: E402

mo = Motor()
mo.yukle()
eski_pipe = mo.pipe_inp
unet9 = UNet2DConditionModel.from_pretrained('Uminosachi/realisticVisionV51_v51VAE-inpainting', subfolder='unet',
                                             torch_dtype=torch.float16).to('cuda')
p = mo.pipe
yeni_pipe = StableDiffusionInpaintPipeline(vae=p.vae, text_encoder=p.text_encoder, tokenizer=p.tokenizer, unet=unet9,
                                           scheduler=p.scheduler, safety_checker=None, feature_extractor=None,
                                           requires_safety_checker=False)
yeni_pipe.set_progress_bar_config(disable=True)
NEG = bolge_promptu('x')[1]

ol = json.load(open('ciktilar/vaka_20261018_041513/rapor.json'))
src_ben = Image.open([o for o in ol if o['islem'] == 'asama2_duzenleme'][-1]['gorseller'][0]).convert('RGB')
m_ben = maske_ciz(src_ben.size, [[383, 344]], 9)
ol = json.load(open('ciktilar/vaka_20261018_035450/rapor.json'))
src_yara = Image.open([o for o in ol if o['islem'] == 'asama1_uretim'][-1]['gorseller'][1]).convert('RGB')
m_yara = Image.open([o for o in ol if o['islem'] == 'asama3_bolge_duzenleme'][-1]['maske']).convert('L')
x0, y0, x1, y1 = m_yara.getbbox()
ky = (max(0, (x0 + x1) // 2 - 55), max(0, (y0 + y1) // 2 - 55))
ky = ky + (ky[0] + 110, ky[1] + 110)

deneyler = [
    ('ben', src_ben, m_ben, agirlikli('dark brown mole, beauty spot', 1.35), (330, 300, 440, 410)),
    ('yara', src_yara, m_yara, agirlikli('pale raised scar tissue', 1.35), ky),
    ('yara_sade', src_yara, m_yara, 'healed scar', ky),
]
satirlar = []
for ad, src, m, detay, kutu in deneyler:
    for etiket, pipe in (('4kanal', eski_pipe), ('9kanal', yeni_pipe)):
        mo.pipe_inp = pipe
        s = mo.bolge_duzenle(src, m, f'{detay}, photo, realistic', NEG, adet=5, tohum=777, adim=25, cfg=7, guc=1.0)
        satirlar.append([im.crop(kutu) for im, _, _ in s])
        print('SONUC', ad, etiket, [round(b or 0, 3) for _, _, b in s], flush=True)
g = Image.new('RGB', (5 * 130, len(satirlar) * 130), 'white')
for r, sat in enumerate(satirlar):
    for c, im in enumerate(sat):
        g.paste(im.resize((130, 130)), (c * 130, r * 130))
g.save('/content/ab_inpaint9.png')
print('SONUC grid')
