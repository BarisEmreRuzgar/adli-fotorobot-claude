"""FLUX.1-dev (4-bit NF4) ile Aşama 1: aynı adli senaryolarda prompt uyumu.

SD 1.5'in CLIP kodlayıcısı sıfatları karıştırıyor ("siyah saç + beyaz sakal" → koyu sakal).
FLUX'un T5 kodlayıcısı uzun doğal dil cümlelerini anlıyor; renk/iz ayrımını tutturuyor mu?
Colab T4'te: python deneyler/flux_test.py [ad]
"""
import gc
import json
import os
import sys
import time

KOK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KOK)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(KOK)

from benchmark_and_test import SENARYOLAR, oz_cikar  # noqa: E402
from olcum import izgara  # noqa: E402

FLUX_MODELI = os.environ.get("FLUX_MODEL", "diffusers/FLUX.1-dev-bnb-4bit")
IDDIALAR = {
    "beyaz_sakal": ["The man has a white or very light grey full beard (not dark)",
                    "The mustache is white or very light grey (not dark)", "The man has short black hair"],
    "sari_kisa": ["The man has short blonde hair"],
    "yara": ["There is a clearly visible healed scar on the person's left cheek (right side of the image)"],
    "pala": ["The man has a thick white or very light grey mustache (not dark)", "The man is bald"],
}


def flux_promptu(oz):
    """Özellik tablosundan FLUX için doğal dil tarif."""
    kadin = oz["cinsiyet"] == "female"
    kisi = f"{oz['yas']}-year-old {'woman' if kadin else 'man'}"
    cumleler = [f"A frontal passport-style ID photo of an ordinary {kisi}, looking straight at the camera "
                f"with a neutral expression."]
    if "bald" in oz["sac"]:
        cumleler.append("He is completely bald with a shaved head." if not kadin else "She has a shaved head.")
    else:
        cumleler.append(f"{'She' if kadin else 'He'} has {oz['sac']}.")
    if not kadin:
        s = oz["sakal"] or "clean shaven"
        cumleler.append("He is clean-shaven." if "clean" in s else f"He has a {s}.")
    cumleler.append(f"{'Her' if kadin else 'His'} skin is {oz['ten'].replace(' skin', '')}, "
                    f"with a {oz['yuz_sekli']}, {oz['kas']}, {oz['goz']}, a {oz['burun']}, {oz['dudak']} "
                    f"and a {oz['cene']}.")
    if oz["ozel"]:
        cumleler.append(f"Distinctive mark: {oz['ozel']}, clearly visible.")
    if oz["aksesuar"]:
        cumleler.append(f"{'She' if kadin else 'He'} is wearing {oz['aksesuar']}.")
    cumleler.append("Plain dark crew-neck t-shirt, plain light grey background, even soft studio lighting, "
                    "sharp focus, natural unretouched skin texture with pores and imperfections, realistic photo.")
    return " ".join(cumleler)


def main(ad="flux"):
    import torch
    from diffusers import FluxPipeline

    from vlm import denetle

    cikti = os.path.join(KOK, "deneyler", "sonuc")
    os.makedirs(cikti, exist_ok=True)
    dtype = torch.bfloat16 if os.environ.get("FLUX_DTYPE", "bf16") == "bf16" else torch.float16
    t0 = time.time()
    # Colab'ın 12 GB RAM'inde CPU offload dönüştürücü + T5'i (~13 GB) aynı anda RAM'de tutmaya
    # çalışıp süreci öldürüyordu. Bunun yerine her şey GPU'da: önce tüm prompt'lar T5 ile
    # kodlanır, sonra T5 silinir ve üretim yalnızca dönüştürücüyle yapılır.
    pipe = FluxPipeline.from_pretrained(FLUX_MODELI, torch_dtype=dtype)
    pipe.to("cuda")
    print(f"SONUC yuklendi {time.time() - t0:.0f} sn", flush=True)
    promptlar = {a: flux_promptu(oz_cikar(i)) for a, i in SENARYOLAR.items()}
    gomuler = {}
    with torch.no_grad():
        for anahtar, prompt in promptlar.items():
            print("PROMPT", anahtar, prompt, flush=True)
            pe, ppe, _ = pipe.encode_prompt(prompt=prompt, prompt_2=None, device="cuda", max_sequence_length=256)
            gomuler[anahtar] = (pe, ppe)
    pipe.text_encoder_2 = None
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    rapor, satirlar, etiketler = {}, [], []
    for anahtar in SENARYOLAR:
        pe, ppe = gomuler[anahtar]
        sat, et, sonuc = [], [], []
        for i in range(4):
            t0 = time.time()
            im = pipe(prompt_embeds=pe, pooled_prompt_embeds=ppe, width=640, height=800, guidance_scale=3.5,
                      num_inference_steps=24,
                      generator=torch.Generator("cpu").manual_seed(4242 + i * 1000)).images[0]
            sure = time.time() - t0
            im.save(os.path.join(cikti, f"{ad}_{anahtar}_{i}.png"))
            v = denetle(im, IDDIALAR[anahtar])
            ok = bool(v["kontroller"]) and all(v["kontroller"])
            sonuc.append({"ok": ok, "kontroller": v["kontroller"], "gercekcilik": v.get("gercekcilik"),
                          "not": v.get("not"), "sure": round(sure, 1)})
            sat.append(im)
            et.append(f"{'OK' if ok else 'X'} {v['kontroller']} r{v.get('gercekcilik')} {sure:.0f}s")
            print("SONUC", anahtar, i, json.dumps(sonuc[-1], ensure_ascii=False), flush=True)
        rapor[anahtar] = sonuc
        satirlar.append(sat)
        etiketler.append(et)
        print("PUAN", anahtar, f"{sum(s['ok'] for s in sonuc)}/4",
              "gercekcilik", [s["gercekcilik"] for s in sonuc], flush=True)
    izgara(satirlar, etiketler).save(os.path.join(cikti, f"{ad}.png"))
    with open(os.path.join(cikti, f"{ad}.json"), "w", encoding="utf-8") as f:
        json.dump(rapor, f, ensure_ascii=False, indent=1)
    print("SONUC bitti", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
