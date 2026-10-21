"""Gemini ile görsel denetim: istenen ayrıntılar görselde var mı, yüz gerçekçi mi?

denetle(gorsel, ["The man has a white beard", ...]) →
    {"kontroller": [true, false, ...], "gercekcilik": 1-5, "not": "..."}
Anahtar GOOGLE_API_KEY ortam değişkeninden (yoksa deneyler/anahtarlar.py ile masaüstündeki dosyadan).
"""
import io
import json
import os
import time

# Ücretsiz katmanda her modelin ayrı günlük kotası var (20 istek); sırayla harcanır
MODELLER = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
            "gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
_istemci = None
_kota_bitti = set()  # günlük kotası dolan modeller bu çalıştırmada bir daha denenmez


def _client():
    global _istemci
    if _istemci is None:
        from google import genai

        if not os.environ.get("GOOGLE_API_KEY"):
            try:
                import anahtarlar
                anahtarlar.ortama_yukle()
            except ImportError:
                pass
        _istemci = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return _istemci


SORU = """You are a strict forensic image auditor. Look at this generated face photo.
For each numbered claim, answer true only if it is clearly and unambiguously visible in the image.
Claims:
{iddialar}

Also rate photorealism from 1 (obviously fake, painted, plastic or deformed) to 5 (indistinguishable from a real photo).
Return JSON only: {{"kontroller": [true/false per claim in order], "gercekcilik": <1-5>, "not": "<one short sentence about the biggest flaw>"}}"""


def denetle(gorsel, iddialar, deneme=4):
    from google.genai import types

    tampon = io.BytesIO()
    gorsel.convert("RGB").save(tampon, format="JPEG", quality=92)
    parca = types.Part.from_bytes(data=tampon.getvalue(), mime_type="image/jpeg")
    metin = SORU.format(iddialar="\n".join(f"{i + 1}. {c}" for i, c in enumerate(iddialar)))
    son_hata = None
    for _ in range(deneme):
        for model in [m for m in MODELLER if m not in _kota_bitti]:
            try:
                r = _client().models.generate_content(
                    model=model, contents=[parca, metin],
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
                )
                sonuc = json.loads(r.text)
                sonuc["kontroller"] = [bool(x) for x in sonuc.get("kontroller", [])][:len(iddialar)]
                sonuc["model"] = model
                return sonuc
            except Exception as e:  # noqa: BLE001 - kota/yoğunluk: sıradaki modeli dene
                son_hata = e
                if "RESOURCE_EXHAUSTED" in str(e) or "NOT_FOUND" in str(e):
                    _kota_bitti.add(model)
        if len(_kota_bitti) == len(MODELLER):
            break
        time.sleep(10)
    return {"kontroller": [], "gercekcilik": None, "not": f"denetim başarısız: {son_hata}"}
