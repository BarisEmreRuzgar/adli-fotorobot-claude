"""Masaüstündeki apı.txt'den Gemini anahtarı ve HuggingFace token'ı (değerler yazdırılmaz)."""
import os

DOSYA = os.path.join(os.path.expanduser("~"), "Desktop", "apı.txt")


def oku():
    gemini = hf = ""
    if os.path.exists(DOSYA):
        for satir in open(DOSYA, encoding="utf-8-sig").read().split():
            satir = satir.strip()
            if satir.startswith("hf_"):
                hf = satir
            elif satir and not gemini:
                gemini = satir
    return gemini, hf


def ortama_yukle():
    gemini, hf = oku()
    if gemini:
        os.environ.setdefault("GOOGLE_API_KEY", gemini)
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
    return bool(gemini), bool(hf)
