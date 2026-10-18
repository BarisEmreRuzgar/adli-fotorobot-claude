"""Uygulama ayarları. Tüm değerler ortam değişkenleriyle değiştirilebilir."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(ad, varsayilan):
    return os.environ.get(ad, varsayilan)


def _env_bool(ad, varsayilan):
    deger = os.environ.get(ad)
    if deger is None:
        return varsayilan
    return deger.strip().lower() in ("1", "true", "evet", "yes", "on")


def colab_mi():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def api_anahtari_bul():
    """Gemini anahtarını sırayla ortam değişkenlerinde ve Colab Secrets'ta arar."""
    for ad in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        deger = os.environ.get(ad, "").strip()
        if deger:
            return deger
    if colab_mi():
        try:
            from google.colab import userdata
            for ad in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                try:
                    deger = (userdata.get(ad) or "").strip()
                except Exception:
                    deger = ""
                if deger:
                    return deger
        except ImportError:
            pass
    return ""


@dataclass
class Ayarlar:
    gemini_modelleri: list = field(default_factory=lambda: [
        m.strip() for m in _env(
            # "-latest" takma adı Google model emekliye ayırdıkça güncel modele geçer
            "FOTOROBOT_GEMINI_MODELS", "gemini-flash-latest,gemini-3.8-flash,gemini-flash-lite-latest"
        ).split(",") if m.strip()
    ])
    sd_model: str = _env("FOTOROBOT_SD_MODEL", "SG161222/Realistic_Vision_V5.1_noVAE")
    sd_vae: str = _env("FOTOROBOT_SD_VAE", "stabilityai/sd-vae-ft-mse")
    # Yalnızca görüntü kısmını içeren CLIP ViT-H/14 (laion2B) kopyası; tam
    # laion deposundan ~1.4 GB daha küçük.
    image_encoder: str = _env("FOTOROBOT_IMAGE_ENCODER", "h94/IP-Adapter")
    image_encoder_klasor: str = _env("FOTOROBOT_IMAGE_ENCODER_SUBFOLDER", "models/image_encoder")
    ip_repo: str = _env("FOTOROBOT_IP_REPO", "h94/IP-Adapter-FaceID")
    ip_agirlik: str = _env("FOTOROBOT_IP_WEIGHT", "ip-adapter-faceid-plusv2_sd15.bin")
    ip_lora: str = _env("FOTOROBOT_IP_LORA", "ip-adapter-faceid-plusv2_sd15_lora.safetensors")
    cikti_klasoru: Path = Path(_env("FOTOROBOT_OUTPUT_DIR", "ciktilar"))
    # Aynı ifade/komut için Gemini yanıtını diskten kullan (kota + yeniden üretilebilirlik)
    gemini_onbellek: bool = _env_bool("FOTOROBOT_GEMINI_CACHE", True)
    # None: GPU belleğine göre otomatik karar verilir.
    dusuk_vram: object = None
    paylas: bool = _env_bool("FOTOROBOT_SHARE", colab_mi())
    port: int = int(_env("FOTOROBOT_PORT", "7860"))

    def __post_init__(self):
        if "FOTOROBOT_LOW_VRAM" in os.environ:
            self.dusuk_vram = _env_bool("FOTOROBOT_LOW_VRAM", False)
        self.cikti_klasoru = Path(self.cikti_klasoru)
