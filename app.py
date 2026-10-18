"""Adli Fotorobot'u başlatır.

Kullanım:
    python app.py                 # yerel: http://127.0.0.1:7860
    python app.py --share         # herkese açık geçici Gradio bağlantısı
    python app.py --low-vram      # 4-6 GB GPU'lar için

Gemini anahtarı GOOGLE_API_KEY ortam değişkeninden (Colab'da Secrets'tan)
okunur ya da arayüzdeki Ayarlar bölümünden girilebilir. Anahtar yoksa
kural tabanlı Türkçe analiz kullanılır.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Adli Fotorobot")
    parser.add_argument("--share", action="store_true", help="Herkese açık Gradio bağlantısı oluştur")
    parser.add_argument("--low-vram", action="store_true", help="Düşük VRAM modunu zorla")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None, help="Ör. 0.0.0.0 (yerel ağdan erişim için)")
    args = parser.parse_args()

    # Windows konsolunda emoji/Türkçe karakterler için
    for akis in (sys.stdout, sys.stderr):
        try:
            akis.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    from fotorobot.arayuz import Uygulama
    from fotorobot.config import Ayarlar
    from fotorobot.motor import Motor

    ayarlar = Ayarlar()
    if args.share:
        ayarlar.paylas = True
    if args.low_vram:
        ayarlar.dusuk_vram = True
    if args.port:
        ayarlar.port = args.port

    motor = Motor(ayarlar)
    try:
        motor.yukle()
    except Exception as e:  # noqa: BLE001 - arayüz yine açılsın, hata gösterilsin
        import traceback
        traceback.print_exc()
        motor.durum_notlari.append(f"Model yükleme hatası: {str(e)[:200]}")

    uygulama = Uygulama(motor, ayarlar)
    demo = uygulama.olustur()
    demo.queue(default_concurrency_limit=1).launch(
        share=ayarlar.paylas,
        server_name=args.host,
        server_port=ayarlar.port,
        allowed_paths=[str(ayarlar.cikti_klasoru.resolve())],
        inbrowser=not ayarlar.paylas and not os.environ.get("FOTOROBOT_NO_BROWSER"),
        debug=ayarlar.paylas,
        show_error=True,
    )


if __name__ == "__main__":
    main()
