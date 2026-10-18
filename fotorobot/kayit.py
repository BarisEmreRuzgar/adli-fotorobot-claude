"""Vaka kayıtları: üretilen her görsel ve işlem, izlenebilirlik için diske yazılır."""

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo


def yeni_vaka_id():
    return datetime.now().strftime("vaka_%Y%m%d_%H%M%S")


class VakaKaydi:
    def __init__(self, kok, vaka_id=None):
        self.vaka_id = vaka_id or yeni_vaka_id()
        self.klasor = Path(kok) / self.vaka_id
        self.klasor.mkdir(parents=True, exist_ok=True)
        self.rapor_yolu = self.klasor / "rapor.json"
        self._sayac = len(list(self.klasor.glob("*.png")))

    def gorsel_kaydet(self, gorsel, etiket, bilgi):
        self._sayac += 1
        guvenli = re.sub(r"[^\w-]+", "_", etiket)[:40]
        yol = self.klasor / f"{self._sayac:03d}_{guvenli}.png"
        meta = PngInfo()
        for k, v in bilgi.items():
            meta.add_text(str(k), v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        gorsel.save(yol, pnginfo=meta)
        return str(yol)

    def olay_ekle(self, tur, veri):
        olaylar = []
        if self.rapor_yolu.exists():
            try:
                olaylar = json.loads(self.rapor_yolu.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                olaylar = []
        olaylar.append({"zaman": datetime.now().isoformat(timespec="seconds"), "islem": tur, **veri})
        self.rapor_yolu.write_text(json.dumps(olaylar, ensure_ascii=False, indent=2), encoding="utf-8")

    def zip_olustur(self):
        hedef = self.klasor.parent / f"{self.vaka_id}.zip"
        with zipfile.ZipFile(hedef, "w", zipfile.ZIP_DEFLATED) as z:
            for dosya in sorted(self.klasor.iterdir()):
                z.write(dosya, arcname=f"{self.vaka_id}/{dosya.name}")
        return str(hedef)


def gorsel_ac(yol_veya_gorsel):
    if yol_veya_gorsel is None:
        return None
    if isinstance(yol_veya_gorsel, Image.Image):
        return yol_veya_gorsel.convert("RGB")
    if isinstance(yol_veya_gorsel, (str, Path)):
        return Image.open(yol_veya_gorsel).convert("RGB")
    try:
        import numpy as np
        if isinstance(yol_veya_gorsel, np.ndarray):
            return Image.fromarray(yol_veya_gorsel).convert("RGB")
    except ImportError:
        pass
    return None
