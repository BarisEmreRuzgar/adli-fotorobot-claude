"""Gemini ile tanık ifadesi analizi ve değişiklik yorumlama.

Her fonksiyon başarısız olursa (anahtar yok, kota doldu, ağ hatası) kural
tabanlı analizciye düşer; böylece uygulama Gemini olmadan da çalışır.
"""

import io
import json
import re
import time

from .kural_analiz import degisiklik_uygula, ifade_analiz as kural_ifade_analiz
from .ozellikler import ALANLAR, normalize

# Geçici sunucu hatalarında (503 yoğunluk vb.) aynı modelle deneme sayısı
GECICI_HATA_DENEME = 3

ALAN_ACIKLAMA = """\
- cinsiyet: "male" veya "female"
- yas: tahmini yaş, TAM SAYI (aralık verilmişse ortalaması)
- ten: ten rengi (ör. "olive skin", "fair pale skin", "tan brown skin")
- yuz_sekli: yüz şekli ve yanaklar (ör. "oval face", "round chubby face, high cheekbones")
- sac: uzunluk + doku + renk + stil (ör. "short curly black hair, receding hairline", "bald shaved head")
- kas: kaşlar (ör. "thick dark straight eyebrows")
- goz: boyut/şekil + renk (ör. "large almond-shaped hazel eyes")
- burun: (ör. "aquiline hooked nose", "small upturned nose")
- dudak: (ör. "thin lips", "full lips")
- cene: çene ve çene hattı (ör. "strong square jawline", "cleft chin")
- sakal: sakal ve bıyık (ör. "clean shaven", "stubble", "short trimmed beard and mustache"); kadınlarda ""
- aksesuar: gözlük, küpe, şapka, kapüşon vb. virgülle; yoksa ""
- ozel: yara izi, ben, dövme, çil gibi ayırt edici izler ve konumları; yoksa \"\""""

IFADE_TALIMATI = f"""Sen bir adli fotorobot uzmanısın. Tanık ifadesini analiz edip şüphelinin
yüz özelliklerini aşağıdaki JSON alanlarına doldur. Değerler Stable Diffusion
için kısa İNGİLİZCE ifadeler olmalı.

Alanlar:
{ALAN_ACIKLAMA}

Kurallar:
- İfadede geçen her ayrıntıyı ilgili alana mutlaka yansıt.
- İfadede geçmeyen alanlar için kişinin yaşına/cinsiyetine uygun, sıradan ve nötr bir değer yaz.
- Belirsiz ifadeleri ("galiba", "emin değilim") en olası değere çevir.
- Ayrıca "belirtilmeyen" alanına ifadede HİÇ geçmeyen alanların adlarını liste olarak yaz.
- Sadece JSON döndür."""

DEGISIKLIK_TALIMATI = f"""Sen bir adli fotorobot uzmanısın. Elinde bir şüphelinin mevcut yüz
özellikleri (JSON) ve kullanıcının Türkçe değişiklik isteği var. İsteği uygula ve
GÜNCELLENMİŞ JSON'un tamamını döndür.

Alanlar:
{ALAN_ACIKLAMA}

Kurallar:
- SADECE istenen değişiklikleri yap, diğer tüm alanları birebir koru.
- "yaşlandır" gibi göreli isteklerde makul bir miktar uygula (yaklaşık 10-15 yıl).
- Bir şeyin kaldırılması istendiyse ilgili değeri boşalt veya "clean shaven" yap.
- Ayrıca "uygulanan" alanına yaptığın değişiklikleri kısa Türkçe maddeler olarak listele.
- Sadece JSON döndür."""

GORSEL_TALIMATI = f"""Sen bir adli fotorobot uzmanısın. Fotoğraftaki kişinin yüz özelliklerini
aşağıdaki JSON alanlarına, Stable Diffusion için kısa İNGİLİZCE ifadelerle yaz.

Alanlar:
{ALAN_ACIKLAMA}

Sadece JSON döndür."""


def _json_ayikla(metin):
    metin = (metin or "").strip()
    metin = re.sub(r"^```(?:json)?\s*|\s*```$", "", metin)
    try:
        return json.loads(metin)
    except json.JSONDecodeError:
        eslesme = re.search(r"\{.*\}", metin, flags=re.DOTALL)
        if eslesme:
            return json.loads(eslesme.group(0))
        raise


def _hata_ozeti(mesaj):
    """API hata metninden kullanıcıya gösterilecek kısa neden."""
    m = mesaj.upper()
    if "429" in m or "RESOURCE_EXHAUSTED" in m or "QUOTA" in m:
        return "kota / istek sınırı doldu (429)"
    if "404" in m or "NOT_FOUND" in m:
        return "model bulunamadı (404), adı değişmiş olabilir"
    if "403" in m or "PERMISSION_DENIED" in m:
        return "bu modele erişim izni yok (403)"
    if "503" in m or "UNAVAILABLE" in m or "OVERLOADED" in m:
        return "model geçici olarak yoğun (503)"
    return mesaj[:120]


class Gemini:
    def __init__(self, api_anahtari="", modeller=("gemini-2.5-flash",)):
        self.modeller = list(modeller)
        self._istemci = None
        self.son_hata = ""
        self.atlanan_modeller = []
        self.anahtar_ayarla(api_anahtari)

    def anahtar_ayarla(self, api_anahtari):
        self.api_anahtari = (api_anahtari or "").strip()
        self._istemci = None

    @property
    def hazir(self):
        return bool(self.api_anahtari)

    def _client(self):
        if self._istemci is None:
            from google import genai
            self._istemci = genai.Client(api_key=self.api_anahtari)
        return self._istemci

    def _uret(self, icerik, sistem):
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=sistem,
            response_mime_type="application/json",
            temperature=0.2,
        )
        hatalar = []
        self.atlanan_modeller = []
        for model in self.modeller:
            for deneme in range(GECICI_HATA_DENEME):
                try:
                    yanit = self._client().models.generate_content(
                        model=model, contents=icerik, config=config,
                    )
                    return _json_ayikla(yanit.text), model
                except Exception as e:  # noqa: BLE001 - yeniden dene veya sonraki modele geç
                    mesaj = str(e)
                    # Anahtar geçersizse diğer modelleri denemenin anlamı yok
                    if "API_KEY_INVALID" in mesaj or "API key not valid" in mesaj:
                        raise RuntimeError(f"Geçersiz API anahtarı: {mesaj[:200]}") from e
                    gecici = any(k in mesaj for k in ("503", "UNAVAILABLE", "overloaded", "500 INTERNAL"))
                    if gecici and deneme + 1 < GECICI_HATA_DENEME:
                        time.sleep(2 * (deneme + 1))
                        continue
                    hatalar.append(f"{model}: {mesaj[:200]}")
                    # Yedek modele sessizce geçilmesin: neden atlandığı rapora ve konsola yazılır
                    self.atlanan_modeller.append(f"{model} → {_hata_ozeti(mesaj)}")
                    print(f"⚠️ Gemini modeli atlandı: {model}: {mesaj[:300]}", flush=True)
                    break
        raise RuntimeError(" | ".join(hatalar))

    def _atlanan_notu(self):
        return [f"ℹ️ Atlanan model: {a}" for a in self.atlanan_modeller]

    def baglanti_testi(self):
        if not self.hazir:
            return False, "API anahtarı girilmedi"
        try:
            _, model = self._uret(['{"durum": "ok"} döndür'], "Sadece JSON döndür.")
            return True, "\n".join([f"Gemini bağlantısı başarılı ({model})"] + self._atlanan_notu())
        except Exception as e:  # noqa: BLE001
            return False, f"Gemini bağlantı hatası: {e}"

    # ── Yüksek seviye işlemler ────────────────────────────────────────

    def ifade_analiz(self, metin):
        """Dönüş: (özellikler, rapor satırları, tanığın belirtmediği / tahmin edilen alanlar)."""
        if self.hazir:
            try:
                ham, model = self._uret([f"Tanık ifadesi:\n{metin}"], IFADE_TALIMATI)
                oz = normalize(ham)
                rapor = [f"✅ Gemini analizi ({model})"] + self._atlanan_notu()
                belirtilmeyen = [a for a in ham.get("belirtilmeyen", []) if a in ALANLAR] \
                    if isinstance(ham.get("belirtilmeyen"), list) else []
                if belirtilmeyen:
                    rapor.append("ℹ️ İfadede geçmeyen (tahmin edilen): " + ", ".join(belirtilmeyen))
                return oz, rapor, belirtilmeyen
            except Exception as e:  # noqa: BLE001
                self.son_hata = str(e)
                rapor = [f"⚠️ Gemini kullanılamadı, kural tabanlı analiz yapıldı.\n   ({str(e)[:300]})"]
        else:
            rapor = ["ℹ️ Gemini anahtarı yok, kural tabanlı Türkçe analiz kullanıldı."]
        oz, bulunan = kural_ifade_analiz(metin)
        eksik = [a for a in ALANLAR if a not in bulunan]
        if eksik:
            rapor.append("ℹ️ İfadede bulunamayan (varsayılan kullanılan): " + ", ".join(eksik))
        return oz, rapor, eksik

    def degisiklik(self, oz, metin):
        """Dönüş: (yeni özellikler, rapor satırları)."""
        if self.hazir:
            try:
                istek = (f"Mevcut özellikler:\n{json.dumps(oz, ensure_ascii=False)}\n\n"
                         f"Değişiklik isteği:\n{metin}")
                ham, model = self._uret([istek], DEGISIKLIK_TALIMATI)
                yeni = normalize(ham, temel=oz)
                uygulanan = ham.get("uygulanan") if isinstance(ham.get("uygulanan"), list) else []
                rapor = [f"✅ Gemini değişikliği yorumladı ({model})"] + self._atlanan_notu()
                rapor += [f"   • {u}" for u in uygulanan]
                return yeni, rapor
            except Exception as e:  # noqa: BLE001
                self.son_hata = str(e)
                rapor = [f"⚠️ Gemini kullanılamadı, kural tabanlı yorum yapıldı.\n   ({str(e)[:300]})"]
        else:
            rapor = ["ℹ️ Gemini anahtarı yok, kural tabanlı yorum kullanıldı."]
        yeni, uygulanan = degisiklik_uygula(oz, metin)
        if uygulanan:
            rapor += [f"   • {u}" for u in uygulanan]
        else:
            rapor.append("⚠️ Değişiklik isteği anlaşılamadı. Örnek: 'sakalı temizle', "
                         "'saçları sarı yap', 'gözlük ekle', 'daha yaşlı göster'")
        return yeni, rapor

    def gorsel_analiz(self, pil_gorsel, temel=None):
        """Yüklenen fotoğraftan özellik çıkarır. Gemini yoksa None döner."""
        if not self.hazir:
            return None, "ℹ️ Gemini anahtarı yok; fotoğraf özellikleri yüz analizinden tahmin edildi."
        try:
            from google.genai import types

            buf = io.BytesIO()
            pil_gorsel.convert("RGB").save(buf, format="JPEG", quality=92)
            parca = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
            ham, model = self._uret([parca, "Bu kişinin yüz özelliklerini çıkar."], GORSEL_TALIMATI)
            return normalize(ham, temel=temel), f"✅ Fotoğraf Gemini ile analiz edildi ({model})"
        except Exception as e:  # noqa: BLE001
            self.son_hata = str(e)
            return None, f"⚠️ Fotoğraf analizi yapılamadı ({str(e)[:200]})"
