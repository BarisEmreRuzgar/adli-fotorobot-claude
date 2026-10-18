# Gemini planı: uygulama ve test günlüğü

Başlangıç: 2026-10-18. Her adım Colab T4'te aynı tohumla eski sürüme karşı ölçülür;
kazanç yoksa geri alınır. Deney betikleri Colab'da `/content/ab_*.py`, görselleri `/content/ab_*.png`.

| # | Adım | Durum | Sonuç |
|---|---|---|---|
| 1 | Aşama 3: ControlNet inpaint + en az 256 px kırpma | ❌ geri alındı | ControlNet yeni ayrıntıyı bastırıyor (0/6 vs 5/6) |
| 1b | Aşama 3: kısa, ağırlıklı ayrıntı tarifi (ControlNet yerine) | ✅ tutuldu (`deneme`) | Ben: ağırlık 1.35 + fırça 9 → 5/5 doğal; yara izi çözülmedi (en iyi 1/5) |
| 3 | Aşama 1: prompt'u anlamsal bloklara ayırma (BREAK) | ✅ tutuldu (`deneme`) | Cinsiyet tutarlılığı ve uzun saç iyileşti; renk (beyaz bıyık/sakal) çözülmedi |
| 3b | Erkeklik vurgusu her erkek yüzde | ✅ tutuldu (`deneme`) | Siyah saç + beyaz sakal 4/4 erkek (önce 1/4 kadın), ince bıyıklı genç 4/4 erkek |
| 5 | YZ tahmini uyarı bandı, negatif prompt, manifest.sha256 | ✅ kod hazır (`deneme` dalı) | Birim testleri geçiyor |
| 4 | Gemini önbelleği (JSON şema zorlaması gerekmedi) | ✅ kod hazır (`deneme` dalı) | Aynı ifade → aynı analiz, kota harcanmaz |
| 2 | Aşama 2: bölgesel maskeli düzenleme (MediaPipe yerine InsightFace noktaları) | ✅ tutuldu (`deneme`) | Kimlik +4/+7 puan, değişmeyen her şey birebir korunuyor |
| 2b | Saç rengi: bölgesel değişim gücü 0.9 vs 1.0 | ✅ 0.9 kaldı | 1.0 uzun saçta yüzü bozuyor; 0.9 uzun saçta 3/3 açık renk, kısa saçta gri |
| 2c | Saç maskesi omuz/kıyafeti kapsamasın | ✅ tutuldu (`deneme`) | Uzun saçta tişört 3/3 korundu (önce değişiyordu), saç 3/3 açık sarı, kimlik aynı |
| 1c | Aşama 3: 9 kanallı RV inpainting UNet'i | ❌ eklenmedi | Hiçbir şey çizmiyor: ben 0/5 (4 kanal 4/5), yara 0/10 |

## Ayrıntılar

### Adım 1: Aşama 3 küçük ben
Kaynak: vaka_20261018_041513 (genç erkek, Aşama 2 v1), maske ağız yanında ~24 px, tohum 20261018.
- Eski yöntem, dünkü tarif: maske içi fark 5.9/6.6/7.8 (dün ile birebir aynı → ölçüm güvenilir). Görselde ben yok.
- ControlNet + 256 px kutu, aynı tarif: 5.0/5.1/6.1. Görselde ben yok. → ControlNet tek başına çözmedi.
- Tarif deneyleri (ControlNet açık):
  - "prominent dark brown mole on the skin, RAW photo, close-up of human skin..." → ben yok
  - "a large dark brown mole, beauty mark on the cheek" (geniş fırça / CFG 11 dahil) → ben yok
  - **"(dark brown mole:1.6), (beauty spot:1.3), skin"**, yarıçap ~20 → 1/3 net ben, 2/3 soluk
  - "(large raised dark brown nevus:1.5) on the cheek" → 3/3 koyu iz ama lekemsi
- Sonuç: "close-up of human skin" gibi cilt tarifleri modeli düz cilde itiyor; ağırlıklı, kısa nesne tarifi gerekli.
- ControlNet açık/kapalı, aynı ağırlıklı tarif, 6 tohum: ControlNet'siz 5/6 iz, ControlNet'li 0/6 → ControlNet geri alındı.
- Ağırlık 1.6 ile (ControlNet'siz): ben r12 6/6 ama iri, yara gibi; yara izi 6/6 ama şeritli/dövme gibi yapay.
  Dünkü "çalışan" yara izi tarifi bu tohumda 0/6 → dünkü başarı şanslı tohummuş.

### Adım 1b: Aşama 3 ağırlık taraması (ControlNet'siz, 5 tohum)
- Ben: ağırlık 1.2 → 1-2/5; **1.35 + yarıçap 9 → 5/5 doğal ben**; 1.6 → iri, yara gibi. → 1.35, varsayılan fırça 9.
- Yara izi: her ağırlıkta (1.2/1.35/1.6) dövme, bant, şerit gibi yapay çizgiler. → düşük değişim gücüyle ayrı test.

### Adım 2: Bölgesel Aşama 2 (aynı kaynak, tohum 2026, 3 varyasyon; ArcFace kimlik)
| Komut | Genel (eski) | Bölgesel (yeni) | Görsel |
|---|---|---|---|
| sakalı temizle + saçları sarı yap | %63 / %61 / %60 | %57 / %70 / %70 | ikisinde de sakal gitti, saç sarı olmadı; bölgeselde gözlük/arka plan aynı |
| gözlüğü çıkar | %66 / %69 / %67 | **%70 / %76 / %78** | ikisinde de 3/3 kalktı; genelde yüz, ışık, tişört değişti, bölgeselde aynı |
| gözlük ekle | %75 / %76 / %76 | %81 / %57 / %84 | ikisinde 3/3 gözlük; bölgeselde 1 gözlük bozuk |
Düzeltme: gözlük çıkarılırken dokunulmayan küpe yüzünden kulaklar da maskeye giriyordu → yalnızca değişen aksesuarın bölgesi.

### Adım 3: Bloklu prompt (sabit özellik tablosu, Gemini yok, tohum 4242, 4 varyasyon)
| Vaka | Eski | Yeni (BREAK blokları) |
|---|---|---|
| Omuza kadar uzun saçlı genç erkek | 1/4 uzun saç | 2/4 |
| Kel, kara kaşlı, beyaz pala bıyıklı 60 yaş | bıyık 0/4 beyaz | bıyık 0/4 beyaz; 1 yüz kel değil (gerileme) |
| Siyah saç + beyaz sakal, 50 yaş | **3/4 kadın** çıktı | 1/4 kadın; sakal koyu |
Sonuç: tutuldu. Renk çelişkileri (beyaz bıyık/sakal) SD 1.5'te prompt'la çözülmüyor → Aşama 2/3'te bölgesel düzeltme gerekir.
Yan bulgu: sakal tarif edilse de kadın yüz çıkabiliyor → erkeklik vurgusu her erkek yüze genişletildi (3b).

### Adım 3b: Erkeklik vurgusu (tohum 4242, 4 varyasyon)
- Siyah saç + beyaz sakal, 50 yaş: 4/4 erkek, ikisinde gri sakal (önce 1/4 kadın, sakal koyu).
- İnce bıyıklı genç erkek: 4/4 erkek. Tutuldu.

### Adım 1b devamı: yara izi (5 tohum)
- Ağırlıklı tarifler (1.2/1.35/1.6): dövme, bant, şerit gibi yapay çizgiler.
- Negatife "tattoo, stripe, bandage" eklenince yapaylık gidiyor ama iz de neredeyse kayboluyor.
- En iyisi "pale raised scar tissue" 1.35, değişim 0.65: 1/5 gerçekçi. Yara izi SD 1.5 + 4 kanallı UNet ile güvenilir değil.

### Adım 2b: Saç rengi (sarı), bölgesel, tohum 2026, 3 varyasyon
| Yüz | Güç 0.9 | Güç 1.0 |
|---|---|---|
| Kısa, seyrek saçlı 50'li erkek | %86/%87/%84, saç griye döndü, sarı değil | %87/%84/%85, aynı; tişört siyahtan beyaza döndü |
| Omuza kadar saçlı genç erkek | %89/%87/%85, 3/3 açık kumral/sarı uç | %77/%76/%77, 2/3 yüz saç bölgesinde tekrarlanıp bozuldu |
Sonuç: 0.9 kaldı. Saç maskesi yüz ovalinin dışındaki her şeyi aldığı için kıyafet de değişiyordu → maske çenenin 0.3 yüz boyu altında kesildi (2c).

### Adım 1c: 9 kanallı inpainting UNet'i (Uminosachi/realisticVisionV51_v51VAE-inpainting), 5 tohum
| Deney | 4 kanal (mevcut) | 9 kanal |
|---|---|---|
| Ben, (dark brown mole, beauty spot:1.35), fırça 9 | 4/5 doğal ben | 0/5, cilt olduğu gibi |
| Yara, (pale raised scar tissue:1.35) | 5/5 dövme gibi yapay şerit | 0/5, düz cilt |
| Yara, "healed scar" (ağırlıksız) | 4/5 yapay iz | 0/5, düz cilt |
Sonuç: 9 kanallı model maskeli bölgeyi çevresine uydurmaya çok yatkın; küçük yeni ayrıntı üretmiyor. Ek ~1.7 GB indirmeye değmez.

### Adım 2c: Saç maskesinin alt sınırı
Maske çenenin 0.3 yüz boyu altında kesiliyor (resmin dışına taşarsa kesilmiyor; ilk sürüm bu durumda
PIL hatası verdi, deneyde yakalandı ve test eklendi). Aynı tohumla uzun saçlı yüz: %90/%87/%84
(önce %89/%87/%85), gri tişört 3/3 korundu, saç 3/3 açık sarı. Kısa saçlı yüzde yüz kadraja çok
yakın olduğu için kesim resmin dışında kalıyor, sonuç değişmedi.
